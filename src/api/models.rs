use super::state::ApiState;

use anyhow::Context as _;
use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use sha2::Digest as _;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, LazyLock};

#[derive(Serialize, Clone, Debug)]
pub(super) struct ModelInfo {
    /// Full routing string (e.g. "openrouter/anthropic/claude-sonnet-4")
    id: String,
    /// Human-readable name
    name: String,
    /// Provider ID for routing ("anthropic", "openrouter", "openai", etc.)
    provider: String,
    /// Context window size in tokens, if known
    context_window: Option<u64>,
    /// Whether this model supports tool/function calling
    tool_call: bool,
    /// Whether this model has reasoning/thinking capability
    reasoning: bool,
}

#[derive(Serialize)]
pub(super) struct ModelsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Deserialize)]
pub(super) struct ModelsQuery {
    provider: Option<String>,
}

#[derive(Deserialize)]
struct ModelsDevProvider {
    #[allow(dead_code)]
    id: Option<String>,
    #[allow(dead_code)]
    name: Option<String>,
    #[serde(default)]
    models: HashMap<String, ModelsDevModel>,
}

#[derive(Deserialize)]
struct ModelsDevModel {
    #[allow(dead_code)]
    id: Option<String>,
    name: String,
    #[serde(default)]
    tool_call: bool,
    #[serde(default)]
    reasoning: bool,
    limit: Option<ModelsDevLimit>,
    modalities: Option<ModelsDevModalities>,
    status: Option<String>,
}

#[derive(Deserialize)]
struct ModelsDevLimit {
    context: u64,
}

#[derive(Deserialize)]
struct ModelsDevModalities {
    #[allow(dead_code)]
    input: Option<Vec<String>>,
    output: Option<Vec<String>>,
}

#[derive(Clone, Debug)]
struct CachedGoogleAntigravityModels {
    key_hash: String,
    models: Vec<ModelInfo>,
    fetched_at: std::time::Instant,
}

#[derive(Debug, Deserialize)]
struct GoogleAntigravityApiKeyPayload {
    #[serde(default, alias = "accessToken", alias = "access_token")]
    token: String,
    #[serde(default, alias = "project_id")]
    #[serde(rename = "projectId")]
    project_id: String,
    #[serde(default, alias = "refresh_token")]
    #[serde(rename = "refreshToken")]
    refresh_token: String,
    #[serde(default, alias = "expires_at")]
    #[serde(rename = "expiresAt")]
    expires_at: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct GoogleAntigravityFetchModelsResponse {
    #[serde(default)]
    models: HashMap<String, GoogleAntigravityModel>,
    #[serde(default)]
    #[serde(rename = "defaultAgentModelId")]
    default_agent_model_id: Option<String>,
    #[serde(default)]
    #[serde(rename = "deprecatedModelIds")]
    deprecated_model_ids: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct GoogleAntigravityModel {
    #[serde(default)]
    #[serde(rename = "displayName")]
    display_name: Option<String>,
    #[serde(default)]
    #[serde(rename = "supportsThinking")]
    supports_thinking: bool,
    #[serde(default)]
    #[serde(rename = "maxTokens")]
    max_tokens: Option<u64>,
    #[serde(default)]
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: Option<u64>,
    #[serde(default)]
    #[serde(rename = "isInternal")]
    is_internal: bool,
}

/// Cached model catalog fetched from models.dev.
static MODELS_CACHE: std::sync::LazyLock<
    tokio::sync::RwLock<(Vec<ModelInfo>, std::time::Instant)>,
> = std::sync::LazyLock::new(|| tokio::sync::RwLock::new((Vec::new(), std::time::Instant::now())));

static GOOGLE_ANTIGRAVITY_MODELS_CACHE: std::sync::LazyLock<
    tokio::sync::RwLock<Option<CachedGoogleAntigravityModels>>,
> = std::sync::LazyLock::new(|| tokio::sync::RwLock::new(None));

const MODELS_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(3600);
const GOOGLE_ANTIGRAVITY_MODELS_CACHE_TTL: std::time::Duration =
    std::time::Duration::from_secs(300);
const GOOGLE_ANTIGRAVITY_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
static GOOGLE_ANTIGRAVITY_CLIENT_ID: LazyLock<String> =
    LazyLock::new(|| std::env::var("GOOGLE_ANTIGRAVITY_CLIENT_ID").unwrap_or_default());
static GOOGLE_ANTIGRAVITY_CLIENT_SECRET: LazyLock<String> =
    LazyLock::new(|| std::env::var("GOOGLE_ANTIGRAVITY_CLIENT_SECRET").unwrap_or_default());
const GOOGLE_ANTIGRAVITY_DEFAULT_PROJECT_ID: &str = "rising-fact-p41fc";
const GOOGLE_ANTIGRAVITY_VERSION_FALLBACK: &str = "1.18.3";

/// Maps models.dev provider IDs to spacebot's internal provider IDs for
/// providers with direct integrations.
fn direct_provider_mapping(models_dev_id: &str) -> Option<&'static str> {
    match models_dev_id {
        "anthropic" => Some("anthropic"),
        "openai" => Some("openai"),
        "deepseek" => Some("deepseek"),
        "xai" => Some("xai"),
        "mistral" => Some("mistral"),
        "groq" => Some("groq"),
        "togetherai" => Some("together"),
        "fireworks-ai" => Some("fireworks"),
        "zhipuai" => Some("zhipu"),
        _ => None,
    }
}

/// Models from providers not in models.dev (private/custom endpoints).
fn extra_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "opencode-zen/kimi-k2.5".into(),
            name: "Kimi K2.5".into(),
            provider: "opencode-zen".into(),
            context_window: None,
            tool_call: true,
            reasoning: true,
        },
        ModelInfo {
            id: "opencode-zen/kimi-k2".into(),
            name: "Kimi K2".into(),
            provider: "opencode-zen".into(),
            context_window: None,
            tool_call: true,
            reasoning: false,
        },
        ModelInfo {
            id: "opencode-zen/kimi-k2-thinking".into(),
            name: "Kimi K2 Thinking".into(),
            provider: "opencode-zen".into(),
            context_window: None,
            tool_call: true,
            reasoning: true,
        },
        ModelInfo {
            id: "opencode-zen/glm-5".into(),
            name: "GLM 5".into(),
            provider: "opencode-zen".into(),
            context_window: None,
            tool_call: true,
            reasoning: false,
        },
        ModelInfo {
            id: "opencode-zen/minimax-m2.5".into(),
            name: "MiniMax M2.5".into(),
            provider: "opencode-zen".into(),
            context_window: None,
            tool_call: true,
            reasoning: false,
        },
        ModelInfo {
            id: "opencode-zen/qwen3-coder".into(),
            name: "Qwen3 Coder 480B".into(),
            provider: "opencode-zen".into(),
            context_window: None,
            tool_call: true,
            reasoning: false,
        },
        ModelInfo {
            id: "opencode-zen/big-pickle".into(),
            name: "Big Pickle".into(),
            provider: "opencode-zen".into(),
            context_window: None,
            tool_call: true,
            reasoning: false,
        },
        // Z.AI Coding Plan
        ModelInfo {
            id: "zai-coding-plan/glm-4.7".into(),
            name: "GLM 4.7 (Coding)".into(),
            provider: "zai-coding-plan".into(),
            context_window: None,
            tool_call: true,
            reasoning: false,
        },
        ModelInfo {
            id: "zai-coding-plan/glm-5".into(),
            name: "GLM 5 (Coding)".into(),
            provider: "zai-coding-plan".into(),
            context_window: None,
            tool_call: true,
            reasoning: false,
        },
        ModelInfo {
            id: "zai-coding-plan/glm-4.5-air".into(),
            name: "GLM 4.5 Air (Coding)".into(),
            provider: "zai-coding-plan".into(),
            context_window: None,
            tool_call: true,
            reasoning: false,
        },
        // MiniMax
        ModelInfo {
            id: "minimax/MiniMax-M1-80k".into(),
            name: "MiniMax M1 80K".into(),
            provider: "minimax".into(),
            context_window: Some(80000),
            tool_call: true,
            reasoning: false,
        },
        // Moonshot AI (Kimi)
        ModelInfo {
            id: "moonshot/kimi-k2.5".into(),
            name: "Kimi K2.5".into(),
            provider: "moonshot".into(),
            context_window: None,
            tool_call: true,
            reasoning: true,
        },
        ModelInfo {
            id: "moonshot/moonshot-v1-8k".into(),
            name: "Moonshot V1 8K".into(),
            provider: "moonshot".into(),
            context_window: Some(8000),
            tool_call: false,
            reasoning: false,
        },
    ]
}

/// Fetch the full model catalog from models.dev and transform into ModelInfo entries.
async fn fetch_models_dev() -> anyhow::Result<Vec<ModelInfo>> {
    let client = reqwest::Client::new();
    let response = client
        .get("https://models.dev/api.json")
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await?
        .error_for_status()?;

    let catalog: HashMap<String, ModelsDevProvider> = response.json().await?;
    let mut models = Vec::new();

    for (provider_id, provider) in &catalog {
        for (model_id, model) in &provider.models {
            if model.status.as_deref() == Some("deprecated") {
                continue;
            }

            let has_text_output = model
                .modalities
                .as_ref()
                .and_then(|m| m.output.as_ref())
                .is_some_and(|outputs| outputs.iter().any(|o| o == "text"));
            if !has_text_output {
                continue;
            }

            let (routing_id, routing_provider) =
                if let Some(spacebot_provider) = direct_provider_mapping(provider_id) {
                    (
                        format!("{spacebot_provider}/{model_id}"),
                        spacebot_provider.to_string(),
                    )
                } else if provider_id == "openrouter" {
                    (format!("openrouter/{model_id}"), "openrouter".into())
                } else {
                    (
                        format!("openrouter/{provider_id}/{model_id}"),
                        "openrouter".into(),
                    )
                };

            let context_window = model.limit.as_ref().map(|l| l.context);

            models.push(ModelInfo {
                id: routing_id,
                name: model.name.clone(),
                provider: routing_provider,
                context_window,
                tool_call: model.tool_call,
                reasoning: model.reasoning,
            });
        }
    }

    models.sort_by(|a, b| a.provider.cmp(&b.provider).then(a.name.cmp(&b.name)));

    Ok(models)
}

/// Ensure the cache is populated (fetches on first call, then uses TTL).
async fn ensure_models_cache() -> Vec<ModelInfo> {
    {
        let cache = MODELS_CACHE.read().await;
        if !cache.0.is_empty() && cache.1.elapsed() < MODELS_CACHE_TTL {
            return cache.0.clone();
        }
    }

    match fetch_models_dev().await {
        Ok(models) => {
            let mut cache = MODELS_CACHE.write().await;
            *cache = (models.clone(), std::time::Instant::now());
            models
        }
        Err(error) => {
            tracing::warn!(%error, "failed to fetch models from models.dev, using stale cache");
            let cache = MODELS_CACHE.read().await;
            cache.0.clone()
        }
    }
}

fn antigravity_models_cache_key(raw_api_key: &str) -> String {
    let digest = sha2::Sha256::digest(raw_api_key.as_bytes());
    format!("{digest:x}")
}

fn normalize_google_antigravity_expires_at_ms(expires_at: Option<i64>) -> Option<i64> {
    expires_at.map(|value| {
        if value < 10_000_000_000 {
            value.saturating_mul(1000)
        } else {
            value
        }
    })
}

fn parse_google_antigravity_api_key_payload(raw_api_key: &str) -> GoogleAntigravityApiKeyPayload {
    let trimmed = raw_api_key.trim();
    if trimmed.is_empty() {
        return GoogleAntigravityApiKeyPayload {
            token: String::new(),
            project_id: GOOGLE_ANTIGRAVITY_DEFAULT_PROJECT_ID.to_string(),
            refresh_token: String::new(),
            expires_at: None,
        };
    }

    if let Ok(mut payload) = serde_json::from_str::<GoogleAntigravityApiKeyPayload>(trimmed) {
        if payload.project_id.trim().is_empty() {
            payload.project_id = GOOGLE_ANTIGRAVITY_DEFAULT_PROJECT_ID.to_string();
        }
        return payload;
    }

    GoogleAntigravityApiKeyPayload {
        token: trimmed.to_string(),
        project_id: GOOGLE_ANTIGRAVITY_DEFAULT_PROJECT_ID.to_string(),
        refresh_token: String::new(),
        expires_at: None,
    }
}

fn resolve_toml_string_value(doc: &toml_edit::DocumentMut, key: &str) -> Option<String> {
    let value = doc
        .get("llm")
        .and_then(|llm| llm.get(key))
        .and_then(|value| value.as_str())?;

    if let Some(variable_name) = value.strip_prefix("env:") {
        return std::env::var(variable_name).ok();
    }

    if value.trim().is_empty() {
        return None;
    }

    Some(value.to_string())
}

async fn google_antigravity_api_key_from_config(config_path: &Path) -> Option<String> {
    let config_api_key = match tokio::fs::read_to_string(config_path).await {
        Ok(content) => content
            .parse::<toml_edit::DocumentMut>()
            .ok()
            .and_then(|doc| resolve_toml_string_value(&doc, "google_antigravity_key")),
        Err(_) => None,
    };

    config_api_key.or_else(|| std::env::var("GOOGLE_ANTIGRAVITY_API_KEY").ok())
}

async fn google_antigravity_access_token(
    client: &reqwest::Client,
    credentials: &GoogleAntigravityApiKeyPayload,
) -> anyhow::Result<String> {
    let now_ms = chrono::Utc::now().timestamp_millis();
    let expires_at_ms = normalize_google_antigravity_expires_at_ms(credentials.expires_at);
    let inline_token = credentials.token.trim();

    if !inline_token.is_empty() {
        if let Some(expires_at_ms) = expires_at_ms {
            if expires_at_ms > now_ms + 60_000 {
                return Ok(inline_token.to_string());
            }
        } else {
            return Ok(inline_token.to_string());
        }
    }

    if credentials.refresh_token.trim().is_empty() {
        anyhow::bail!(
            "Google Antigravity credentials are missing both a valid access token and refresh token"
        )
    }
    if GOOGLE_ANTIGRAVITY_CLIENT_ID.is_empty() || GOOGLE_ANTIGRAVITY_CLIENT_SECRET.is_empty() {
        anyhow::bail!(
            "Google Antigravity OAuth client credentials are not configured. \
Set GOOGLE_ANTIGRAVITY_CLIENT_ID and GOOGLE_ANTIGRAVITY_CLIENT_SECRET."
        )
    }

    let token_response = client
        .post(GOOGLE_ANTIGRAVITY_TOKEN_URL)
        .header("content-type", "application/x-www-form-urlencoded")
        .form(&[
            ("client_id", GOOGLE_ANTIGRAVITY_CLIENT_ID.as_str()),
            ("client_secret", GOOGLE_ANTIGRAVITY_CLIENT_SECRET.as_str()),
            ("refresh_token", credentials.refresh_token.as_str()),
            ("grant_type", "refresh_token"),
        ])
        .send()
        .await
        .context("Google Antigravity token refresh request failed")?;

    let status = token_response.status();
    let response_text = token_response
        .text()
        .await
        .context("failed to read Google Antigravity token refresh response")?;
    let response_body: serde_json::Value = serde_json::from_str(&response_text)
        .context("Google Antigravity token refresh response is not valid JSON")?;

    if !status.is_success() {
        let error_message = response_body["error_description"]
            .as_str()
            .or_else(|| response_body["error"]["message"].as_str())
            .or_else(|| response_body["error"].as_str())
            .unwrap_or("unknown error");
        anyhow::bail!("Google Antigravity token refresh failed ({status}): {error_message}");
    }

    let refreshed_token = response_body["access_token"]
        .as_str()
        .unwrap_or("")
        .trim()
        .to_string();
    if refreshed_token.is_empty() {
        anyhow::bail!("Google Antigravity token refresh succeeded but returned no access_token");
    }

    Ok(refreshed_token)
}

fn humanize_google_antigravity_model_id(model_id: &str) -> String {
    model_id
        .replace(['-', '_'], " ")
        .split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => format!("{}{}", first.to_ascii_uppercase(), chars.as_str()),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn should_include_google_antigravity_model(
    model_id: &str,
    model: &GoogleAntigravityModel,
    deprecated_model_ids: &HashSet<String>,
) -> bool {
    if deprecated_model_ids.contains(model_id) {
        return false;
    }
    if model.is_internal {
        return false;
    }
    if model_id.starts_with("chat_") || model_id.starts_with("tab_") {
        return false;
    }
    true
}

async fn fetch_google_antigravity_models_with_api_key(
    raw_api_key: &str,
) -> anyhow::Result<Vec<ModelInfo>> {
    let credentials = parse_google_antigravity_api_key_payload(raw_api_key);
    let client = reqwest::Client::new();
    let access_token = google_antigravity_access_token(&client, &credentials).await?;
    let project_id = if credentials.project_id.trim().is_empty() {
        GOOGLE_ANTIGRAVITY_DEFAULT_PROJECT_ID.to_string()
    } else {
        credentials.project_id.trim().to_string()
    };

    let endpoints = [
        "https://cloudcode-pa.googleapis.com",
        "https://daily-cloudcode-pa.sandbox.googleapis.com",
    ];
    let mut errors = Vec::new();

    for endpoint in endpoints {
        let response = match client
            .post(format!("{endpoint}/v1internal:fetchAvailableModels"))
            .header("authorization", format!("Bearer {access_token}"))
            .header("content-type", "application/json")
            .header(
                "user-agent",
                format!("antigravity/{GOOGLE_ANTIGRAVITY_VERSION_FALLBACK} darwin/arm64"),
            )
            .json(&serde_json::json!({ "project": project_id }))
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                errors.push(format!(
                    "fetchAvailableModels request failed at {endpoint}: {error}"
                ));
                continue;
            }
        };

        let status = response.status();
        let response_text = match response.text().await {
            Ok(text) => text,
            Err(error) => {
                errors.push(format!(
                    "failed to read fetchAvailableModels response body at {endpoint}: {error}"
                ));
                continue;
            }
        };

        if !status.is_success() {
            errors.push(format!(
                "fetchAvailableModels failed ({status}) at {endpoint}: {}",
                response_text.chars().take(200).collect::<String>()
            ));
            continue;
        }

        let body: GoogleAntigravityFetchModelsResponse = match serde_json::from_str(&response_text)
        {
            Ok(body) => body,
            Err(error) => {
                errors.push(format!(
                    "fetchAvailableModels returned invalid JSON at {endpoint}: {error}"
                ));
                continue;
            }
        };

        let deprecated_model_ids = body
            .deprecated_model_ids
            .keys()
            .cloned()
            .collect::<HashSet<_>>();
        let default_model_id = body.default_agent_model_id.clone();

        let mut models = Vec::new();
        for (model_id, model) in body.models {
            if !should_include_google_antigravity_model(&model_id, &model, &deprecated_model_ids) {
                continue;
            }

            let display_name = model
                .display_name
                .filter(|name| !name.trim().is_empty())
                .unwrap_or_else(|| humanize_google_antigravity_model_id(&model_id));
            let reasoning = model.supports_thinking || model_id.contains("thinking");

            models.push(ModelInfo {
                id: format!("google-antigravity/{model_id}"),
                name: display_name,
                provider: "google-antigravity".to_string(),
                context_window: model.max_tokens.or(model.max_output_tokens),
                tool_call: true,
                reasoning,
            });
        }

        let default_routing_id =
            default_model_id.map(|model_id| format!("google-antigravity/{model_id}"));
        models.sort_by(|left, right| {
            let left_is_default = default_routing_id.as_ref().is_some_and(|id| left.id == *id);
            let right_is_default = default_routing_id
                .as_ref()
                .is_some_and(|id| right.id == *id);
            right_is_default
                .cmp(&left_is_default)
                .then(left.name.cmp(&right.name))
        });

        return Ok(models);
    }

    anyhow::bail!(
        "failed to fetch Google Antigravity models: {}",
        errors.join("; ")
    );
}

async fn ensure_google_antigravity_models_cache(raw_api_key: &str) -> Vec<ModelInfo> {
    let key_hash = antigravity_models_cache_key(raw_api_key);
    {
        let cache = GOOGLE_ANTIGRAVITY_MODELS_CACHE.read().await;
        if let Some(entry) = cache.as_ref() {
            if entry.key_hash == key_hash
                && entry.fetched_at.elapsed() < GOOGLE_ANTIGRAVITY_MODELS_CACHE_TTL
            {
                return entry.models.clone();
            }
        }
    }

    let stale_models = {
        let cache = GOOGLE_ANTIGRAVITY_MODELS_CACHE.read().await;
        cache
            .as_ref()
            .filter(|entry| entry.key_hash == key_hash)
            .map(|entry| entry.models.clone())
            .unwrap_or_default()
    };

    match fetch_google_antigravity_models_with_api_key(raw_api_key).await {
        Ok(models) => {
            let mut cache = GOOGLE_ANTIGRAVITY_MODELS_CACHE.write().await;
            *cache = Some(CachedGoogleAntigravityModels {
                key_hash,
                models: models.clone(),
                fetched_at: std::time::Instant::now(),
            });
            models
        }
        Err(error) => {
            tracing::warn!(
                %error,
                "failed to fetch Google Antigravity models, using stale cache"
            );
            stale_models
        }
    }
}

/// Helper: which providers have keys configured.
pub(super) async fn configured_providers(config_path: &std::path::Path) -> Vec<&'static str> {
    let mut providers = Vec::new();

    let content = match tokio::fs::read_to_string(config_path).await {
        Ok(c) => c,
        Err(_) => return providers,
    };
    let doc: toml_edit::DocumentMut = match content.parse() {
        Ok(d) => d,
        Err(_) => return providers,
    };

    let has_key = |key: &str, env_var: &str| -> bool {
        if let Some(llm) = doc.get("llm") {
            if let Some(val) = llm.get(key) {
                if let Some(s) = val.as_str() {
                    if let Some(var_name) = s.strip_prefix("env:") {
                        return std::env::var(var_name).is_ok();
                    }
                    return !s.is_empty();
                }
            }
        }
        std::env::var(env_var).is_ok()
    };

    if has_key("anthropic_key", "ANTHROPIC_API_KEY") {
        providers.push("anthropic");
    }
    if has_key("openai_key", "OPENAI_API_KEY") {
        providers.push("openai");
    }
    if has_key("openrouter_key", "OPENROUTER_API_KEY") {
        providers.push("openrouter");
    }
    if has_key("zhipu_key", "ZHIPU_API_KEY") {
        providers.push("zhipu");
    }
    if has_key("groq_key", "GROQ_API_KEY") {
        providers.push("groq");
    }
    if has_key("together_key", "TOGETHER_API_KEY") {
        providers.push("together");
    }
    if has_key("fireworks_key", "FIREWORKS_API_KEY") {
        providers.push("fireworks");
    }
    if has_key("deepseek_key", "DEEPSEEK_API_KEY") {
        providers.push("deepseek");
    }
    if has_key("xai_key", "XAI_API_KEY") {
        providers.push("xai");
    }
    if has_key("mistral_key", "MISTRAL_API_KEY") {
        providers.push("mistral");
    }
    if has_key("opencode_zen_key", "OPENCODE_ZEN_API_KEY") {
        providers.push("opencode-zen");
    }
    if has_key("minimax_key", "MINIMAX_API_KEY") {
        providers.push("minimax");
    }
    if has_key("moonshot_key", "MOONSHOT_API_KEY") {
        providers.push("moonshot");
    }
    if has_key("zai_coding_plan_key", "ZAI_CODING_PLAN_API_KEY") {
        providers.push("zai-coding-plan");
    }
    if has_key("google_antigravity_key", "GOOGLE_ANTIGRAVITY_API_KEY") {
        providers.push("google-antigravity");
    }

    providers
}

pub(super) async fn get_models(
    State(state): State<Arc<ApiState>>,
    Query(query): Query<ModelsQuery>,
) -> Result<Json<ModelsResponse>, StatusCode> {
    let config_path = state.config_path.read().await.clone();
    let configured = configured_providers(&config_path).await;
    let requested_provider = query
        .provider
        .as_deref()
        .map(str::trim)
        .filter(|provider| !provider.is_empty());

    let catalog = ensure_models_cache().await;

    let mut models: Vec<ModelInfo> = catalog
        .into_iter()
        .filter(|model| {
            if let Some(provider) = requested_provider {
                model.provider == provider
            } else {
                configured.contains(&model.provider.as_str())
            }
        })
        .collect();

    for model in extra_models() {
        if let Some(provider) = requested_provider {
            if model.provider == provider {
                models.push(model);
            }
        } else if configured.contains(&model.provider.as_str()) {
            models.push(model);
        }
    }

    let should_include_google_antigravity = match requested_provider {
        Some(provider) => provider == "google-antigravity",
        None => configured.contains(&"google-antigravity"),
    };

    if should_include_google_antigravity {
        if let Some(raw_api_key) = google_antigravity_api_key_from_config(&config_path).await {
            let antigravity_models = ensure_google_antigravity_models_cache(&raw_api_key).await;
            models.extend(antigravity_models);
        }
    }

    models.sort_by(|left, right| left.id.cmp(&right.id));
    models.dedup_by(|left, right| left.id == right.id);
    models.sort_by(|left, right| {
        left.provider
            .cmp(&right.provider)
            .then(left.name.cmp(&right.name))
    });

    Ok(Json(ModelsResponse { models }))
}

pub(super) async fn refresh_models(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ModelsResponse>, StatusCode> {
    {
        let mut cache = MODELS_CACHE.write().await;
        *cache = (Vec::new(), std::time::Instant::now() - MODELS_CACHE_TTL);
    }

    get_models(State(state), Query(ModelsQuery { provider: None })).await
}
