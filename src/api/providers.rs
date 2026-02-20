use super::state::ApiState;

use axum::Json;
use axum::extract::Query;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::http::StatusCode;
use axum::response::Html;
use base64::Engine as _;
use rig::agent::AgentBuilder;
use rig::completion::{CompletionModel as _, Prompt as _};
use serde::{Deserialize, Serialize};
use sha2::Digest as _;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Serialize)]
pub(super) struct ProviderStatus {
    anthropic: bool,
    openai: bool,
    openrouter: bool,
    zhipu: bool,
    groq: bool,
    together: bool,
    fireworks: bool,
    deepseek: bool,
    xai: bool,
    mistral: bool,
    ollama: bool,
    opencode_zen: bool,
    nvidia: bool,
    minimax: bool,
    moonshot: bool,
    zai_coding_plan: bool,
    google_antigravity: bool,
}

#[derive(Serialize)]
pub(super) struct ProvidersResponse {
    providers: ProviderStatus,
    has_any: bool,
}

#[derive(Deserialize)]
pub(super) struct ProviderUpdateRequest {
    provider: String,
    api_key: String,
    model: String,
}

#[derive(Serialize)]
pub(super) struct ProviderUpdateResponse {
    success: bool,
    message: String,
}

#[derive(Deserialize)]
pub(super) struct ProviderModelTestRequest {
    provider: String,
    api_key: String,
    model: String,
}

#[derive(Serialize)]
pub(super) struct ProviderModelTestResponse {
    success: bool,
    message: String,
    provider: String,
    model: String,
    sample: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct GoogleAntigravityOauthStartRequest {
    redirect_base_url: Option<String>,
}

#[derive(Serialize)]
pub(super) struct GoogleAntigravityOauthStartResponse {
    auth_url: String,
    state: String,
}

#[derive(Deserialize)]
pub(super) struct GoogleAntigravityOauthStatusRequest {
    state: String,
}

#[derive(Serialize)]
pub(super) struct GoogleAntigravityOauthStatusResponse {
    status: String,
    message: String,
    api_key: Option<String>,
    email: Option<String>,
    project_id: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct GoogleAntigravityOauthCallbackQuery {
    code: Option<String>,
    state: Option<String>,
    error: Option<String>,
}

#[derive(Clone)]
struct GoogleAntigravityOauthSession {
    verifier: String,
    redirect_uri: String,
    created_at: Instant,
    result: Option<GoogleAntigravityOauthResult>,
}

#[derive(Clone)]
enum GoogleAntigravityOauthResult {
    Success {
        api_key: String,
        email: Option<String>,
        project_id: String,
    },
    Error(String),
}

static GOOGLE_ANTIGRAVITY_OAUTH_SESSIONS: LazyLock<
    RwLock<HashMap<String, GoogleAntigravityOauthSession>>,
> = LazyLock::new(|| RwLock::new(HashMap::new()));

static GOOGLE_ANTIGRAVITY_CLIENT_ID: LazyLock<String> =
    LazyLock::new(|| std::env::var("GOOGLE_ANTIGRAVITY_CLIENT_ID").unwrap_or_default());
static GOOGLE_ANTIGRAVITY_CLIENT_SECRET: LazyLock<String> =
    LazyLock::new(|| std::env::var("GOOGLE_ANTIGRAVITY_CLIENT_SECRET").unwrap_or_default());
const GOOGLE_ANTIGRAVITY_AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const GOOGLE_ANTIGRAVITY_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const GOOGLE_ANTIGRAVITY_DEFAULT_PROJECT_ID: &str = "rising-fact-p41fc";
const GOOGLE_ANTIGRAVITY_SESSION_TTL: Duration = Duration::from_secs(10 * 60);
const GOOGLE_ANTIGRAVITY_SCOPES: &[&str] = &[
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
];

fn provider_toml_key(provider: &str) -> Option<&'static str> {
    match provider {
        "anthropic" => Some("anthropic_key"),
        "openai" => Some("openai_key"),
        "openrouter" => Some("openrouter_key"),
        "zhipu" => Some("zhipu_key"),
        "groq" => Some("groq_key"),
        "together" => Some("together_key"),
        "fireworks" => Some("fireworks_key"),
        "deepseek" => Some("deepseek_key"),
        "xai" => Some("xai_key"),
        "mistral" => Some("mistral_key"),
        "ollama" => Some("ollama_base_url"),
        "opencode-zen" => Some("opencode_zen_key"),
        "nvidia" => Some("nvidia_key"),
        "minimax" => Some("minimax_key"),
        "moonshot" => Some("moonshot_key"),
        "zai-coding-plan" => Some("zai_coding_plan_key"),
        "google-antigravity" => Some("google_antigravity_key"),
        _ => None,
    }
}

fn model_matches_provider(provider: &str, model: &str) -> bool {
    crate::llm::routing::provider_from_model(model) == provider
}

fn build_test_llm_config(provider: &str, credential: &str) -> crate::config::LlmConfig {
    use crate::config::{ApiType, ProviderConfig};

    let mut providers = HashMap::new();
    let provider_config = match provider {
        "anthropic" => Some(ProviderConfig {
            api_type: ApiType::Anthropic,
            base_url: "https://api.anthropic.com".to_string(),
            api_key: credential.to_string(),
            name: None,
        }),
        "openai" => Some(ProviderConfig {
            api_type: ApiType::OpenAiCompletions,
            base_url: "https://api.openai.com".to_string(),
            api_key: credential.to_string(),
            name: None,
        }),
        "openrouter" => Some(ProviderConfig {
            api_type: ApiType::OpenAiCompletions,
            base_url: "https://openrouter.ai/api".to_string(),
            api_key: credential.to_string(),
            name: None,
        }),
        "zhipu" => Some(ProviderConfig {
            api_type: ApiType::OpenAiCompletions,
            base_url: "https://api.z.ai/api/paas/v4".to_string(),
            api_key: credential.to_string(),
            name: None,
        }),
        "groq" => Some(ProviderConfig {
            api_type: ApiType::OpenAiCompletions,
            base_url: "https://api.groq.com/openai".to_string(),
            api_key: credential.to_string(),
            name: None,
        }),
        "together" => Some(ProviderConfig {
            api_type: ApiType::OpenAiCompletions,
            base_url: "https://api.together.xyz".to_string(),
            api_key: credential.to_string(),
            name: None,
        }),
        "fireworks" => Some(ProviderConfig {
            api_type: ApiType::OpenAiCompletions,
            base_url: "https://api.fireworks.ai/inference".to_string(),
            api_key: credential.to_string(),
            name: None,
        }),
        "deepseek" => Some(ProviderConfig {
            api_type: ApiType::OpenAiCompletions,
            base_url: "https://api.deepseek.com".to_string(),
            api_key: credential.to_string(),
            name: None,
        }),
        "xai" => Some(ProviderConfig {
            api_type: ApiType::OpenAiCompletions,
            base_url: "https://api.x.ai".to_string(),
            api_key: credential.to_string(),
            name: None,
        }),
        "mistral" => Some(ProviderConfig {
            api_type: ApiType::OpenAiCompletions,
            base_url: "https://api.mistral.ai".to_string(),
            api_key: credential.to_string(),
            name: None,
        }),
        "opencode-zen" => Some(ProviderConfig {
            api_type: ApiType::OpenAiCompletions,
            base_url: "https://opencode.ai/zen".to_string(),
            api_key: credential.to_string(),
            name: None,
        }),
        "nvidia" => Some(ProviderConfig {
            api_type: ApiType::OpenAiCompletions,
            base_url: "https://integrate.api.nvidia.com".to_string(),
            api_key: credential.to_string(),
            name: None,
        }),
        "minimax" => Some(ProviderConfig {
            api_type: ApiType::Anthropic,
            base_url: "https://api.minimax.io/anthropic".to_string(),
            api_key: credential.to_string(),
            name: None,
        }),
        "moonshot" => Some(ProviderConfig {
            api_type: ApiType::OpenAiCompletions,
            base_url: "https://api.moonshot.ai".to_string(),
            api_key: credential.to_string(),
            name: None,
        }),
        "zai-coding-plan" => Some(ProviderConfig {
            api_type: ApiType::OpenAiCompletions,
            base_url: "https://api.z.ai/api/coding/paas/v4".to_string(),
            api_key: credential.to_string(),
            name: None,
        }),
        "google-antigravity" => Some(ProviderConfig {
            api_type: ApiType::OpenAiCompletions,
            base_url: "https://daily-cloudcode-pa.sandbox.googleapis.com".to_string(),
            api_key: credential.to_string(),
            name: None,
        }),
        _ => None,
    };

    if let Some(provider_config) = provider_config {
        providers.insert(provider.to_string(), provider_config);
    }

    crate::config::LlmConfig {
        anthropic_key: (provider == "anthropic").then(|| credential.to_string()),
        openai_key: (provider == "openai").then(|| credential.to_string()),
        openrouter_key: (provider == "openrouter").then(|| credential.to_string()),
        zhipu_key: (provider == "zhipu").then(|| credential.to_string()),
        groq_key: (provider == "groq").then(|| credential.to_string()),
        together_key: (provider == "together").then(|| credential.to_string()),
        fireworks_key: (provider == "fireworks").then(|| credential.to_string()),
        deepseek_key: (provider == "deepseek").then(|| credential.to_string()),
        xai_key: (provider == "xai").then(|| credential.to_string()),
        mistral_key: (provider == "mistral").then(|| credential.to_string()),
        ollama_key: None,
        ollama_base_url: (provider == "ollama").then(|| credential.to_string()),
        opencode_zen_key: (provider == "opencode-zen").then(|| credential.to_string()),
        nvidia_key: (provider == "nvidia").then(|| credential.to_string()),
        minimax_key: (provider == "minimax").then(|| credential.to_string()),
        moonshot_key: (provider == "moonshot").then(|| credential.to_string()),
        zai_coding_plan_key: (provider == "zai-coding-plan").then(|| credential.to_string()),
        google_antigravity_key: (provider == "google-antigravity").then(|| credential.to_string()),
        providers,
    }
}

pub(super) async fn get_providers(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ProvidersResponse>, StatusCode> {
    let config_path = state.config_path.read().await.clone();

    let (
        anthropic,
        openai,
        openrouter,
        zhipu,
        groq,
        together,
        fireworks,
        deepseek,
        xai,
        mistral,
        ollama,
        opencode_zen,
        nvidia,
        minimax,
        moonshot,
        zai_coding_plan,
        google_antigravity,
    ) = if config_path.exists() {
        let content = tokio::fs::read_to_string(&config_path)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let doc: toml_edit::DocumentMut = content
            .parse()
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        let has_value = |key: &str, env_var: &str| -> bool {
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

        (
            has_value("anthropic_key", "ANTHROPIC_API_KEY"),
            has_value("openai_key", "OPENAI_API_KEY"),
            has_value("openrouter_key", "OPENROUTER_API_KEY"),
            has_value("zhipu_key", "ZHIPU_API_KEY"),
            has_value("groq_key", "GROQ_API_KEY"),
            has_value("together_key", "TOGETHER_API_KEY"),
            has_value("fireworks_key", "FIREWORKS_API_KEY"),
            has_value("deepseek_key", "DEEPSEEK_API_KEY"),
            has_value("xai_key", "XAI_API_KEY"),
            has_value("mistral_key", "MISTRAL_API_KEY"),
            has_value("ollama_base_url", "OLLAMA_BASE_URL")
                || has_value("ollama_key", "OLLAMA_API_KEY"),
            has_value("opencode_zen_key", "OPENCODE_ZEN_API_KEY"),
            has_value("nvidia_key", "NVIDIA_API_KEY"),
            has_value("minimax_key", "MINIMAX_API_KEY"),
            has_value("moonshot_key", "MOONSHOT_API_KEY"),
            has_value("zai_coding_plan_key", "ZAI_CODING_PLAN_API_KEY"),
            has_value("google_antigravity_key", "GOOGLE_ANTIGRAVITY_API_KEY"),
        )
    } else {
        (
            std::env::var("ANTHROPIC_API_KEY").is_ok(),
            std::env::var("OPENAI_API_KEY").is_ok(),
            std::env::var("OPENROUTER_API_KEY").is_ok(),
            std::env::var("ZHIPU_API_KEY").is_ok(),
            std::env::var("GROQ_API_KEY").is_ok(),
            std::env::var("TOGETHER_API_KEY").is_ok(),
            std::env::var("FIREWORKS_API_KEY").is_ok(),
            std::env::var("DEEPSEEK_API_KEY").is_ok(),
            std::env::var("XAI_API_KEY").is_ok(),
            std::env::var("MISTRAL_API_KEY").is_ok(),
            std::env::var("OLLAMA_BASE_URL").is_ok() || std::env::var("OLLAMA_API_KEY").is_ok(),
            std::env::var("OPENCODE_ZEN_API_KEY").is_ok(),
            std::env::var("NVIDIA_API_KEY").is_ok(),
            std::env::var("MINIMAX_API_KEY").is_ok(),
            std::env::var("MOONSHOT_API_KEY").is_ok(),
            std::env::var("ZAI_CODING_PLAN_API_KEY").is_ok(),
            std::env::var("GOOGLE_ANTIGRAVITY_API_KEY").is_ok(),
        )
    };

    let providers = ProviderStatus {
        anthropic,
        openai,
        openrouter,
        zhipu,
        groq,
        together,
        fireworks,
        deepseek,
        xai,
        mistral,
        ollama,
        opencode_zen,
        nvidia,
        minimax,
        moonshot,
        zai_coding_plan,
        google_antigravity,
    };
    let has_any = providers.anthropic
        || providers.openai
        || providers.openrouter
        || providers.zhipu
        || providers.groq
        || providers.together
        || providers.fireworks
        || providers.deepseek
        || providers.xai
        || providers.mistral
        || providers.ollama
        || providers.opencode_zen
        || providers.nvidia
        || providers.minimax
        || providers.moonshot
        || providers.zai_coding_plan
        || providers.google_antigravity;

    Ok(Json(ProvidersResponse { providers, has_any }))
}

fn cleanup_google_antigravity_oauth_sessions(
    sessions: &mut HashMap<String, GoogleAntigravityOauthSession>,
) {
    sessions.retain(|_, session| session.created_at.elapsed() < GOOGLE_ANTIGRAVITY_SESSION_TTL);
}

fn infer_redirect_base_url(headers: &HeaderMap) -> String {
    let host = headers
        .get("x-forwarded-host")
        .or_else(|| headers.get("host"))
        .and_then(|value| value.to_str().ok())
        .unwrap_or("127.0.0.1:19898");

    let scheme = headers
        .get("x-forwarded-proto")
        .and_then(|value| value.to_str().ok())
        .filter(|value| *value == "http" || *value == "https")
        .unwrap_or_else(|| {
            if host.starts_with("localhost") || host.starts_with("127.0.0.1") {
                "http"
            } else {
                "https"
            }
        });

    format!("{scheme}://{host}")
}

fn resolve_redirect_base_url(headers: &HeaderMap, requested_base_url: Option<&str>) -> String {
    let inferred = infer_redirect_base_url(headers);
    let requested = requested_base_url
        .map(str::trim)
        .filter(|value| !value.is_empty());

    if let Some(base_url) = requested {
        if let Ok(parsed) = reqwest::Url::parse(base_url) {
            if parsed.scheme() == "http" || parsed.scheme() == "https" {
                return base_url.trim_end_matches('/').to_string();
            }
        }
    }

    inferred.trim_end_matches('/').to_string()
}

fn build_google_antigravity_redirect_uri(base_url: &str) -> String {
    format!(
        "{}/api/providers/google-antigravity/oauth/callback",
        base_url.trim_end_matches('/')
    )
}

fn generate_pkce_verifier_and_challenge() -> (String, String) {
    let verifier = format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple());
    let digest = sha2::Sha256::digest(verifier.as_bytes());
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);
    (verifier, challenge)
}

fn build_google_antigravity_auth_url(
    challenge: &str,
    state: &str,
    redirect_uri: &str,
) -> Result<String, StatusCode> {
    if GOOGLE_ANTIGRAVITY_CLIENT_ID.is_empty() {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    let mut auth_url = reqwest::Url::parse(GOOGLE_ANTIGRAVITY_AUTH_URL)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    auth_url
        .query_pairs_mut()
        .append_pair("client_id", GOOGLE_ANTIGRAVITY_CLIENT_ID.as_str())
        .append_pair("response_type", "code")
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("scope", &GOOGLE_ANTIGRAVITY_SCOPES.join(" "))
        .append_pair("code_challenge", challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", state)
        .append_pair("access_type", "offline")
        .append_pair("prompt", "consent");
    Ok(auth_url.to_string())
}

fn oauth_result_page(title: &str, message: &str) -> Html<String> {
    Html(format!(
        "<!doctype html><html><head><meta charset=\"utf-8\" /><title>{}</title></head><body><main><h1>{}</h1><p>{}</p><p>You can close this window.</p></main></body></html>",
        title, title, message
    ))
}

async fn fetch_google_antigravity_user_email(
    client: &reqwest::Client,
    access_token: &str,
) -> Option<String> {
    let response = client
        .get("https://www.googleapis.com/oauth2/v1/userinfo?alt=json")
        .header("authorization", format!("Bearer {access_token}"))
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    let body = response.json::<serde_json::Value>().await.ok()?;
    body.get("email")
        .and_then(|value| value.as_str())
        .map(std::string::ToString::to_string)
}

async fn fetch_google_antigravity_project_id(
    client: &reqwest::Client,
    access_token: &str,
) -> String {
    let endpoints = [
        "https://cloudcode-pa.googleapis.com",
        "https://daily-cloudcode-pa.sandbox.googleapis.com",
    ];

    for endpoint in endpoints {
        let response = match client
            .post(format!("{endpoint}/v1internal:loadCodeAssist"))
            .header("authorization", format!("Bearer {access_token}"))
            .header("content-type", "application/json")
            .header("user-agent", "google-api-nodejs-client/9.15.1")
            .header(
                "x-goog-api-client",
                "google-cloud-sdk vscode_cloudshelleditor/0.1",
            )
            .header(
                "client-metadata",
                r#"{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}"#,
            )
            .json(&serde_json::json!({
                "metadata": {
                    "ideType": "IDE_UNSPECIFIED",
                    "platform": "PLATFORM_UNSPECIFIED",
                    "pluginType": "GEMINI",
                }
            }))
            .send()
            .await
        {
            Ok(response) => response,
            Err(_) => continue,
        };

        if !response.status().is_success() {
            continue;
        }

        let body = match response.json::<serde_json::Value>().await {
            Ok(body) => body,
            Err(_) => continue,
        };

        if let Some(project_id) = body
            .get("cloudaicompanionProject")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
        {
            return project_id.to_string();
        }

        if let Some(project_id) = body
            .get("cloudaicompanionProject")
            .and_then(|value| value.get("id"))
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
        {
            return project_id.to_string();
        }
    }

    GOOGLE_ANTIGRAVITY_DEFAULT_PROJECT_ID.to_string()
}

pub(super) async fn start_google_antigravity_oauth(
    headers: HeaderMap,
    Json(request): Json<GoogleAntigravityOauthStartRequest>,
) -> Result<Json<GoogleAntigravityOauthStartResponse>, StatusCode> {
    let state = Uuid::new_v4().simple().to_string();
    let (verifier, challenge) = generate_pkce_verifier_and_challenge();
    let base_url = resolve_redirect_base_url(&headers, request.redirect_base_url.as_deref());
    let redirect_uri = build_google_antigravity_redirect_uri(&base_url);
    let auth_url = build_google_antigravity_auth_url(&challenge, &state, &redirect_uri)?;

    let mut sessions = GOOGLE_ANTIGRAVITY_OAUTH_SESSIONS.write().await;
    cleanup_google_antigravity_oauth_sessions(&mut sessions);
    sessions.insert(
        state.clone(),
        GoogleAntigravityOauthSession {
            verifier,
            redirect_uri,
            created_at: Instant::now(),
            result: None,
        },
    );

    Ok(Json(GoogleAntigravityOauthStartResponse {
        auth_url,
        state,
    }))
}

pub(super) async fn google_antigravity_oauth_status(
    Query(request): Query<GoogleAntigravityOauthStatusRequest>,
) -> Json<GoogleAntigravityOauthStatusResponse> {
    let mut sessions = GOOGLE_ANTIGRAVITY_OAUTH_SESSIONS.write().await;
    cleanup_google_antigravity_oauth_sessions(&mut sessions);

    let Some(session) = sessions.get(&request.state) else {
        return Json(GoogleAntigravityOauthStatusResponse {
            status: "not_found".to_string(),
            message: "OAuth session not found or expired".to_string(),
            api_key: None,
            email: None,
            project_id: None,
        });
    };

    match &session.result {
        None => Json(GoogleAntigravityOauthStatusResponse {
            status: "pending".to_string(),
            message: "Waiting for Google callback".to_string(),
            api_key: None,
            email: None,
            project_id: None,
        }),
        Some(GoogleAntigravityOauthResult::Success {
            api_key,
            email,
            project_id,
        }) => Json(GoogleAntigravityOauthStatusResponse {
            status: "success".to_string(),
            message: "Google authentication complete".to_string(),
            api_key: Some(api_key.clone()),
            email: email.clone(),
            project_id: Some(project_id.clone()),
        }),
        Some(GoogleAntigravityOauthResult::Error(message)) => {
            Json(GoogleAntigravityOauthStatusResponse {
                status: "error".to_string(),
                message: message.clone(),
                api_key: None,
                email: None,
                project_id: None,
            })
        }
    }
}

pub(super) async fn google_antigravity_oauth_callback(
    Query(query): Query<GoogleAntigravityOauthCallbackQuery>,
) -> Html<String> {
    let Some(state) = query.state.clone().filter(|value| !value.is_empty()) else {
        return oauth_result_page("Authentication Failed", "Missing OAuth state.");
    };

    let Some(code) = query.code.clone().filter(|value| !value.is_empty()) else {
        let mut sessions = GOOGLE_ANTIGRAVITY_OAUTH_SESSIONS.write().await;
        if let Some(session) = sessions.get_mut(&state) {
            session.result = Some(GoogleAntigravityOauthResult::Error(
                query
                    .error
                    .clone()
                    .unwrap_or_else(|| "Missing OAuth code".to_string()),
            ));
        }
        return oauth_result_page(
            "Authentication Failed",
            query
                .error
                .as_deref()
                .unwrap_or("Google did not provide an authorization code."),
        );
    };

    let (verifier, redirect_uri, expired) = {
        let mut sessions = GOOGLE_ANTIGRAVITY_OAUTH_SESSIONS.write().await;
        cleanup_google_antigravity_oauth_sessions(&mut sessions);
        let Some(session) = sessions.get(&state) else {
            return oauth_result_page(
                "Authentication Failed",
                "OAuth session expired or was not found. Start the flow again.",
            );
        };
        (
            session.verifier.clone(),
            session.redirect_uri.clone(),
            session.created_at.elapsed() >= GOOGLE_ANTIGRAVITY_SESSION_TTL,
        )
    };

    if expired {
        return oauth_result_page(
            "Authentication Failed",
            "OAuth session expired. Start the flow again.",
        );
    }

    let client = reqwest::Client::new();
    if GOOGLE_ANTIGRAVITY_CLIENT_ID.is_empty() || GOOGLE_ANTIGRAVITY_CLIENT_SECRET.is_empty() {
        let mut sessions = GOOGLE_ANTIGRAVITY_OAUTH_SESSIONS.write().await;
        if let Some(session) = sessions.get_mut(&state) {
            session.result = Some(GoogleAntigravityOauthResult::Error(
                "OAuth client credentials are not configured on this server.".to_string(),
            ));
        }
        return oauth_result_page(
            "Authentication Failed",
            "Google Antigravity OAuth client credentials are not configured on this server.",
        );
    }
    let token_response = match client
        .post(GOOGLE_ANTIGRAVITY_TOKEN_URL)
        .header("content-type", "application/x-www-form-urlencoded")
        .form(&[
            ("client_id", GOOGLE_ANTIGRAVITY_CLIENT_ID.as_str()),
            ("client_secret", GOOGLE_ANTIGRAVITY_CLIENT_SECRET.as_str()),
            ("code", code.as_str()),
            ("grant_type", "authorization_code"),
            ("redirect_uri", redirect_uri.as_str()),
            ("code_verifier", verifier.as_str()),
        ])
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            let mut sessions = GOOGLE_ANTIGRAVITY_OAUTH_SESSIONS.write().await;
            if let Some(session) = sessions.get_mut(&state) {
                session.result = Some(GoogleAntigravityOauthResult::Error(format!(
                    "Token exchange failed: {error}"
                )));
            }
            return oauth_result_page("Authentication Failed", "Token exchange failed.");
        }
    };

    if !token_response.status().is_success() {
        let body = token_response.text().await.unwrap_or_default();
        let mut sessions = GOOGLE_ANTIGRAVITY_OAUTH_SESSIONS.write().await;
        if let Some(session) = sessions.get_mut(&state) {
            session.result = Some(GoogleAntigravityOauthResult::Error(format!(
                "Token exchange failed: {}",
                body.chars().take(250).collect::<String>()
            )));
        }
        return oauth_result_page(
            "Authentication Failed",
            "Google rejected the authorization code.",
        );
    }

    let token_body = match token_response.json::<serde_json::Value>().await {
        Ok(body) => body,
        Err(_) => {
            let mut sessions = GOOGLE_ANTIGRAVITY_OAUTH_SESSIONS.write().await;
            if let Some(session) = sessions.get_mut(&state) {
                session.result = Some(GoogleAntigravityOauthResult::Error(
                    "Token exchange returned invalid JSON".to_string(),
                ));
            }
            return oauth_result_page(
                "Authentication Failed",
                "Google token response was invalid.",
            );
        }
    };

    let access_token = token_body
        .get("access_token")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string();
    let refresh_token = token_body
        .get("refresh_token")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string();
    let expires_in = token_body
        .get("expires_in")
        .and_then(|value| value.as_i64())
        .unwrap_or(3600);

    if access_token.is_empty() || refresh_token.is_empty() {
        let mut sessions = GOOGLE_ANTIGRAVITY_OAUTH_SESSIONS.write().await;
        if let Some(session) = sessions.get_mut(&state) {
            session.result = Some(GoogleAntigravityOauthResult::Error(
                "Google OAuth response is missing tokens".to_string(),
            ));
        }
        return oauth_result_page(
            "Authentication Failed",
            "Google OAuth response did not include required tokens.",
        );
    }

    let expires_at = chrono::Utc::now().timestamp_millis() + (expires_in * 1000) - (5 * 60 * 1000);
    let email = fetch_google_antigravity_user_email(&client, &access_token).await;
    let project_id = fetch_google_antigravity_project_id(&client, &access_token).await;
    let api_key = serde_json::json!({
        "token": access_token,
        "projectId": project_id,
        "refreshToken": refresh_token,
        "expiresAt": expires_at,
        "email": email,
    })
    .to_string();

    let mut sessions = GOOGLE_ANTIGRAVITY_OAUTH_SESSIONS.write().await;
    if let Some(session) = sessions.get_mut(&state) {
        session.result = Some(GoogleAntigravityOauthResult::Success {
            api_key,
            email,
            project_id,
        });
    }

    oauth_result_page(
        "Authentication Complete",
        "Google Antigravity authentication succeeded.",
    )
}

pub(super) async fn update_provider(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<ProviderUpdateRequest>,
) -> Result<Json<ProviderUpdateResponse>, StatusCode> {
    let Some(key_name) = provider_toml_key(&request.provider) else {
        return Ok(Json(ProviderUpdateResponse {
            success: false,
            message: format!("Unknown provider: {}", request.provider),
        }));
    };

    if request.api_key.trim().is_empty() {
        return Ok(Json(ProviderUpdateResponse {
            success: false,
            message: "API key cannot be empty".into(),
        }));
    }

    if request.model.trim().is_empty() {
        return Ok(Json(ProviderUpdateResponse {
            success: false,
            message: "Model cannot be empty".into(),
        }));
    }

    if !model_matches_provider(&request.provider, &request.model) {
        return Ok(Json(ProviderUpdateResponse {
            success: false,
            message: format!(
                "Model '{}' does not match provider '{}'.",
                request.model, request.provider
            ),
        }));
    }

    let config_path = state.config_path.read().await.clone();

    let content = if config_path.exists() {
        tokio::fs::read_to_string(&config_path)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    } else {
        String::new()
    };

    let mut doc: toml_edit::DocumentMut = content
        .parse()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if doc.get("llm").is_none() {
        doc["llm"] = toml_edit::Item::Table(toml_edit::Table::new());
    }

    doc["llm"][key_name] = toml_edit::value(request.api_key);

    if doc.get("defaults").is_none() {
        doc["defaults"] = toml_edit::Item::Table(toml_edit::Table::new());
    }
    if let Some(defaults) = doc.get_mut("defaults").and_then(|d| d.as_table_mut()) {
        if defaults.get("routing").is_none() {
            defaults["routing"] = toml_edit::Item::Table(toml_edit::Table::new());
        }
        if let Some(routing_table) = defaults.get_mut("routing").and_then(|r| r.as_table_mut()) {
            routing_table["channel"] = toml_edit::value(request.model.as_str());
            routing_table["branch"] = toml_edit::value(request.model.as_str());
            routing_table["worker"] = toml_edit::value(request.model.as_str());
            routing_table["compactor"] = toml_edit::value(request.model.as_str());
            routing_table["cortex"] = toml_edit::value(request.model.as_str());
        }
    }

    if let Some(agents) = doc
        .get_mut("agents")
        .and_then(|agents_item| agents_item.as_array_of_tables_mut())
    {
        if let Some(default_agent) = agents.iter_mut().find(|agent| {
            agent
                .get("default")
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
        }) {
            if default_agent.get("routing").is_none() {
                default_agent["routing"] = toml_edit::Item::Table(toml_edit::Table::new());
            }
            if let Some(routing_table) = default_agent
                .get_mut("routing")
                .and_then(|routing_item| routing_item.as_table_mut())
            {
                routing_table["channel"] = toml_edit::value(request.model.as_str());
                routing_table["branch"] = toml_edit::value(request.model.as_str());
                routing_table["worker"] = toml_edit::value(request.model.as_str());
                routing_table["compactor"] = toml_edit::value(request.model.as_str());
                routing_table["cortex"] = toml_edit::value(request.model.as_str());
            }
        }
    }

    tokio::fs::write(&config_path, doc.to_string())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    state
        .provider_setup_tx
        .try_send(crate::ProviderSetupEvent::ProvidersConfigured)
        .ok();

    Ok(Json(ProviderUpdateResponse {
        success: true,
        message: format!(
            "Provider '{}' configured. Model '{}' verified and applied to defaults and the default agent routing.",
            request.provider, request.model
        ),
    }))
}

pub(super) async fn test_provider_model(
    Json(request): Json<ProviderModelTestRequest>,
) -> Result<Json<ProviderModelTestResponse>, StatusCode> {
    if provider_toml_key(&request.provider).is_none() {
        return Ok(Json(ProviderModelTestResponse {
            success: false,
            message: format!("Unknown provider: {}", request.provider),
            provider: request.provider,
            model: request.model,
            sample: None,
        }));
    }

    if request.api_key.trim().is_empty() {
        return Ok(Json(ProviderModelTestResponse {
            success: false,
            message: "API key cannot be empty".to_string(),
            provider: request.provider,
            model: request.model,
            sample: None,
        }));
    }

    if request.model.trim().is_empty() {
        return Ok(Json(ProviderModelTestResponse {
            success: false,
            message: "Model cannot be empty".to_string(),
            provider: request.provider,
            model: request.model,
            sample: None,
        }));
    }

    if !model_matches_provider(&request.provider, &request.model) {
        return Ok(Json(ProviderModelTestResponse {
            success: false,
            message: format!(
                "Model '{}' does not match provider '{}'.",
                request.model, request.provider
            ),
            provider: request.provider,
            model: request.model,
            sample: None,
        }));
    }

    let llm_config = build_test_llm_config(&request.provider, request.api_key.trim());
    let llm_manager = match crate::llm::LlmManager::new(llm_config).await {
        Ok(manager) => Arc::new(manager),
        Err(error) => {
            return Ok(Json(ProviderModelTestResponse {
                success: false,
                message: format!("Failed to initialize provider: {error}"),
                provider: request.provider,
                model: request.model,
                sample: None,
            }));
        }
    };

    let model = crate::llm::SpacebotModel::make(&llm_manager, request.model.clone());
    let agent = AgentBuilder::new(model)
        .preamble("You are running a provider connectivity check. Reply with exactly: OK")
        .build();

    match agent.prompt("Connection test").await {
        Ok(sample) => Ok(Json(ProviderModelTestResponse {
            success: true,
            message: "Model responded successfully".to_string(),
            provider: request.provider,
            model: request.model,
            sample: Some(sample),
        })),
        Err(error) => Ok(Json(ProviderModelTestResponse {
            success: false,
            message: format!("Model test failed: {error}"),
            provider: request.provider,
            model: request.model,
            sample: None,
        })),
    }
}

pub(super) async fn delete_provider(
    State(state): State<Arc<ApiState>>,
    axum::extract::Path(provider): axum::extract::Path<String>,
) -> Result<Json<ProviderUpdateResponse>, StatusCode> {
    let Some(key_name) = provider_toml_key(&provider) else {
        return Ok(Json(ProviderUpdateResponse {
            success: false,
            message: format!("Unknown provider: {}", provider),
        }));
    };

    let config_path = state.config_path.read().await.clone();
    if !config_path.exists() {
        return Ok(Json(ProviderUpdateResponse {
            success: false,
            message: "No config file found".into(),
        }));
    }

    let content = tokio::fs::read_to_string(&config_path)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let mut doc: toml_edit::DocumentMut = content
        .parse()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if let Some(llm) = doc.get_mut("llm") {
        if let Some(table) = llm.as_table_mut() {
            table.remove(key_name);
        }
    }

    tokio::fs::write(&config_path, doc.to_string())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(ProviderUpdateResponse {
        success: true,
        message: format!("Provider '{}' removed", provider),
    }))
}
