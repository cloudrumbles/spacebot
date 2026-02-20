//! SpacebotModel: Custom CompletionModel implementation that routes through LlmManager.

use crate::config::{ApiType, ProviderConfig};
use crate::llm::manager::LlmManager;
use crate::llm::routing::{
    self, MAX_FALLBACK_ATTEMPTS, MAX_RETRIES_PER_MODEL, RETRY_BASE_DELAY_MS, RoutingConfig,
};

use rig::completion::{self, CompletionError, CompletionModel, CompletionRequest, GetTokenUsage};
use rig::message::{
    AssistantContent, DocumentSourceKind, Image, Message, MimeType, Text, ToolCall, ToolFunction,
    UserContent,
};
use rig::one_or_many::OneOrMany;
use rig::streaming::StreamingCompletionResponse;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Raw provider response. Wraps the JSON so Rig can carry it through.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawResponse {
    pub body: serde_json::Value,
}

/// Streaming response placeholder. Streaming will be implemented per-provider
/// when we wire up SSE parsing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawStreamingResponse {
    pub body: serde_json::Value,
}

impl GetTokenUsage for RawStreamingResponse {
    fn token_usage(&self) -> Option<completion::Usage> {
        None
    }
}

#[derive(Clone, Debug)]
struct CachedGoogleAntigravityToken {
    token: String,
    expires_at_ms: i64,
}

#[derive(Clone, Debug)]
struct CachedGoogleAntigravityVersion {
    version: String,
    fetched_at: Instant,
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

static GOOGLE_ANTIGRAVITY_ACCESS_TOKEN_CACHE: LazyLock<
    RwLock<HashMap<String, CachedGoogleAntigravityToken>>,
> = LazyLock::new(|| RwLock::new(HashMap::new()));
static GOOGLE_ANTIGRAVITY_VERSION_CACHE: LazyLock<RwLock<Option<CachedGoogleAntigravityVersion>>> =
    LazyLock::new(|| RwLock::new(None));

const GOOGLE_ANTIGRAVITY_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
static GOOGLE_ANTIGRAVITY_CLIENT_ID: LazyLock<String> =
    LazyLock::new(|| std::env::var("GOOGLE_ANTIGRAVITY_CLIENT_ID").unwrap_or_default());
static GOOGLE_ANTIGRAVITY_CLIENT_SECRET: LazyLock<String> =
    LazyLock::new(|| std::env::var("GOOGLE_ANTIGRAVITY_CLIENT_SECRET").unwrap_or_default());
const GOOGLE_ANTIGRAVITY_DEFAULT_PROJECT_ID: &str = "rising-fact-p41fc";
const GOOGLE_ANTIGRAVITY_DAILY_ENDPOINT: &str = "https://daily-cloudcode-pa.sandbox.googleapis.com";
const GOOGLE_ANTIGRAVITY_PROD_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com";
const GOOGLE_ANTIGRAVITY_STREAM_PATH: &str = "/v1internal:streamGenerateContent?alt=sse";
const GOOGLE_ANTIGRAVITY_VERSION_URL: &str =
    "https://antigravity-auto-updater-974169037036.us-central1.run.app";
const GOOGLE_ANTIGRAVITY_CHANGELOG_URL: &str = "https://antigravity.google/changelog";
const GOOGLE_ANTIGRAVITY_VERSION_FALLBACK: &str = "1.18.3";
const GOOGLE_ANTIGRAVITY_VERSION_CACHE_TTL: Duration = Duration::from_secs(6 * 60 * 60);
const GOOGLE_ANTIGRAVITY_VERSION_FETCH_TIMEOUT: Duration = Duration::from_secs(5);
const GOOGLE_ANTIGRAVITY_CHANGELOG_SCAN_CHARS: usize = 5000;
const GOOGLE_ANTIGRAVITY_CLIENT_METADATA: &str =
    r#"{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}"#;
const GOOGLE_ANTIGRAVITY_API_CLIENT_HEADER: &str = "google-cloud-sdk vscode_cloudshelleditor/0.1";
const GOOGLE_ANTIGRAVITY_CLAUDE_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
const GOOGLE_ANTIGRAVITY_ANTIGRAVITY_PROMPT: &str = "You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind team working on Advanced Agentic Coding.\
You are pair programming with a USER to solve their coding task. The task may require creating a new codebase, modifying or debugging an existing codebase, or simply answering a question.\
**Absolute paths only**\
**Proactiveness**";

/// Custom completion model that routes through LlmManager.
///
/// Optionally holds a RoutingConfig for fallback behavior. When present,
/// completion() will try fallback models on retriable errors.
#[derive(Clone)]
pub struct SpacebotModel {
    llm_manager: Arc<LlmManager>,
    model_name: String,
    provider: String,
    full_model_name: String,
    routing: Option<RoutingConfig>,
}

impl SpacebotModel {
    pub fn provider(&self) -> &str {
        &self.provider
    }
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
    pub fn full_model_name(&self) -> &str {
        &self.full_model_name
    }

    /// Attach routing config for fallback behavior.
    pub fn with_routing(mut self, routing: RoutingConfig) -> Self {
        self.routing = Some(routing);
        self
    }

    /// Direct call to the provider (no fallback logic).
    async fn attempt_completion(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        let provider_id = self
            .full_model_name
            .split_once('/')
            .map(|(provider, _)| provider)
            .unwrap_or("anthropic");

        let provider_config = self
            .llm_manager
            .get_provider(provider_id)
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        if provider_id == "google-antigravity" {
            return self
                .call_google_antigravity(request, &provider_config)
                .await;
        }

        if provider_id == "zai-coding-plan" || provider_id == "zhipu" {
            let display_name = if provider_id == "zhipu" {
                "Z.AI (GLM)"
            } else {
                "Z.AI Coding Plan"
            };
            let endpoint = format!(
                "{}/chat/completions",
                provider_config.base_url.trim_end_matches('/')
            );
            return self
                .call_openai_compatible_with_optional_auth(
                    request,
                    display_name,
                    &endpoint,
                    Some(provider_config.api_key.clone()),
                )
                .await;
        }

        match provider_config.api_type {
            ApiType::Anthropic => self.call_anthropic(request, &provider_config).await,
            ApiType::OpenAiCompletions => self.call_openai(request, &provider_config).await,
            ApiType::OpenAiResponses => self.call_openai_responses(request, &provider_config).await,
        }
    }

    /// Try a model with retries and exponential backoff on transient errors.
    ///
    /// Returns `Ok(response)` on success, or `Err((last_error, was_rate_limit))`
    /// after exhausting retries. `was_rate_limit` indicates the final failure was
    /// a 429/rate-limit (as opposed to a timeout or server error), so the caller
    /// can decide whether to record cooldown.
    async fn attempt_with_retries(
        &self,
        model_name: &str,
        request: &CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, (CompletionError, bool)> {
        let model = if model_name == self.full_model_name {
            self.clone()
        } else {
            SpacebotModel::make(&self.llm_manager, model_name)
        };

        let mut last_error = None;
        for attempt in 0..MAX_RETRIES_PER_MODEL {
            if attempt > 0 {
                let delay_ms = RETRY_BASE_DELAY_MS * 2u64.pow((attempt - 1) as u32);
                tracing::debug!(
                    model = %model_name,
                    attempt = attempt + 1,
                    delay_ms,
                    "retrying after backoff"
                );
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }

            match model.attempt_completion(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(error) => {
                    let error_str = error.to_string();
                    if !routing::is_retriable_error(&error_str) {
                        // Non-retriable (auth error, bad request, etc) — bail immediately
                        return Err((error, false));
                    }
                    tracing::warn!(
                        model = %model_name,
                        attempt = attempt + 1,
                        %error,
                        "retriable error"
                    );
                    last_error = Some(error_str);
                }
            }
        }

        let error_str = last_error.unwrap_or_default();
        let was_rate_limit = routing::is_rate_limit_error(&error_str);
        Err((
            CompletionError::ProviderError(format!(
                "{model_name} failed after {MAX_RETRIES_PER_MODEL} attempts: {error_str}"
            )),
            was_rate_limit,
        ))
    }
}

impl CompletionModel for SpacebotModel {
    type Response = RawResponse;
    type StreamingResponse = RawStreamingResponse;
    type Client = Arc<LlmManager>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        let full_name = model.into();

        // OpenRouter model names have the form "openrouter/provider/model",
        // so split on the first "/" only and keep the rest as the model name.
        let (provider, model_name) = if let Some(rest) = full_name.strip_prefix("openrouter/") {
            ("openrouter".to_string(), rest.to_string())
        } else if let Some((p, m)) = full_name.split_once('/') {
            (p.to_string(), m.to_string())
        } else {
            ("anthropic".to_string(), full_name.clone())
        };

        let full_model_name = format!("{provider}/{model_name}");

        Self {
            llm_manager: client.clone(),
            model_name,
            provider,
            full_model_name,
            routing: None,
        }
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();

        let result = async move {
            let Some(routing) = &self.routing else {
                // No routing config — just call the model directly, no fallback/retry
                return self.attempt_completion(request).await;
            };

            let cooldown = routing.rate_limit_cooldown_secs;
            let fallbacks = routing.get_fallbacks(&self.full_model_name);
            let mut last_error: Option<CompletionError> = None;

            // Try the primary model (with retries) unless it's in rate-limit cooldown
            // and we have fallbacks to try instead.
            let primary_rate_limited = self
                .llm_manager
                .is_rate_limited(&self.full_model_name, cooldown)
                .await;

            let skip_primary = primary_rate_limited && !fallbacks.is_empty();

            if skip_primary {
                tracing::debug!(
                    model = %self.full_model_name,
                    "primary model in rate-limit cooldown, skipping to fallbacks"
                );
            } else {
                match self
                    .attempt_with_retries(&self.full_model_name, &request)
                    .await
                {
                    Ok(response) => return Ok(response),
                    Err((error, was_rate_limit)) => {
                        if was_rate_limit {
                            self.llm_manager
                                .record_rate_limit(&self.full_model_name)
                                .await;
                        }
                        if fallbacks.is_empty() {
                            // No fallbacks — this is the final error
                            return Err(error);
                        }
                        tracing::warn!(
                            model = %self.full_model_name,
                            "primary model exhausted retries, trying fallbacks"
                        );
                        last_error = Some(error);
                    }
                }
            }

            // Try fallback chain, each with their own retry loop
            for (index, fallback_name) in fallbacks.iter().take(MAX_FALLBACK_ATTEMPTS).enumerate() {
                if self
                    .llm_manager
                    .is_rate_limited(fallback_name, cooldown)
                    .await
                {
                    tracing::debug!(
                        fallback = %fallback_name,
                        "fallback model in cooldown, skipping"
                    );
                    continue;
                }

                match self.attempt_with_retries(fallback_name, &request).await {
                    Ok(response) => {
                        tracing::info!(
                            original = %self.full_model_name,
                            fallback = %fallback_name,
                            attempt = index + 1,
                            "fallback model succeeded"
                        );
                        return Ok(response);
                    }
                    Err((error, was_rate_limit)) => {
                        if was_rate_limit {
                            self.llm_manager.record_rate_limit(fallback_name).await;
                        }
                        tracing::warn!(
                            fallback = %fallback_name,
                            "fallback model exhausted retries, continuing chain"
                        );
                        last_error = Some(error);
                    }
                }
            }

            Err(last_error.unwrap_or_else(|| {
                CompletionError::ProviderError("all models in fallback chain failed".into())
            }))
        }
        .await;

        #[cfg(feature = "metrics")]
        {
            let elapsed = start.elapsed().as_secs_f64();
            let metrics = crate::telemetry::Metrics::global();
            // TODO: agent_id and tier are "unknown" because SpacebotModel doesn't
            // carry process context. Thread agent_id/ProcessType through to get
            // per-agent, per-tier breakdowns.
            metrics
                .llm_requests_total
                .with_label_values(&["unknown", &self.full_model_name, "unknown"])
                .inc();
            metrics
                .llm_request_duration_seconds
                .with_label_values(&["unknown", &self.full_model_name, "unknown"])
                .observe(elapsed);
        }

        result
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<RawStreamingResponse>, CompletionError> {
        Err(CompletionError::ProviderError(
            "streaming not yet implemented".into(),
        ))
    }
}

impl SpacebotModel {
    async fn call_google_antigravity(
        &self,
        request: CompletionRequest,
        provider_config: &ProviderConfig,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        let mut credentials =
            parse_google_antigravity_api_key_payload(provider_config.api_key.as_str())?;

        if credentials.project_id.trim().is_empty() {
            credentials.project_id = GOOGLE_ANTIGRAVITY_DEFAULT_PROJECT_ID.to_string();
        }

        let endpoints = google_antigravity_endpoints(provider_config.base_url.as_str());
        let request_body = build_google_antigravity_request_body(
            &self.model_name,
            &request,
            &credentials.project_id,
        );
        let include_claude_thinking_header =
            google_antigravity_is_claude_thinking_model(&self.model_name);

        let mut access_token = self
            .google_antigravity_access_token(&credentials, false)
            .await?;
        let mut forced_refresh_attempted = false;
        let mut last_error: Option<String> = None;

        for endpoint in endpoints {
            let request_url = format!(
                "{}{GOOGLE_ANTIGRAVITY_STREAM_PATH}",
                endpoint.trim_end_matches('/')
            );

            let mut response = self
                .send_google_antigravity_request(
                    &request_url,
                    &access_token,
                    &request_body,
                    include_claude_thinking_header,
                )
                .await?;

            if response.status() == reqwest::StatusCode::UNAUTHORIZED
                && !credentials.refresh_token.is_empty()
                && !forced_refresh_attempted
            {
                access_token = self
                    .google_antigravity_access_token(&credentials, true)
                    .await?;
                forced_refresh_attempted = true;

                response = self
                    .send_google_antigravity_request(
                        &request_url,
                        &access_token,
                        &request_body,
                        include_claude_thinking_header,
                    )
                    .await?;
            }

            let status = response.status();
            let response_text = response.text().await.map_err(|error| {
                CompletionError::ProviderError(format!(
                    "failed to read Google Antigravity response body: {error}"
                ))
            })?;

            if !status.is_success() {
                let error_message = extract_google_antigravity_error_message(&response_text);
                last_error = Some(format!(
                    "Google Antigravity API error ({status}) at {request_url}: {error_message}"
                ));
                continue;
            }

            return parse_google_antigravity_sse_response(&response_text);
        }

        Err(CompletionError::ProviderError(last_error.unwrap_or_else(
            || "Google Antigravity request failed for all configured endpoints".to_string(),
        )))
    }

    async fn send_google_antigravity_request(
        &self,
        request_url: &str,
        access_token: &str,
        request_body: &serde_json::Value,
        include_claude_thinking_header: bool,
    ) -> Result<reqwest::Response, CompletionError> {
        let antigravity_version =
            resolve_google_antigravity_version(self.llm_manager.http_client()).await;
        let mut request_builder = self
            .llm_manager
            .http_client()
            .post(request_url)
            .header("authorization", format!("Bearer {access_token}"))
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .header(
                "user-agent",
                format!("antigravity/{antigravity_version} darwin/arm64"),
            )
            .header("x-goog-api-client", GOOGLE_ANTIGRAVITY_API_CLIENT_HEADER)
            .header("client-metadata", GOOGLE_ANTIGRAVITY_CLIENT_METADATA);

        if include_claude_thinking_header {
            request_builder =
                request_builder.header("anthropic-beta", GOOGLE_ANTIGRAVITY_CLAUDE_THINKING_BETA);
        }

        request_builder
            .json(request_body)
            .send()
            .await
            .map_err(|error| CompletionError::ProviderError(error.to_string()))
    }

    async fn google_antigravity_access_token(
        &self,
        credentials: &GoogleAntigravityApiKeyPayload,
        force_refresh: bool,
    ) -> Result<String, CompletionError> {
        let now_ms = current_epoch_millis();
        let refresh_key = credentials.refresh_token.trim();
        let expires_at_ms = normalize_google_antigravity_expires_at_ms(credentials.expires_at);

        if !refresh_key.is_empty() {
            if GOOGLE_ANTIGRAVITY_CLIENT_ID.is_empty() || GOOGLE_ANTIGRAVITY_CLIENT_SECRET.is_empty()
            {
                return Err(CompletionError::ProviderError(
                    "Google Antigravity OAuth client credentials are not configured. \
Set GOOGLE_ANTIGRAVITY_CLIENT_ID and GOOGLE_ANTIGRAVITY_CLIENT_SECRET.".to_string(),
                ));
            }
            if !force_refresh {
                if let Some(cached_token) = GOOGLE_ANTIGRAVITY_ACCESS_TOKEN_CACHE
                    .read()
                    .await
                    .get(refresh_key)
                    .cloned()
                {
                    if cached_token.expires_at_ms > now_ms + 60_000 {
                        return Ok(cached_token.token);
                    }
                }

                if !credentials.token.trim().is_empty() {
                    if let Some(expires_at_ms) = expires_at_ms {
                        if expires_at_ms > now_ms + 60_000 {
                            GOOGLE_ANTIGRAVITY_ACCESS_TOKEN_CACHE.write().await.insert(
                                refresh_key.to_string(),
                                CachedGoogleAntigravityToken {
                                    token: credentials.token.clone(),
                                    expires_at_ms,
                                },
                            );
                            return Ok(credentials.token.clone());
                        }
                    } else {
                        return Ok(credentials.token.clone());
                    }
                }
            }

            let token_response = self
                .llm_manager
                .http_client()
                .post(GOOGLE_ANTIGRAVITY_TOKEN_URL)
                .header("content-type", "application/x-www-form-urlencoded")
                .form(&[
                    ("client_id", GOOGLE_ANTIGRAVITY_CLIENT_ID.as_str()),
                    ("client_secret", GOOGLE_ANTIGRAVITY_CLIENT_SECRET.as_str()),
                    ("refresh_token", refresh_key),
                    ("grant_type", "refresh_token"),
                ])
                .send()
                .await
                .map_err(|error| {
                    CompletionError::ProviderError(format!(
                        "Google Antigravity token refresh request failed: {error}"
                    ))
                })?;

            let status = token_response.status();
            let response_text = token_response.text().await.map_err(|error| {
                CompletionError::ProviderError(format!(
                    "failed to read Google Antigravity refresh response: {error}"
                ))
            })?;

            let response_body: serde_json::Value =
                serde_json::from_str(&response_text).map_err(|error| {
                    CompletionError::ProviderError(format!(
                        "Google Antigravity refresh response ({status}) is not valid JSON: {error}\nBody: {}",
                        truncate_body(&response_text)
                    ))
                })?;

            if !status.is_success() {
                let error_message = response_body["error_description"]
                    .as_str()
                    .or_else(|| response_body["error"]["message"].as_str())
                    .or_else(|| response_body["error"].as_str())
                    .unwrap_or("unknown error");
                return Err(CompletionError::ProviderError(format!(
                    "Google Antigravity token refresh failed ({status}): {error_message}"
                )));
            }

            let refreshed_token = response_body["access_token"]
                .as_str()
                .unwrap_or("")
                .trim()
                .to_string();
            if refreshed_token.is_empty() {
                return Err(CompletionError::ProviderError(
                    "Google Antigravity token refresh succeeded but returned no access_token"
                        .to_string(),
                ));
            }

            let expires_in_seconds = response_body["expires_in"].as_i64().unwrap_or(3600).max(60);
            let expires_at_ms = now_ms
                .saturating_add(expires_in_seconds.saturating_mul(1000))
                .saturating_sub(5 * 60 * 1000);

            GOOGLE_ANTIGRAVITY_ACCESS_TOKEN_CACHE.write().await.insert(
                refresh_key.to_string(),
                CachedGoogleAntigravityToken {
                    token: refreshed_token.clone(),
                    expires_at_ms,
                },
            );

            return Ok(refreshed_token);
        }

        let inline_token = credentials.token.trim();
        if inline_token.is_empty() {
            return Err(CompletionError::ProviderError(
                "Google Antigravity credentials are missing both token and refresh token"
                    .to_string(),
            ));
        }

        Ok(inline_token.to_string())
    }

    async fn call_anthropic(
        &self,
        request: CompletionRequest,
        provider_config: &ProviderConfig,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        let base_url = provider_config.base_url.trim_end_matches('/');
        let messages_url = format!("{base_url}/v1/messages");
        let api_key = provider_config.api_key.as_str();

        let messages = convert_messages_to_anthropic(&request.chat_history);

        let mut body = serde_json::json!({
            "model": self.model_name,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(4096),
        });

        if let Some(preamble) = &request.preamble {
            body["system"] = serde_json::json!(preamble);
        }

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.parameters,
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);
        }

        let is_oauth = api_key.starts_with("sk-ant-oat");
        let mut req = self
            .llm_manager
            .http_client()
            .post(&messages_url)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        if is_oauth {
            req = req
                .header("authorization", format!("Bearer {api_key}"))
                .header("anthropic-beta", "oauth-2025-04-20");
        } else {
            req = req.header("x-api-key", api_key);
        }

        let response = req
            .json(&body)
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let status = response.status();
        let response_text = response.text().await.map_err(|e| {
            CompletionError::ProviderError(format!("failed to read response body: {e}"))
        })?;

        let response_body: serde_json::Value =
            serde_json::from_str(&response_text).map_err(|e| {
                CompletionError::ProviderError(format!(
                    "Anthropic response ({status}) is not valid JSON: {e}\nBody: {}",
                    truncate_body(&response_text)
                ))
            })?;

        if !status.is_success() {
            let message = response_body["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            return Err(CompletionError::ProviderError(format!(
                "Anthropic API error ({status}): {message}"
            )));
        }

        parse_anthropic_response(response_body)
    }

    async fn call_openai(
        &self,
        request: CompletionRequest,
        provider_config: &ProviderConfig,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        let api_key = provider_config.api_key.as_str();

        let mut messages = Vec::new();

        if let Some(preamble) = &request.preamble {
            messages.push(serde_json::json!({
                "role": "system",
                "content": preamble,
            }));
        }

        messages.extend(convert_messages_to_openai(&request.chat_history));

        let mut body = serde_json::json!({
            "model": self.model_name,
            "messages": messages,
        });

        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);
        }

        let chat_completions_url = format!(
            "{}/v1/chat/completions",
            provider_config.base_url.trim_end_matches('/')
        );

        let mut request_builder = self
            .llm_manager
            .http_client()
            .post(&chat_completions_url)
            .header("authorization", format!("Bearer {api_key}"))
            .header("content-type", "application/json");

        // Kimi endpoints require a specific user-agent header.
        if chat_completions_url.contains("kimi.com") || chat_completions_url.contains("moonshot.ai")
        {
            request_builder = request_builder.header("user-agent", "KimiCLI/1.3");
        }

        let response = request_builder
            .json(&body)
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let status = response.status();
        let response_text = response.text().await.map_err(|e| {
            CompletionError::ProviderError(format!("failed to read response body: {e}"))
        })?;

        let response_body: serde_json::Value =
            serde_json::from_str(&response_text).map_err(|e| {
                CompletionError::ProviderError(format!(
                    "OpenAI response ({status}) is not valid JSON: {e}\nBody: {}",
                    truncate_body(&response_text)
                ))
            })?;

        if !status.is_success() {
            let message = response_body["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            return Err(CompletionError::ProviderError(format!(
                "OpenAI API error ({status}): {message}"
            )));
        }

        parse_openai_response(response_body, "OpenAI")
    }

    async fn call_openai_responses(
        &self,
        request: CompletionRequest,
        provider_config: &ProviderConfig,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        let base_url = provider_config.base_url.trim_end_matches('/');
        let responses_url = format!("{base_url}/v1/responses");
        let api_key = provider_config.api_key.as_str();

        let input = convert_messages_to_openai_responses(&request.chat_history);

        let mut body = serde_json::json!({
            "model": self.model_name,
            "input": input,
        });

        if let Some(preamble) = &request.preamble {
            body["instructions"] = serde_json::json!(preamble);
        }

        if let Some(max_tokens) = request.max_tokens {
            body["max_output_tokens"] = serde_json::json!(max_tokens);
        }

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|tool_definition| {
                    serde_json::json!({
                        "type": "function",
                        "name": tool_definition.name,
                        "description": tool_definition.description,
                        "parameters": tool_definition.parameters,
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);
        }

        let response = self
            .llm_manager
            .http_client()
            .post(&responses_url)
            .header("authorization", format!("Bearer {api_key}"))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let status = response.status();
        let response_text = response.text().await.map_err(|e| {
            CompletionError::ProviderError(format!("failed to read response body: {e}"))
        })?;

        let response_body: serde_json::Value =
            serde_json::from_str(&response_text).map_err(|e| {
                CompletionError::ProviderError(format!(
                    "OpenAI Responses API response ({status}) is not valid JSON: {e}\nBody: {}",
                    truncate_body(&response_text)
                ))
            })?;

        if !status.is_success() {
            let message = response_body["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            return Err(CompletionError::ProviderError(format!(
                "OpenAI Responses API error ({status}): {message}"
            )));
        }

        parse_openai_responses_response(response_body)
    }

    /// Generic OpenAI-compatible API call.
    /// Used by providers that implement the OpenAI chat completions format.
    async fn call_openai_compatible(
        &self,
        request: CompletionRequest,
        provider_display_name: &str,
        provider_config: &ProviderConfig,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        let base_url = provider_config.base_url.trim_end_matches('/');
        let endpoint_path = match provider_config.api_type {
            ApiType::OpenAiCompletions | ApiType::OpenAiResponses => "/v1/chat/completions",
            ApiType::Anthropic => {
                return Err(CompletionError::ProviderError(format!(
                    "{provider_display_name} is configured with anthropic API type, but this call expects an OpenAI-compatible API"
                )));
            }
        };
        let endpoint = format!("{base_url}{endpoint_path}");
        let api_key = provider_config.api_key.as_str();

        let mut messages = Vec::new();

        if let Some(preamble) = &request.preamble {
            messages.push(serde_json::json!({
                "role": "system",
                "content": preamble,
            }));
        }

        messages.extend(convert_messages_to_openai(&request.chat_history));

        let mut body = serde_json::json!({
            "model": self.model_name,
            "messages": messages,
        });

        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);
        }

        let response = self
            .llm_manager
            .http_client()
            .post(&endpoint)
            .header("authorization", format!("Bearer {api_key}"))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let status = response.status();
        let response_text = response.text().await.map_err(|e| {
            CompletionError::ProviderError(format!("failed to read response body: {e}"))
        })?;

        let response_body: serde_json::Value =
            serde_json::from_str(&response_text).map_err(|e| {
                CompletionError::ProviderError(format!(
                    "{provider_display_name} response ({status}) is not valid JSON: {e}\nBody: {}",
                    truncate_body(&response_text)
                ))
            })?;

        if !status.is_success() {
            let message = response_body["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            return Err(CompletionError::ProviderError(format!(
                "{provider_display_name} API error ({status}): {message}"
            )));
        }

        parse_openai_response(response_body, provider_display_name)
    }

    /// Generic OpenAI-compatible API call with optional bearer auth.
    async fn call_openai_compatible_with_optional_auth(
        &self,
        request: CompletionRequest,
        provider_display_name: &str,
        endpoint: &str,
        api_key: Option<String>,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        let mut messages = Vec::new();

        if let Some(preamble) = &request.preamble {
            messages.push(serde_json::json!({
                "role": "system",
                "content": preamble,
            }));
        }

        messages.extend(convert_messages_to_openai(&request.chat_history));

        let mut body = serde_json::json!({
            "model": self.model_name,
            "messages": messages,
        });

        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);
        }

        let response = self.llm_manager.http_client().post(endpoint);

        let response = if let Some(api_key) = api_key {
            response.header("authorization", format!("Bearer {api_key}"))
        } else {
            response
        };

        let response = response
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let status = response.status();
        let response_text = response.text().await.map_err(|e| {
            CompletionError::ProviderError(format!("failed to read response body: {e}"))
        })?;

        let response_body: serde_json::Value =
            serde_json::from_str(&response_text).map_err(|e| {
                CompletionError::ProviderError(format!(
                    "{provider_display_name} response ({status}) is not valid JSON: {e}\nBody: {}",
                    truncate_body(&response_text)
                ))
            })?;

        if !status.is_success() {
            let message = response_body["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            return Err(CompletionError::ProviderError(format!(
                "{provider_display_name} API error ({status}): {message}"
            )));
        }

        parse_openai_response(response_body, provider_display_name)
    }
}
// --- Helpers ---

fn normalize_ollama_base_url(configured: Option<String>) -> String {
    let mut base_url = configured
        .unwrap_or_else(|| "http://localhost:11434".to_string())
        .trim()
        .trim_end_matches('/')
        .to_string();

    if base_url.ends_with("/api") {
        base_url.truncate(base_url.len() - "/api".len());
    } else if base_url.ends_with("/v1") {
        base_url.truncate(base_url.len() - "/v1".len());
    }

    base_url
}

fn current_epoch_millis() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_millis() as i64,
        Err(_) => 0,
    }
}

fn extract_semver_from_text(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut start = 0usize;

    while start < bytes.len() {
        if !bytes[start].is_ascii_digit() {
            start += 1;
            continue;
        }

        let mut index = start;
        let mut dots = 0usize;
        let mut previous_was_dot = false;

        while index < bytes.len() {
            let byte = bytes[index];
            if byte.is_ascii_digit() {
                previous_was_dot = false;
                index += 1;
                continue;
            }
            if byte == b'.' && !previous_was_dot && dots < 2 {
                dots += 1;
                previous_was_dot = true;
                index += 1;
                continue;
            }
            break;
        }

        if dots == 2 && !previous_was_dot {
            return Some(text[start..index].to_string());
        }

        start = index.saturating_add(1);
    }

    None
}

async fn fetch_google_antigravity_version_from_url(
    client: &reqwest::Client,
    url: &str,
    max_chars: Option<usize>,
) -> Option<String> {
    let response = client
        .get(url)
        .timeout(GOOGLE_ANTIGRAVITY_VERSION_FETCH_TIMEOUT)
        .send()
        .await
        .ok()?;

    if !response.status().is_success() {
        return None;
    }

    let mut body = response.text().await.ok()?;
    if let Some(max_chars) = max_chars {
        body.truncate(body.len().min(max_chars));
    }

    extract_semver_from_text(&body)
}

async fn resolve_google_antigravity_version(client: &reqwest::Client) -> String {
    {
        let cache = GOOGLE_ANTIGRAVITY_VERSION_CACHE.read().await;
        if let Some(entry) = cache.as_ref() {
            if entry.fetched_at.elapsed() < GOOGLE_ANTIGRAVITY_VERSION_CACHE_TTL {
                return entry.version.clone();
            }
        }
    }

    let version = if let Some(version) =
        fetch_google_antigravity_version_from_url(client, GOOGLE_ANTIGRAVITY_VERSION_URL, None)
            .await
    {
        version
    } else if let Some(version) = fetch_google_antigravity_version_from_url(
        client,
        GOOGLE_ANTIGRAVITY_CHANGELOG_URL,
        Some(GOOGLE_ANTIGRAVITY_CHANGELOG_SCAN_CHARS),
    )
    .await
    {
        version
    } else {
        GOOGLE_ANTIGRAVITY_VERSION_FALLBACK.to_string()
    };

    let mut cache = GOOGLE_ANTIGRAVITY_VERSION_CACHE.write().await;
    *cache = Some(CachedGoogleAntigravityVersion {
        version: version.clone(),
        fetched_at: Instant::now(),
    });

    version
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

fn google_antigravity_requires_tool_call_id(model_name: &str) -> bool {
    model_name.starts_with("claude-") || model_name.starts_with("gpt-oss-")
}

fn google_antigravity_is_claude_thinking_model(model_name: &str) -> bool {
    let normalized = model_name.to_ascii_lowercase();
    normalized.contains("claude") && normalized.contains("thinking")
}

fn google_antigravity_endpoints(configured_base_url: &str) -> Vec<String> {
    let mut endpoints = Vec::new();

    let configured = configured_base_url.trim().trim_end_matches('/');
    if !configured.is_empty() {
        endpoints.push(configured.to_string());
    }

    for fallback in [GOOGLE_ANTIGRAVITY_DAILY_ENDPOINT] {
        if !endpoints.iter().any(|endpoint| endpoint == fallback) {
            endpoints.push(fallback.to_string());
        }
    }

    endpoints
}

fn parse_google_antigravity_api_key_payload(
    raw_api_key: &str,
) -> Result<GoogleAntigravityApiKeyPayload, CompletionError> {
    let trimmed = raw_api_key.trim();
    if trimmed.is_empty() {
        return Err(CompletionError::ProviderError(
            "Google Antigravity API key is empty".to_string(),
        ));
    }

    if let Ok(mut payload) = serde_json::from_str::<GoogleAntigravityApiKeyPayload>(trimmed) {
        if payload.project_id.trim().is_empty() {
            payload.project_id = GOOGLE_ANTIGRAVITY_DEFAULT_PROJECT_ID.to_string();
        }
        return Ok(payload);
    }

    Ok(GoogleAntigravityApiKeyPayload {
        token: trimmed.to_string(),
        project_id: GOOGLE_ANTIGRAVITY_DEFAULT_PROJECT_ID.to_string(),
        refresh_token: String::new(),
        expires_at: None,
    })
}

fn extract_google_antigravity_error_message(response_text: &str) -> String {
    if let Ok(response_body) = serde_json::from_str::<serde_json::Value>(response_text) {
        if let Some(message) = response_body["error"]["message"].as_str() {
            return message.to_string();
        }
        if let Some(message) = response_body["error_description"].as_str() {
            return message.to_string();
        }
        if let Some(message) = response_body["error"].as_str() {
            return message.to_string();
        }
    }

    truncate_body(response_text).to_string()
}

fn build_google_antigravity_request_body(
    model_name: &str,
    request: &CompletionRequest,
    project_id: &str,
) -> serde_json::Value {
    let include_tool_call_id = google_antigravity_requires_tool_call_id(model_name);
    let contents =
        convert_messages_to_google_antigravity(&request.chat_history, include_tool_call_id);

    let mut request_payload = serde_json::json!({
        "contents": contents,
    });

    let mut system_parts = vec![
        serde_json::json!({
            "text": GOOGLE_ANTIGRAVITY_ANTIGRAVITY_PROMPT,
        }),
        serde_json::json!({
            "text": format!(
                "Please ignore following [ignore]{GOOGLE_ANTIGRAVITY_ANTIGRAVITY_PROMPT}[/ignore]"
            ),
        }),
    ];

    if let Some(preamble) = &request.preamble {
        if !preamble.trim().is_empty() {
            system_parts.push(serde_json::json!({
                "text": preamble,
            }));
        }
    }

    request_payload["systemInstruction"] = serde_json::json!({
        "role": "user",
        "parts": system_parts,
    });

    let mut generation_config = serde_json::Map::new();
    if let Some(max_tokens) = request.max_tokens {
        generation_config.insert("maxOutputTokens".to_string(), serde_json::json!(max_tokens));
    }
    if let Some(temperature) = request.temperature {
        generation_config.insert("temperature".to_string(), serde_json::json!(temperature));
    }
    if !generation_config.is_empty() {
        request_payload["generationConfig"] = serde_json::Value::Object(generation_config);
    }

    if !request.tools.is_empty() {
        let use_legacy_parameters = model_name.starts_with("claude-");
        let function_declarations: Vec<serde_json::Value> = request
            .tools
            .iter()
            .map(|tool| {
                if use_legacy_parameters {
                    serde_json::json!({
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    })
                } else {
                    serde_json::json!({
                        "name": tool.name,
                        "description": tool.description,
                        "parametersJsonSchema": tool.parameters,
                    })
                }
            })
            .collect();

        request_payload["tools"] = serde_json::json!([
            {
                "functionDeclarations": function_declarations,
            }
        ]);
    }

    serde_json::json!({
        "project": project_id,
        "model": model_name,
        "request": request_payload,
        "requestType": "agent",
        "userAgent": "antigravity",
        "requestId": format!("agent-{}-{}", current_epoch_millis(), Uuid::new_v4().simple()),
    })
}

fn tool_result_content_to_string(content: &OneOrMany<rig::message::ToolResultContent>) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            rig::message::ToolResultContent::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// --- Message conversion ---

fn convert_messages_to_anthropic(messages: &OneOrMany<Message>) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|message| match message {
            Message::User { content } => {
                let parts: Vec<serde_json::Value> = content
                    .iter()
                    .filter_map(|c| match c {
                        UserContent::Text(t) => {
                            Some(serde_json::json!({"type": "text", "text": t.text}))
                        }
                        UserContent::Image(image) => convert_image_anthropic(image),
                        UserContent::ToolResult(result) => Some(serde_json::json!({
                            "type": "tool_result",
                            "tool_use_id": result.id,
                            "content": tool_result_content_to_string(&result.content),
                        })),
                        _ => None,
                    })
                    .collect();
                serde_json::json!({"role": "user", "content": parts})
            }
            Message::Assistant { content, .. } => {
                let parts: Vec<serde_json::Value> = content
                    .iter()
                    .filter_map(|c| match c {
                        AssistantContent::Text(t) => {
                            Some(serde_json::json!({"type": "text", "text": t.text}))
                        }
                        AssistantContent::ToolCall(tc) => Some(serde_json::json!({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": tc.function.arguments,
                        })),
                        _ => None,
                    })
                    .collect();
                serde_json::json!({"role": "assistant", "content": parts})
            }
        })
        .collect()
}

fn convert_messages_to_openai(messages: &OneOrMany<Message>) -> Vec<serde_json::Value> {
    let mut result = Vec::new();

    for message in messages.iter() {
        match message {
            Message::User { content } => {
                // Separate tool results (they need their own messages) from content parts
                let mut content_parts: Vec<serde_json::Value> = Vec::new();
                let mut tool_results: Vec<serde_json::Value> = Vec::new();

                for item in content.iter() {
                    match item {
                        UserContent::Text(t) => {
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": t.text,
                            }));
                        }
                        UserContent::Image(image) => {
                            if let Some(part) = convert_image_openai(image) {
                                content_parts.push(part);
                            }
                        }
                        UserContent::ToolResult(tr) => {
                            tool_results.push(serde_json::json!({
                                "role": "tool",
                                "tool_call_id": tr.id,
                                "content": tool_result_content_to_string(&tr.content),
                            }));
                        }
                        _ => {}
                    }
                }

                if !content_parts.is_empty() {
                    // If there's only one text part and no images, use simple string format
                    if content_parts.len() == 1 && content_parts[0]["type"] == "text" {
                        result.push(serde_json::json!({
                            "role": "user",
                            "content": content_parts[0]["text"],
                        }));
                    } else {
                        // Mixed content (text + images): use array-of-parts format
                        result.push(serde_json::json!({
                            "role": "user",
                            "content": content_parts,
                        }));
                    }
                }

                result.extend(tool_results);
            }
            Message::Assistant { content, .. } => {
                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();

                for item in content.iter() {
                    match item {
                        AssistantContent::Text(t) => {
                            text_parts.push(t.text.clone());
                        }
                        AssistantContent::ToolCall(tc) => {
                            // OpenAI expects arguments as a JSON string
                            let args_string = serde_json::to_string(&tc.function.arguments)
                                .unwrap_or_else(|_| "{}".to_string());
                            tool_calls.push(serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": args_string,
                                }
                            }));
                        }
                        _ => {}
                    }
                }

                let mut msg = serde_json::json!({"role": "assistant"});
                if !text_parts.is_empty() {
                    msg["content"] = serde_json::json!(text_parts.join("\n"));
                }
                if !tool_calls.is_empty() {
                    msg["tool_calls"] = serde_json::json!(tool_calls);
                }
                result.push(msg);
            }
        }
    }

    result
}

fn convert_messages_to_openai_responses(messages: &OneOrMany<Message>) -> Vec<serde_json::Value> {
    let mut result = Vec::new();

    for message in messages.iter() {
        match message {
            Message::User { content } => {
                let mut content_parts = Vec::new();

                for item in content.iter() {
                    match item {
                        UserContent::Text(text) => {
                            content_parts.push(serde_json::json!({
                                "type": "input_text",
                                "text": text.text,
                            }));
                        }
                        UserContent::Image(image) => {
                            if let Some(part) = convert_image_openai_responses(image) {
                                content_parts.push(part);
                            }
                        }
                        UserContent::ToolResult(tool_result) => {
                            result.push(serde_json::json!({
                                "type": "function_call_output",
                                "call_id": tool_result.id,
                                "output": tool_result_content_to_string(&tool_result.content),
                            }));
                        }
                        _ => {}
                    }
                }

                if !content_parts.is_empty() {
                    result.push(serde_json::json!({
                        "role": "user",
                        "content": content_parts,
                    }));
                }
            }
            Message::Assistant { content, .. } => {
                let mut text_parts = Vec::new();

                for item in content.iter() {
                    match item {
                        AssistantContent::Text(text) => {
                            text_parts.push(serde_json::json!({
                                "type": "output_text",
                                "text": text.text,
                            }));
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            let arguments = serde_json::to_string(&tool_call.function.arguments)
                                .unwrap_or_else(|_| "{}".to_string());
                            result.push(serde_json::json!({
                                "type": "function_call",
                                "name": tool_call.function.name,
                                "arguments": arguments,
                                "call_id": tool_call.id,
                            }));
                        }
                        _ => {}
                    }
                }

                if !text_parts.is_empty() {
                    result.push(serde_json::json!({
                        "role": "assistant",
                        "content": text_parts,
                    }));
                }
            }
        }
    }

    result
}

fn convert_messages_to_google_antigravity(
    messages: &OneOrMany<Message>,
    include_tool_call_id: bool,
) -> Vec<serde_json::Value> {
    let mut result = Vec::new();
    let mut tool_call_names: HashMap<String, String> = HashMap::new();

    for message in messages.iter() {
        match message {
            Message::User { content } => {
                let mut user_parts = Vec::new();
                let mut tool_result_parts = Vec::new();

                for item in content.iter() {
                    match item {
                        UserContent::Text(text) => {
                            if !text.text.is_empty() {
                                user_parts.push(serde_json::json!({
                                    "text": text.text,
                                }));
                            }
                        }
                        UserContent::Image(image) => {
                            if let Some(part) = convert_image_google_antigravity(image) {
                                user_parts.push(part);
                            }
                        }
                        UserContent::ToolResult(tool_result) => {
                            let tool_name = tool_call_names
                                .get(&tool_result.id)
                                .cloned()
                                .unwrap_or_else(|| "tool".to_string());

                            let mut tool_result_part = serde_json::json!({
                                "functionResponse": {
                                    "name": tool_name,
                                    "response": {
                                        "output": tool_result_content_to_string(&tool_result.content),
                                    }
                                }
                            });

                            if include_tool_call_id {
                                tool_result_part["functionResponse"]["id"] =
                                    serde_json::json!(tool_result.id);
                            }

                            tool_result_parts.push(tool_result_part);
                        }
                        _ => {}
                    }
                }

                if !user_parts.is_empty() {
                    result.push(serde_json::json!({
                        "role": "user",
                        "parts": user_parts,
                    }));
                }

                if !tool_result_parts.is_empty() {
                    if let Some(last) = result.last_mut() {
                        let should_merge = last["role"].as_str() == Some("user")
                            && last["parts"]
                                .as_array()
                                .map(|parts| {
                                    parts
                                        .iter()
                                        .all(|part| part.get("functionResponse").is_some())
                                })
                                .unwrap_or(false);

                        if should_merge {
                            if let Some(parts) = last["parts"].as_array_mut() {
                                parts.extend(tool_result_parts);
                            }
                        } else {
                            result.push(serde_json::json!({
                                "role": "user",
                                "parts": tool_result_parts,
                            }));
                        }
                    } else {
                        result.push(serde_json::json!({
                            "role": "user",
                            "parts": tool_result_parts,
                        }));
                    }
                }
            }
            Message::Assistant { content, .. } => {
                let mut assistant_parts = Vec::new();

                for item in content.iter() {
                    match item {
                        AssistantContent::Text(text) => {
                            if !text.text.trim().is_empty() {
                                assistant_parts.push(serde_json::json!({
                                    "text": text.text,
                                }));
                            }
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            tool_call_names
                                .insert(tool_call.id.clone(), tool_call.function.name.clone());

                            let mut function_call = serde_json::json!({
                                "name": tool_call.function.name,
                                "args": tool_call.function.arguments,
                            });

                            if include_tool_call_id {
                                function_call["id"] = serde_json::json!(tool_call.id);
                            }

                            assistant_parts.push(serde_json::json!({
                                "functionCall": function_call,
                            }));
                        }
                        _ => {}
                    }
                }

                if !assistant_parts.is_empty() {
                    result.push(serde_json::json!({
                        "role": "model",
                        "parts": assistant_parts,
                    }));
                }
            }
        }
    }

    result
}

// --- Image conversion helpers ---

/// Convert a rig Image to an Anthropic image content block.
/// Anthropic format: {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}
fn convert_image_anthropic(image: &Image) -> Option<serde_json::Value> {
    let media_type = image
        .media_type
        .as_ref()
        .map(|mt| mt.to_mime_type())
        .unwrap_or("image/jpeg");

    match &image.data {
        DocumentSourceKind::Base64(data) => Some(serde_json::json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            }
        })),
        DocumentSourceKind::Url(url) => Some(serde_json::json!({
            "type": "image",
            "source": {
                "type": "url",
                "url": url,
            }
        })),
        _ => None,
    }
}

/// Convert a rig Image to an OpenAI image_url content part.
/// OpenAI/OpenRouter format: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
fn convert_image_openai(image: &Image) -> Option<serde_json::Value> {
    let media_type = image
        .media_type
        .as_ref()
        .map(|mt| mt.to_mime_type())
        .unwrap_or("image/jpeg");

    match &image.data {
        DocumentSourceKind::Base64(data) => {
            let data_url = format!("data:{media_type};base64,{data}");
            Some(serde_json::json!({
                "type": "image_url",
                "image_url": { "url": data_url }
            }))
        }
        DocumentSourceKind::Url(url) => Some(serde_json::json!({
            "type": "image_url",
            "image_url": { "url": url }
        })),
        _ => None,
    }
}

fn convert_image_openai_responses(image: &Image) -> Option<serde_json::Value> {
    let media_type = image
        .media_type
        .as_ref()
        .map(|mime_type| mime_type.to_mime_type())
        .unwrap_or("image/jpeg");

    match &image.data {
        DocumentSourceKind::Base64(data) => {
            let data_url = format!("data:{media_type};base64,{data}");
            Some(serde_json::json!({
                "type": "input_image",
                "image_url": data_url,
            }))
        }
        DocumentSourceKind::Url(url) => Some(serde_json::json!({
            "type": "input_image",
            "image_url": url,
        })),
        _ => None,
    }
}

fn convert_image_google_antigravity(image: &Image) -> Option<serde_json::Value> {
    let media_type = image
        .media_type
        .as_ref()
        .map(|mime_type| mime_type.to_mime_type())
        .unwrap_or("image/jpeg");

    match &image.data {
        DocumentSourceKind::Base64(data) => Some(serde_json::json!({
            "inlineData": {
                "mimeType": media_type,
                "data": data,
            }
        })),
        DocumentSourceKind::Url(url) => Some(serde_json::json!({
            "text": format!("Image URL: {url}"),
        })),
        _ => None,
    }
}

/// Truncate a response body for error messages to avoid dumping megabytes of HTML.
fn truncate_body(body: &str) -> &str {
    let limit = 500;
    if body.len() <= limit {
        body
    } else {
        &body[..limit]
    }
}

// --- Response parsing ---

fn make_tool_call(id: String, name: String, arguments: serde_json::Value) -> ToolCall {
    ToolCall {
        id,
        call_id: None,
        function: ToolFunction {
            name: name.trim().to_string(),
            arguments,
        },
        signature: None,
        additional_params: None,
    }
}

fn parse_google_antigravity_sse_response(
    sse_body: &str,
) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
    let mut text_output = String::new();
    let mut reasoning_output = String::new();
    let mut tool_calls = Vec::new();
    let mut seen_tool_call_ids = HashSet::new();
    let mut generated_tool_call_index = 0_u64;
    let mut saw_sse_payload = false;

    let mut input_tokens = 0_u64;
    let mut output_tokens = 0_u64;
    let mut cached_input_tokens = 0_u64;
    let mut total_tokens = 0_u64;
    let mut finish_reason: Option<String> = None;

    for line in sse_body.lines() {
        let trimmed_line = line.trim();
        if !trimmed_line.starts_with("data:") {
            continue;
        }

        let data = trimmed_line.trim_start_matches("data:").trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }

        let chunk: serde_json::Value = match serde_json::from_str(data) {
            Ok(chunk) => chunk,
            Err(_) => continue,
        };

        saw_sse_payload = true;

        if let Some(reason) = chunk["response"]["candidates"][0]["finishReason"].as_str() {
            finish_reason = Some(reason.to_string());
        }

        if let Some(parts) = chunk["response"]["candidates"][0]["content"]["parts"].as_array() {
            for part in parts {
                if let Some(text) = part["text"].as_str() {
                    if !text.is_empty() {
                        if part["thought"].as_bool().unwrap_or(false) {
                            reasoning_output.push_str(text);
                        } else {
                            text_output.push_str(text);
                        }
                    }
                }

                if let Some(function_call) = part["functionCall"].as_object() {
                    let name = function_call
                        .get("name")
                        .and_then(|value| value.as_str())
                        .unwrap_or("")
                        .trim();
                    if name.is_empty() {
                        continue;
                    }

                    let mut tool_call_id = function_call
                        .get("id")
                        .and_then(|value| value.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();

                    if tool_call_id.is_empty() || !seen_tool_call_ids.insert(tool_call_id.clone()) {
                        generated_tool_call_index = generated_tool_call_index.saturating_add(1);
                        tool_call_id = format!("tool_call_{generated_tool_call_index}");
                        seen_tool_call_ids.insert(tool_call_id.clone());
                    }

                    let arguments = function_call
                        .get("args")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({}));

                    tool_calls.push(serde_json::json!({
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        }
                    }));
                }
            }
        }

        if let Some(usage_metadata) = chunk["response"]["usageMetadata"].as_object() {
            let prompt_tokens = usage_metadata
                .get("promptTokenCount")
                .and_then(|value| value.as_u64())
                .unwrap_or(0);
            let cached_tokens = usage_metadata
                .get("cachedContentTokenCount")
                .and_then(|value| value.as_u64())
                .unwrap_or(0);
            let candidate_tokens = usage_metadata
                .get("candidatesTokenCount")
                .and_then(|value| value.as_u64())
                .unwrap_or(0);
            let thought_tokens = usage_metadata
                .get("thoughtsTokenCount")
                .and_then(|value| value.as_u64())
                .unwrap_or(0);

            input_tokens = prompt_tokens.saturating_sub(cached_tokens);
            cached_input_tokens = cached_tokens;
            output_tokens = candidate_tokens.saturating_add(thought_tokens);
            total_tokens = usage_metadata
                .get("totalTokenCount")
                .and_then(|value| value.as_u64())
                .unwrap_or_else(|| input_tokens.saturating_add(output_tokens));
        }
    }

    if !saw_sse_payload {
        return Err(CompletionError::ResponseError(
            "Google Antigravity returned no SSE payload".to_string(),
        ));
    }

    if text_output.trim().is_empty() && reasoning_output.trim().is_empty() && tool_calls.is_empty()
    {
        return Err(CompletionError::ResponseError(
            "Google Antigravity returned an empty response".to_string(),
        ));
    }

    let mut message = serde_json::json!({});
    if !text_output.trim().is_empty() {
        message["content"] = serde_json::json!(text_output);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = serde_json::json!(tool_calls);
    }
    if message["content"].is_null() && !reasoning_output.trim().is_empty() {
        message["reasoning_content"] = serde_json::json!(reasoning_output);
    }

    let finish_reason = match finish_reason.as_deref() {
        Some("MAX_TOKENS") => "length",
        Some("MALFORMED_FUNCTION_CALL") | Some("UNEXPECTED_TOOL_CALL") => "tool_calls",
        _ => "stop",
    };

    let completion_body = serde_json::json!({
        "id": format!("antigravity-{}", current_epoch_millis()),
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
            "prompt_tokens_details": {
                "cached_tokens": cached_input_tokens,
            }
        }
    });

    parse_openai_response(completion_body, "Google Antigravity")
}

fn parse_anthropic_response(
    body: serde_json::Value,
) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
    let content_blocks = body["content"]
        .as_array()
        .ok_or_else(|| CompletionError::ResponseError("missing content array".into()))?;

    let mut assistant_content = Vec::new();
    let mut saw_thinking_block = false;

    for block in content_blocks {
        match block["type"].as_str() {
            Some("text") => {
                let text = block["text"].as_str().unwrap_or("").to_string();
                assistant_content.push(AssistantContent::Text(Text { text }));
            }
            Some("tool_use") => {
                let id = block["id"].as_str().unwrap_or("").to_string();
                let name = block["name"].as_str().unwrap_or("").to_string();
                let arguments = block["input"].clone();
                assistant_content.push(AssistantContent::ToolCall(make_tool_call(
                    id, name, arguments,
                )));
            }
            Some("thinking") | Some("redacted_thinking") => {
                // These are provider-side reasoning blocks. They can appear without a
                // final text/tool block, which previously caused an empty-response error.
                saw_thinking_block = true;
            }
            _ => {}
        }
    }

    if assistant_content.is_empty() && saw_thinking_block {
        // Return an empty assistant text instead of raising an error. Channel/worker
        // logic already treats empty text as "no direct reply".
        assistant_content.push(AssistantContent::Text(Text {
            text: String::new(),
        }));
    }

    let choice = OneOrMany::many(assistant_content)
        .map_err(|_| CompletionError::ResponseError("empty response from Anthropic".into()))?;

    let input_tokens = body["usage"]["input_tokens"].as_u64().unwrap_or(0);
    let output_tokens = body["usage"]["output_tokens"].as_u64().unwrap_or(0);
    let cached = body["usage"]["cache_read_input_tokens"]
        .as_u64()
        .unwrap_or(0);

    Ok(completion::CompletionResponse {
        choice,
        usage: completion::Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cached_input_tokens: cached,
        },
        raw_response: RawResponse { body },
    })
}

fn parse_openai_response(
    body: serde_json::Value,
    provider_label: &str,
) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
    let choice = &body["choices"][0]["message"];

    let mut assistant_content = Vec::new();

    if let Some(text) = choice["content"].as_str() {
        if !text.is_empty() {
            assistant_content.push(AssistantContent::Text(Text {
                text: text.to_string(),
            }));
        }
    }

    // Some reasoning models (e.g., NVIDIA kimi-k2.5) return reasoning in a separate field
    if assistant_content.is_empty() {
        if let Some(reasoning) = choice["reasoning_content"].as_str() {
            if !reasoning.is_empty() {
                tracing::debug!(
                    provider = %provider_label,
                    "extracted reasoning_content as main content"
                );
                assistant_content.push(AssistantContent::Text(Text {
                    text: reasoning.to_string(),
                }));
            }
        }
    }

    if let Some(tool_calls) = choice["tool_calls"].as_array() {
        for tc in tool_calls {
            let id = tc["id"].as_str().unwrap_or("").to_string();
            let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
            // OpenAI-compatible APIs usually return arguments as a JSON string.
            // Some providers return it as a raw JSON object instead.
            let arguments_field = &tc["function"]["arguments"];
            let arguments = arguments_field
                .as_str()
                .and_then(|raw| serde_json::from_str(raw).ok())
                .or_else(|| arguments_field.as_object().map(|_| arguments_field.clone()))
                .unwrap_or(serde_json::json!({}));
            assistant_content.push(AssistantContent::ToolCall(make_tool_call(
                id, name, arguments,
            )));
        }
    }

    let result_choice = OneOrMany::many(assistant_content.clone()).map_err(|_| {
        tracing::warn!(
            provider = %provider_label,
            choice = ?choice,
            "empty response from provider"
        );
        CompletionError::ResponseError(format!("empty response from {provider_label}"))
    })?;

    let input_tokens = body["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    let output_tokens = body["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    let cached = body["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);

    Ok(completion::CompletionResponse {
        choice: result_choice,
        usage: completion::Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cached_input_tokens: cached,
        },
        raw_response: RawResponse { body },
    })
}

fn parse_openai_responses_response(
    body: serde_json::Value,
) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
    let output_items = body["output"]
        .as_array()
        .ok_or_else(|| CompletionError::ResponseError("missing output array".into()))?;

    let mut assistant_content = Vec::new();

    for output_item in output_items {
        match output_item["type"].as_str() {
            Some("message") => {
                if let Some(content_items) = output_item["content"].as_array() {
                    for content_item in content_items {
                        if content_item["type"].as_str() == Some("output_text") {
                            if let Some(text) = content_item["text"].as_str() {
                                if !text.is_empty() {
                                    assistant_content.push(AssistantContent::Text(Text {
                                        text: text.to_string(),
                                    }));
                                }
                            }
                        }
                    }
                }
            }
            Some("function_call") => {
                let call_id = output_item["call_id"]
                    .as_str()
                    .or_else(|| output_item["id"].as_str())
                    .unwrap_or("")
                    .to_string();
                let name = output_item["name"].as_str().unwrap_or("").to_string();
                let arguments = output_item["arguments"]
                    .as_str()
                    .and_then(|arguments| serde_json::from_str(arguments).ok())
                    .unwrap_or(serde_json::json!({}));

                assistant_content.push(AssistantContent::ToolCall(make_tool_call(
                    call_id, name, arguments,
                )));
            }
            _ => {}
        }
    }

    let choice = OneOrMany::many(assistant_content).map_err(|_| {
        CompletionError::ResponseError("empty response from OpenAI Responses API".into())
    })?;

    let input_tokens = body["usage"]["input_tokens"].as_u64().unwrap_or(0);
    let output_tokens = body["usage"]["output_tokens"].as_u64().unwrap_or(0);
    let cached = body["usage"]["input_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);

    Ok(completion::CompletionResponse {
        choice,
        usage: completion::Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cached_input_tokens: cached,
        },
        raw_response: RawResponse { body },
    })
}
