//! Google Antigravity provider implementation.

use crate::config::ProviderConfig;
use crate::llm::manager::LlmManager;
use crate::llm::model::{RawResponse, parse_openai_response};

use rig::completion::{self, CompletionError, CompletionRequest};
use rig::message::{AssistantContent, DocumentSourceKind, Image, Message, MimeType, UserContent};
use rig::one_or_many::OneOrMany;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use uuid::Uuid;

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

pub(crate) async fn call_completion(
    llm_manager: &LlmManager,
    model_name: &str,
    request: CompletionRequest,
    provider_config: &ProviderConfig,
) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
    let mut credentials =
        parse_google_antigravity_api_key_payload(provider_config.api_key.as_str())?;

    if credentials.project_id.trim().is_empty() {
        credentials.project_id = GOOGLE_ANTIGRAVITY_DEFAULT_PROJECT_ID.to_string();
    }

    let endpoints = google_antigravity_endpoints(provider_config.base_url.as_str());
    let request_body =
        build_google_antigravity_request_body(model_name, &request, &credentials.project_id);
    let include_claude_thinking_header = google_antigravity_is_claude_thinking_model(model_name);

    let mut access_token =
        google_antigravity_access_token(llm_manager, &credentials, false).await?;
    let mut forced_refresh_attempted = false;
    let mut last_error: Option<String> = None;

    for endpoint in endpoints {
        let request_url = format!(
            "{}{GOOGLE_ANTIGRAVITY_STREAM_PATH}",
            endpoint.trim_end_matches('/')
        );

        let mut response = send_google_antigravity_request(
            llm_manager,
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
            access_token = google_antigravity_access_token(llm_manager, &credentials, true).await?;
            forced_refresh_attempted = true;

            response = send_google_antigravity_request(
                llm_manager,
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
    llm_manager: &LlmManager,
    request_url: &str,
    access_token: &str,
    request_body: &serde_json::Value,
    include_claude_thinking_header: bool,
) -> Result<reqwest::Response, CompletionError> {
    let antigravity_version = resolve_google_antigravity_version(llm_manager.http_client()).await;
    let mut request_builder = llm_manager
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
    llm_manager: &LlmManager,
    credentials: &GoogleAntigravityApiKeyPayload,
    force_refresh: bool,
) -> Result<String, CompletionError> {
    let now_ms = current_epoch_millis();
    let refresh_key = credentials.refresh_token.trim();
    let expires_at_ms = normalize_google_antigravity_expires_at_ms(credentials.expires_at);

    if !refresh_key.is_empty() {
        if GOOGLE_ANTIGRAVITY_CLIENT_ID.is_empty() || GOOGLE_ANTIGRAVITY_CLIENT_SECRET.is_empty() {
            return Err(CompletionError::ProviderError(
                "Google Antigravity OAuth client credentials are not configured. \
Set GOOGLE_ANTIGRAVITY_CLIENT_ID and GOOGLE_ANTIGRAVITY_CLIENT_SECRET."
                    .to_string(),
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

        let token_response = llm_manager
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
            "Google Antigravity credentials are missing both token and refresh token".to_string(),
        ));
    }

    Ok(inline_token.to_string())
}

fn current_epoch_millis() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_millis() as i64,
        Err(_) => 0,
    }
}

fn truncate_body(body: &str) -> &str {
    let limit = 500;
    if body.len() <= limit {
        body
    } else {
        &body[..limit]
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
        .filter_map(|item| match item {
            rig::message::ToolResultContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn convert_messages_to_google_antigravity(
    messages: &OneOrMany<Message>,
    include_tool_call_id: bool,
) -> Vec<serde_json::Value> {
    let mut result = Vec::new();
    let mut tool_call_names: HashMap<String, String> = HashMap::new();
    let mut tool_call_wire_ids: HashMap<String, String> = HashMap::new();

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
                            let lookup_id =
                                tool_result.call_id.as_deref().unwrap_or(&tool_result.id);
                            let tool_name = tool_call_names
                                .get(lookup_id)
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
                                let wire_id = tool_call_wire_ids
                                    .get(lookup_id)
                                    .cloned()
                                    .or_else(|| tool_result.call_id.clone())
                                    .unwrap_or_else(|| tool_result.id.clone());
                                tool_result_part["functionResponse"]["id"] =
                                    serde_json::json!(wire_id);
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
                            let tool_name = tool_call.function.name.clone();
                            let wire_id = tool_call
                                .call_id
                                .clone()
                                .unwrap_or_else(|| tool_call.id.clone());

                            tool_call_names.insert(tool_call.id.clone(), tool_name.clone());
                            tool_call_wire_ids.insert(tool_call.id.clone(), wire_id.clone());

                            if let Some(call_id) = &tool_call.call_id {
                                tool_call_names.insert(call_id.clone(), tool_name.clone());
                                tool_call_wire_ids.insert(call_id.clone(), wire_id.clone());
                            }

                            let mut function_call = serde_json::json!({
                                "name": tool_call.function.name,
                                "args": tool_call.function.arguments,
                            });

                            if include_tool_call_id {
                                function_call["id"] = serde_json::json!(wire_id);
                            }

                            let thought_signature = tool_call
                                .signature
                                .clone()
                                .filter(|signature| !signature.trim().is_empty())
                                .or_else(|| {
                                    tool_call
                                        .additional_params
                                        .as_ref()
                                        .and_then(|params| {
                                            params
                                                .get("thoughtSignature")
                                                .or_else(|| params.get("thought_signature"))
                                        })
                                        .and_then(|value| value.as_str())
                                        .map(|value| value.to_string())
                                });

                            if let Some(signature) = thought_signature {
                                function_call["thoughtSignature"] = serde_json::json!(signature);
                            }

                            if let Some(additional_params) = &tool_call.additional_params {
                                if let Some(additional_object) = additional_params.as_object() {
                                    if let Some(function_call_object) =
                                        function_call.as_object_mut()
                                    {
                                        for (key, value) in additional_object {
                                            if !function_call_object.contains_key(key) {
                                                function_call_object
                                                    .insert(key.clone(), value.clone());
                                            }
                                        }
                                    }
                                }
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

                    let thought_signature = function_call
                        .get("thoughtSignature")
                        .or_else(|| function_call.get("thought_signature"))
                        .and_then(|value| value.as_str())
                        .map(|value| value.trim())
                        .filter(|value| !value.is_empty())
                        .map(|value| value.to_string());

                    let call_id = function_call
                        .get("callId")
                        .or_else(|| function_call.get("call_id"))
                        .and_then(|value| value.as_str())
                        .map(|value| value.trim())
                        .filter(|value| !value.is_empty())
                        .map(|value| value.to_string());

                    let mut additional_params = serde_json::Map::new();
                    for (key, value) in function_call {
                        let is_reserved = matches!(
                            key.as_str(),
                            "name"
                                | "args"
                                | "id"
                                | "callId"
                                | "call_id"
                                | "thoughtSignature"
                                | "thought_signature"
                        );
                        if !is_reserved {
                            additional_params.insert(key.clone(), value.clone());
                        }
                    }

                    let mut tool_call_entry = serde_json::json!({
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        }
                    });

                    if let Some(signature) = thought_signature {
                        tool_call_entry["signature"] = serde_json::json!(signature);
                    }
                    if let Some(call_id) = call_id {
                        tool_call_entry["call_id"] = serde_json::json!(call_id);
                    }
                    if !additional_params.is_empty() {
                        tool_call_entry["additional_params"] =
                            serde_json::Value::Object(additional_params);
                    }

                    tool_calls.push(tool_call_entry);
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

#[cfg(test)]
mod tests {
    use super::*;
    use rig::message::{AssistantContent, Message, Text, ToolCall, ToolFunction};

    #[test]
    fn parse_google_antigravity_sse_response_preserves_tool_call_signature() {
        let sse_payload = r#"data: {"response":{"candidates":[{"finishReason":"STOP","content":{"parts":[{"functionCall":{"id":"call_123","name":"default_api:reply","args":{"message":"hi"},"thoughtSignature":"sig_abc_123"}}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":3,"totalTokenCount":13}}}"#;

        let response = parse_google_antigravity_sse_response(sse_payload)
            .expect("expected parser to handle tool call with thought signature");

        let tool_call = response
            .choice
            .iter()
            .find_map(|item| match item {
                AssistantContent::ToolCall(tool_call) => Some(tool_call),
                _ => None,
            })
            .expect("expected parsed response to contain a tool call");

        assert_eq!(tool_call.id, "call_123");
        assert_eq!(tool_call.function.name, "default_api:reply");
        assert_eq!(tool_call.signature.as_deref(), Some("sig_abc_123"));
    }

    #[test]
    fn convert_messages_to_google_antigravity_includes_thought_signature() {
        let assistant_message = Message::Assistant {
            id: Some("assistant-1".to_string()),
            content: OneOrMany::one(AssistantContent::ToolCall(ToolCall {
                id: "tool_call_1".to_string(),
                call_id: None,
                function: ToolFunction {
                    name: "default_api:reply".to_string(),
                    arguments: serde_json::json!({"message": "hello"}),
                },
                signature: Some("sig_xyz".to_string()),
                additional_params: None,
            })),
        };

        let messages = OneOrMany::one(assistant_message);
        let converted = convert_messages_to_google_antigravity(&messages, true);
        let function_call = &converted[0]["parts"][0]["functionCall"];

        assert_eq!(function_call["thoughtSignature"].as_str(), Some("sig_xyz"));
        assert_eq!(function_call["id"].as_str(), Some("tool_call_1"));
    }

    #[test]
    fn convert_messages_to_google_antigravity_uses_tool_result_call_id_for_response() {
        let assistant_message = Message::Assistant {
            id: Some("assistant-1".to_string()),
            content: OneOrMany::one(AssistantContent::ToolCall(ToolCall {
                id: "tool_call_internal".to_string(),
                call_id: Some("provider_call_id".to_string()),
                function: ToolFunction {
                    name: "default_api:reply".to_string(),
                    arguments: serde_json::json!({"message": "hello"}),
                },
                signature: Some("sig_xyz".to_string()),
                additional_params: None,
            })),
        };

        let tool_result = rig::message::ToolResult {
            id: "tool_call_internal".to_string(),
            call_id: Some("provider_call_id".to_string()),
            content: OneOrMany::one(rig::message::ToolResultContent::Text(Text {
                text: "ok".to_string(),
            })),
        };

        let user_message = Message::User {
            content: OneOrMany::one(rig::message::UserContent::ToolResult(tool_result)),
        };

        let messages = OneOrMany::many(vec![assistant_message, user_message])
            .expect("assistant + user messages should be valid");
        let converted = convert_messages_to_google_antigravity(&messages, true);

        let function_response_id = converted[1]["parts"][0]["functionResponse"]["id"]
            .as_str()
            .expect("function response id should be present");

        assert_eq!(function_response_id, "provider_call_id");
    }
}
