//! Dynamic script-based tools that can be created, discovered, and executed at runtime.
//!
//! Tools are stored as directories under the agent workspace at `workspace/tools/`,
//! each containing a `tool.json` manifest and an executable script
//! (e.g. `run.py`, `run.sh`). The bot can create new tools by writing these files
//! to disk, then search for and execute them without recompiling.
//!
//! Two meta-tools provide access:
//! - `tool_search` — discovers available tools by name/description
//! - `tool_execute` — runs a discovered tool with JSON args

use crate::tools::{MAX_TOOL_OUTPUT_BYTES, truncate_output};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

// ---------------------------------------------------------------------------
// Manifest (tool.json)
// ---------------------------------------------------------------------------

/// On-disk tool manifest deserialized from `tool.json`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolManifest {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub parameters: serde_json::Value,
    #[serde(default)]
    pub examples: Vec<serde_json::Value>,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

fn default_timeout() -> u64 {
    60
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Scan a directory for tool subdirectories containing a valid `tool.json`.
/// Returns a map of tool name → (manifest, tool_dir_path).
fn scan_tools_dir(dir: &Path) -> HashMap<String, (ToolManifest, PathBuf)> {
    let mut tools = HashMap::new();

    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return tools, // dir doesn't exist yet — that's fine
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let manifest_path = path.join("tool.json");
        if !manifest_path.exists() {
            continue;
        }

        let content = match std::fs::read_to_string(&manifest_path) {
            Ok(c) => c,
            Err(error) => {
                tracing::warn!(path = %manifest_path.display(), %error, "failed to read tool.json");
                continue;
            }
        };

        let manifest: ToolManifest = match serde_json::from_str(&content) {
            Ok(m) => m,
            Err(error) => {
                tracing::warn!(path = %manifest_path.display(), %error, "invalid tool.json");
                continue;
            }
        };

        tools.insert(manifest.name.clone(), (manifest, path));
    }

    tools
}

/// Scan the agent workspace tools directory (`workspace/tools`) for dynamic tools.
fn scan_all_tools(workspace: &Path) -> HashMap<String, (ToolManifest, PathBuf)> {
    scan_tools_dir(&workspace.join("tools"))
}

/// Find the tool directory for a given tool name in `workspace/tools`.
fn find_tool_dir(name: &str, workspace: &Path) -> Option<(ToolManifest, PathBuf)> {
    scan_all_tools(workspace).remove(name)
}

/// Find the runner script in a tool directory.
/// Returns (runner_command, script_path). For `.py` → `python3`, `.sh` → `sh`.
fn find_runner(tool_dir: &Path) -> Option<(Option<String>, PathBuf)> {
    // Priority order: run.py, run.sh, run (executable)
    let candidates = [
        ("run.py", Some("python3")),
        ("run.sh", Some("sh")),
        ("run", None),
    ];

    for (filename, runner) in candidates {
        let path = tool_dir.join(filename);
        if path.exists() {
            return Some((runner.map(String::from), path));
        }
    }

    // Fallback: any file starting with "run."
    if let Ok(entries) = std::fs::read_dir(tool_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("run.") && name_str != "run.json" {
                let path = entry.path();
                let runner = match path.extension().and_then(|e| e.to_str()) {
                    Some("py") => Some("python3".to_string()),
                    Some("sh") => Some("sh".to_string()),
                    Some("js") => Some("node".to_string()),
                    _ => None,
                };
                return Some((runner, path));
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// ToolSearchTool
// ---------------------------------------------------------------------------

/// Tool for discovering available script tools.
#[derive(Debug, Clone)]
pub struct ToolSearchTool {
    workspace: PathBuf,
}

impl ToolSearchTool {
    pub fn new(workspace: PathBuf) -> Self {
        Self { workspace }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Tool search failed: {0}")]
pub struct ToolSearchError(String);

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ToolSearchArgs {
    /// Search query to match against tool names and descriptions.
    /// Use an empty string to list all available tools.
    pub query: String,
}

#[derive(Debug, Serialize)]
pub struct ToolSearchOutput {
    pub tools: Vec<ToolInfo>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    pub parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<serde_json::Value>,
}

impl Tool for ToolSearchTool {
    const NAME: &'static str = "tool_search";

    type Error = ToolSearchError;
    type Args = ToolSearchArgs;
    type Output = ToolSearchOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: crate::prompts::text::get("tools/tool_search").to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to match against tool names and descriptions. Empty string lists all tools."
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let all_tools = scan_all_tools(&self.workspace);
        let query = args.query.to_lowercase();

        let mut results: Vec<ToolInfo> = all_tools
            .into_values()
            .filter(|(manifest, _)| {
                if query.is_empty() {
                    return true;
                }
                manifest.name.to_lowercase().contains(&query)
                    || manifest.description.to_lowercase().contains(&query)
            })
            .map(|(manifest, _)| ToolInfo {
                name: manifest.name,
                description: manifest.description,
                parameters: manifest.parameters,
                examples: manifest.examples,
            })
            .collect();

        results.sort_by(|a, b| a.name.cmp(&b.name));
        let total = results.len();

        tracing::debug!(query = %args.query, results = total, "tool search completed");

        Ok(ToolSearchOutput {
            tools: results,
            total,
        })
    }
}

// ---------------------------------------------------------------------------
// ToolExecuteTool
// ---------------------------------------------------------------------------

/// Tool for executing a discovered script tool.
#[derive(Debug, Clone)]
pub struct ToolExecuteTool {
    workspace: PathBuf,
}

impl ToolExecuteTool {
    pub fn new(workspace: PathBuf) -> Self {
        Self { workspace }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolExecuteError {
    #[error("Tool not found: '{0}'")]
    NotFound(String),
    #[error("No runner script found for tool '{0}' (expected run.py, run.sh, or run)")]
    NoRunner(String),
    #[error("Execution failed: {0}")]
    ExecFailed(String),
    #[error("Tool timed out after {0} seconds")]
    Timeout(u64),
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ToolExecuteArgs {
    /// The name of the tool to execute (as returned by tool_search).
    pub name: String,
    /// Arguments to pass to the tool as a JSON object.
    #[serde(default = "default_args")]
    pub args: serde_json::Value,
}

fn default_args() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}

#[derive(Debug, Serialize)]
pub struct ToolExecuteOutput {
    /// Whether the script exited successfully (exit code 0).
    pub success: bool,
    /// Standard output from the script.
    pub output: String,
    /// Standard error from the script (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr: Option<String>,
    /// The exit code of the script.
    pub exit_code: i32,
}

impl Tool for ToolExecuteTool {
    const NAME: &'static str = "tool_execute";

    type Error = ToolExecuteError;
    type Args = ToolExecuteArgs;
    type Output = ToolExecuteOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: crate::prompts::text::get("tools/tool_execute").to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the tool to execute (as returned by tool_search)"
                    },
                    "args": {
                        "type": "object",
                        "description": "Arguments to pass to the tool as a JSON object"
                    }
                },
                "required": ["name"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Find the tool directory
        let (manifest, tool_dir) = find_tool_dir(&args.name, &self.workspace)
            .ok_or_else(|| ToolExecuteError::NotFound(args.name.clone()))?;

        // Find the runner script
        let (runner, script_path) =
            find_runner(&tool_dir).ok_or_else(|| ToolExecuteError::NoRunner(args.name.clone()))?;

        // Build the command
        let mut cmd = match runner {
            Some(ref runner_bin) => {
                let mut c = Command::new(runner_bin);
                c.arg(&script_path);
                c
            }
            None => Command::new(&script_path),
        };

        cmd.current_dir(&self.workspace)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Serialize args to JSON for stdin
        let args_json = serde_json::to_string(&args.args)
            .map_err(|e| ToolExecuteError::ExecFailed(format!("failed to serialize args: {e}")))?;

        tracing::info!(
            tool = %args.name,
            runner = ?runner,
            script = %script_path.display(),
            "executing dynamic tool"
        );

        // Spawn the process
        let mut child = cmd
            .spawn()
            .map_err(|e| ToolExecuteError::ExecFailed(format!("failed to spawn: {e}")))?;

        // Write args to stdin
        if let Some(mut stdin) = child.stdin.take() {
            if let Err(e) = stdin.write_all(args_json.as_bytes()).await {
                tracing::warn!(%e, "failed to write args to tool stdin");
            }
            // stdin is dropped here, signaling EOF
        }

        // Take stdout/stderr handles before waiting so we retain child ownership for kill
        let mut child_stdout = child.stdout.take();
        let mut child_stderr = child.stderr.take();

        let timeout_dur = tokio::time::Duration::from_secs(manifest.timeout);
        let status = match tokio::time::timeout(timeout_dur, child.wait()).await {
            Ok(Ok(status)) => status,
            Ok(Err(e)) => return Err(ToolExecuteError::ExecFailed(e.to_string())),
            Err(_) => {
                let _ = child.kill().await;
                return Err(ToolExecuteError::Timeout(manifest.timeout));
            }
        };

        // Read captured output
        let mut stdout_bytes = Vec::new();
        let mut stderr_bytes = Vec::new();
        if let Some(ref mut out) = child_stdout {
            use tokio::io::AsyncReadExt;
            let _ = out.read_to_end(&mut stdout_bytes).await;
        }
        if let Some(ref mut err) = child_stderr {
            use tokio::io::AsyncReadExt;
            let _ = err.read_to_end(&mut stderr_bytes).await;
        }

        let stdout = truncate_output(
            &String::from_utf8_lossy(&stdout_bytes),
            MAX_TOOL_OUTPUT_BYTES,
        );
        let stderr_raw = String::from_utf8_lossy(&stderr_bytes).to_string();
        let stderr = if stderr_raw.is_empty() {
            None
        } else {
            Some(truncate_output(&stderr_raw, MAX_TOOL_OUTPUT_BYTES))
        };
        let exit_code = status.code().unwrap_or(-1);
        let success = status.success();

        tracing::info!(
            tool = %args.name,
            %success,
            %exit_code,
            "dynamic tool execution completed"
        );

        Ok(ToolExecuteOutput {
            success,
            output: stdout,
            stderr,
            exit_code,
        })
    }
}
