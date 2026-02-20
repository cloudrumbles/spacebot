//! Spawn worker tool for creating new workers.

use crate::WorkerId;
use crate::agent::channel::{
    ChannelState, spawn_opencode_worker_from_state, spawn_worker_from_state,
};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Tool for spawning workers.
#[derive(Debug, Clone)]
pub struct SpawnWorkerTool {
    state: ChannelState,
}

impl SpawnWorkerTool {
    /// Create a new spawn worker tool with access to channel state.
    pub fn new(state: ChannelState) -> Self {
        Self { state }
    }
}

/// Error type for spawn worker tool.
#[derive(Debug, thiserror::Error)]
#[error("Worker spawn failed: {0}")]
pub struct SpawnWorkerError(String);

/// Arguments for spawn worker tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SpawnWorkerArgs {
    /// The task description for the worker.
    pub task: String,
    /// Whether this is an interactive worker (accepts follow-up messages).
    #[serde(default)]
    pub interactive: bool,
    /// Optional skill name to load into the worker's context. The worker will
    /// receive the full skill instructions in its system prompt.
    #[serde(default)]
    pub skill: Option<String>,
    /// Task type for model routing. One of: "coding", "summarization", "deep_reasoning".
    /// Determines which model is used if task_overrides are configured. Omit to use
    /// the default worker model.
    #[serde(default)]
    pub task_type: Option<String>,
    /// Worker type: "builtin" (default) runs a Rig agent loop with shell/file/exec
    /// tools. "opencode" spawns an OpenCode subprocess with full coding agent
    /// capabilities. Use "opencode" for complex coding tasks that benefit from
    /// codebase exploration and context management.
    #[serde(default)]
    pub worker_type: Option<String>,
    /// Working directory for the worker. Required for "opencode" workers.
    /// The OpenCode agent will operate in this directory.
    #[serde(default)]
    pub directory: Option<String>,
}

/// Output from spawn worker tool.
#[derive(Debug, Serialize)]
pub struct SpawnWorkerOutput {
    /// The ID of the spawned worker.
    pub worker_id: WorkerId,
    /// Whether the worker was spawned successfully.
    pub spawned: bool,
    /// Whether this is an interactive worker.
    pub interactive: bool,
    /// Status message.
    pub message: String,
}

impl Tool for SpawnWorkerTool {
    const NAME: &'static str = "spawn_worker";

    type Error = SpawnWorkerError;
    type Args = SpawnWorkerArgs;
    type Output = SpawnWorkerOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let rc = &self.state.deps.runtime_config;
        let browser_enabled = rc.browser_config.load().enabled;
        let web_search_enabled = rc.web_search_provider().is_some();
        let opencode_enabled = rc.opencode.load().enabled;

        let mut tools_list = vec!["shell", "file", "exec"];
        if browser_enabled {
            tools_list.push("browser");
        }
        if web_search_enabled {
            tools_list.push("web_search");
        }

        let opencode_note = if opencode_enabled {
            " Set worker_type to \"opencode\" with a directory path for complex coding tasks — this spawns a full OpenCode coding agent with codebase exploration, context management, and its own tool suite."
        } else {
            ""
        };

        let base_description = crate::prompts::text::get("tools/spawn_worker");
        let description = base_description
            .replace("{tools}", &tools_list.join(", "))
            .replace("{opencode_note}", opencode_note);

        let mut properties = serde_json::json!({
            "task": {
                "type": "string",
                "description": "Clear, specific description of what the worker should do. Include all context needed since the worker can't see your conversation."
            },
            "interactive": {
                "type": "boolean",
                "default": false,
                "description": "If true, the worker stays alive and accepts follow-up messages via route_to_worker. If false (default), the worker runs once and returns."
            },
            "skill": {
                "type": "string",
                "description": "Name of a skill to load into the worker. The worker receives the full skill instructions in its system prompt. Only use skill names from <available_skills>."
            },
            "task_type": {
                "type": "string",
                "enum": ["coding", "summarization", "deep_reasoning"],
                "description": "Hint for model routing. \"coding\" for software development tasks, \"summarization\" for condensing/compacting content, \"deep_reasoning\" for complex analysis. Omit to use the default worker model."
            }
        });

        if opencode_enabled {
            properties.as_object_mut().unwrap().insert(
                "worker_type".to_string(),
                serde_json::json!({
                    "type": "string",
                    "enum": ["builtin", "opencode"],
                    "default": "builtin",
                    "description": "\"builtin\" (default) runs a Rig agent loop. \"opencode\" spawns a full OpenCode coding agent — use for complex multi-file coding tasks."
                }),
            );
            properties.as_object_mut().unwrap().insert(
                "directory".to_string(),
                serde_json::json!({
                    "type": "string",
                    "description": "Working directory for the worker. Required when worker_type is \"opencode\". The OpenCode agent operates in this directory."
                }),
            );
        }

        ToolDefinition {
            name: Self::NAME.to_string(),
            description,
            parameters: serde_json::json!({
                "type": "object",
                "properties": properties,
                "required": ["task"],
                "allOf": if opencode_enabled {
                    vec![
                        serde_json::json!({
                            "if": {
                                "properties": {
                                    "worker_type": { "const": "opencode" }
                                },
                                "required": ["worker_type"]
                            },
                            "then": {
                                "required": ["task", "directory"]
                            }
                        })
                    ]
                } else {
                    vec![]
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let task = args.task.trim();
        if task.is_empty() {
            return Err(SpawnWorkerError("'task' must not be empty".into()));
        }

        let worker_type = args.worker_type.as_deref().unwrap_or("builtin");
        let opencode_enabled = self.state.deps.runtime_config.opencode.load().enabled;

        let is_opencode = match worker_type {
            "builtin" => false,
            "opencode" => {
                if !opencode_enabled {
                    return Err(SpawnWorkerError(
                        "opencode workers are disabled in this agent config".into(),
                    ));
                }
                true
            }
            other => {
                return Err(SpawnWorkerError(format!(
                    "unknown worker_type '{other}'. Use 'builtin' or 'opencode'."
                )));
            }
        };

        let worker_id = if is_opencode {
            if args.skill.is_some() {
                return Err(SpawnWorkerError(
                    "skill is only supported for builtin workers; include extra instructions in task for opencode workers".into(),
                ));
            }
            let directory = args.directory.as_deref().ok_or_else(|| {
                SpawnWorkerError("directory is required for opencode workers".into())
            })?;

            spawn_opencode_worker_from_state(
                &self.state,
                task,
                directory,
                args.interactive,
                args.task_type.as_deref(),
            )
            .await
            .map_err(|e| SpawnWorkerError(format!("{e}")))?
        } else {
            spawn_worker_from_state(
                &self.state,
                task,
                args.interactive,
                args.skill.as_deref(),
                args.task_type.as_deref(),
            )
            .await
            .map_err(|e| SpawnWorkerError(format!("{e}")))?
        };

        let worker_type_label = if is_opencode { "OpenCode" } else { "builtin" };
        let message = if args.interactive {
            format!(
                "Interactive {worker_type_label} worker {worker_id} spawned for: {}. Route follow-ups with route.",
                task
            )
        } else {
            format!(
                "{worker_type_label} worker {worker_id} spawned for: {}. It will report back when done.",
                task
            )
        };

        Ok(SpawnWorkerOutput {
            worker_id,
            spawned: true,
            interactive: args.interactive,
            message,
        })
    }
}
