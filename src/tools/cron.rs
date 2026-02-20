//! Cron job management tool for creating, listing, and deleting scheduled tasks.

use crate::cron::scheduler::{CronConfig, Scheduler, is_one_shot_schedule, validate_schedule};
use crate::cron::store::CronStore;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::sync::Arc;

/// Tool for managing cron jobs (scheduled recurring tasks).
#[derive(Debug, Clone)]
pub struct CronTool {
    store: Arc<CronStore>,
    scheduler: Arc<Scheduler>,
}

impl CronTool {
    pub fn new(store: Arc<CronStore>, scheduler: Arc<Scheduler>) -> Self {
        Self { store, scheduler }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Cron operation failed: {0}")]
pub struct CronError(String);

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CronArgs {
    /// The operation to perform: "create", "list", or "delete".
    pub action: String,
    /// Required for "create": a short unique ID for the cron job (e.g. "check-email", "daily-summary").
    #[serde(default)]
    pub id: Option<String>,
    /// Required for "create": the prompt/instruction to execute on each run.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Required for "create": a cron expression defining the schedule (e.g. "0 9 * * *" for daily at 9am).
    #[serde(default)]
    pub schedule: Option<String>,
    /// Required for "create": where to deliver results, in "adapter:target" format (e.g. "telegram:525365593").
    #[serde(default)]
    pub delivery_target: Option<String>,
    /// Required for "delete": the ID of the cron job to remove.
    #[serde(default)]
    pub delete_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CronOutput {
    pub success: bool,
    pub message: String,
    /// Populated on "list" action.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jobs: Option<Vec<CronEntry>>,
}

#[derive(Debug, Serialize)]
pub struct CronEntry {
    pub id: String,
    pub prompt: String,
    pub schedule: String,
    pub delivery_target: String,
}

impl Tool for CronTool {
    const NAME: &'static str = "cron";

    type Error = CronError;
    type Args = CronArgs;
    type Output = CronOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: crate::prompts::text::get("tools/cron").to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "delete"],
                        "description": "The operation: create a new cron job, list all cron jobs, or delete one."
                    },
                    "id": {
                        "type": "string",
                        "description": "For 'create': a short unique ID (e.g. 'check-email', 'daily-summary')."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "For 'create': the instruction to execute on each run."
                    },
                    "schedule": {
                        "type": "string",
                        "description": "For 'create': either a cron expression (e.g. '0 9 * * *' for daily at 9am, '*/5 * * * *' for every 5 minutes) or a one-shot timestamp ('@once:2026-02-20T14:30:00Z')."
                    },
                    "delivery_target": {
                        "type": "string",
                        "description": "For 'create': where to send results, format 'adapter:target' (e.g. 'telegram:525365593')."
                    },
                    "delete_id": {
                        "type": "string",
                        "description": "For 'delete': the ID of the cron job to remove."
                    }
                },
                "required": ["action"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        match args.action.as_str() {
            "create" => self.create(args).await,
            "list" => self.list().await,
            "delete" => self.delete(args).await,
            other => Ok(CronOutput {
                success: false,
                message: format!("Unknown action '{other}'. Use 'create', 'list', or 'delete'."),
                jobs: None,
            }),
        }
    }
}

impl CronTool {
    async fn create(&self, args: CronArgs) -> Result<CronOutput, CronError> {
        let id = args
            .id
            .ok_or_else(|| CronError("'id' is required for create".into()))?;
        let prompt = args
            .prompt
            .ok_or_else(|| CronError("'prompt' is required for create".into()))?;
        let schedule = args
            .schedule
            .ok_or_else(|| CronError("'schedule' is required for create".into()))?;
        let delivery_target = args
            .delivery_target
            .ok_or_else(|| CronError("'delivery_target' is required for create".into()))?;

        validate_schedule(&schedule)
            .map_err(|error| CronError(format!("invalid schedule '{schedule}': {error}")))?;

        let config = CronConfig {
            id: id.clone(),
            prompt: prompt.clone(),
            schedule: schedule.clone(),
            delivery_target: delivery_target.clone(),
            enabled: true,
        };

        // Persist to database
        self.store
            .save(&config)
            .await
            .map_err(|error| CronError(format!("failed to save: {error}")))?;

        // Register with the running scheduler so it starts immediately
        self.scheduler
            .register(config)
            .await
            .map_err(|error| CronError(format!("failed to register: {error}")))?;

        // Generate a human-readable description of the schedule.
        let message = if is_one_shot_schedule(&schedule) {
            format!("Cron job '{id}' created. It will run once at {schedule}.")
        } else {
            let cron = croner::Cron::from_str(&schedule).map_err(|error| {
                CronError(format!("invalid cron expression '{schedule}': {error}"))
            })?;
            let description = cron.describe();
            format!("Cron job '{id}' created. Schedule: {schedule} ({description}).")
        };

        tracing::info!(cron_id = %id, %schedule, %delivery_target, "cron job created via tool");

        Ok(CronOutput {
            success: true,
            message,
            jobs: None,
        })
    }

    async fn list(&self) -> Result<CronOutput, CronError> {
        let configs = self
            .store
            .load_all()
            .await
            .map_err(|error| CronError(format!("failed to list: {error}")))?;

        let entries: Vec<CronEntry> = configs
            .into_iter()
            .map(|config| CronEntry {
                id: config.id,
                prompt: config.prompt,
                schedule: config.schedule,
                delivery_target: config.delivery_target,
            })
            .collect();

        let count = entries.len();
        Ok(CronOutput {
            success: true,
            message: format!("{count} active cron job(s)."),
            jobs: Some(entries),
        })
    }

    async fn delete(&self, args: CronArgs) -> Result<CronOutput, CronError> {
        let id = args
            .delete_id
            .or(args.id)
            .ok_or_else(|| CronError("'delete_id' or 'id' is required for delete".into()))?;

        // Unregister from the running scheduler first
        self.scheduler.unregister(&id).await;

        self.store
            .delete(&id)
            .await
            .map_err(|error| CronError(format!("failed to delete: {error}")))?;

        tracing::info!(cron_id = %id, "cron job deleted via tool");

        Ok(CronOutput {
            success: true,
            message: format!("Cron job '{id}' deleted."),
            jobs: None,
        })
    }
}
