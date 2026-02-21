//! Cron scheduler: cron-expression-based job scheduling and execution.
//!
//! Each cron job gets its own tokio task that sleeps until the next
//! scheduled time, fires, then computes the next occurrence. When a job
//! fires, it creates a fresh short-lived channel, runs the job's prompt
//! through the LLM, and delivers the result to the delivery target via
//! the messaging system.

use crate::agent::channel::Channel;
use crate::cron::store::CronStore;
use crate::error::Result;
use crate::messaging::MessagingManager;
use crate::{AgentDeps, InboundMessage, MessageContent, OutboundResponse};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::Duration;

/// A cron job definition loaded from the database.
#[derive(Debug, Clone)]
pub struct CronJob {
    pub id: String,
    pub prompt: String,
    pub schedule: String,
    pub delivery_target: DeliveryTarget,
    pub enabled: bool,
    pub run_once: bool,
    pub consecutive_failures: u32,
    /// Maximum wall-clock seconds to wait for the job to complete.
    /// `None` uses the default of 120 seconds.
    pub timeout_secs: Option<u64>,
}

/// Where to send cron job results.
#[derive(Debug, Clone)]
pub struct DeliveryTarget {
    /// Messaging adapter name (e.g. "telegram").
    pub adapter: String,
    /// Platform-specific target (e.g. a chat ID).
    pub target: String,
}

impl DeliveryTarget {
    /// Parse a delivery target string in the format "adapter:target".
    pub fn parse(raw: &str) -> Option<Self> {
        let (adapter, target) = raw.split_once(':')?;
        if adapter.is_empty() || target.is_empty() {
            return None;
        }
        Some(Self {
            adapter: adapter.to_string(),
            target: target.to_string(),
        })
    }
}

impl std::fmt::Display for DeliveryTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.adapter, self.target)
    }
}

/// Serializable cron job config (for storage and TOML parsing).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CronConfig {
    pub id: String,
    pub prompt: String,
    #[serde(default = "default_schedule")]
    pub schedule: String,
    /// Delivery target in "adapter:target" format (e.g. "telegram:525365593").
    pub delivery_target: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub run_once: bool,
    /// Maximum wall-clock seconds to wait for the job to complete.
    /// `None` uses the default of 120 seconds.
    pub timeout_secs: Option<u64>,
}

fn default_schedule() -> String {
    "0 * * * *".to_string()
}

fn default_true() -> bool {
    true
}

const ONE_SHOT_PREFIX: &str = "@once:";

/// Validate a schedule string.
///
/// Supported formats:
/// - Cron expression (e.g. `0 9 * * *`)
/// - One-shot absolute timestamp (`@once:2026-02-20T14:30:00Z`)
pub fn validate_schedule(schedule: &str) -> std::result::Result<(), String> {
    if parse_one_shot_timestamp(schedule)?.is_some() {
        return Ok(());
    }

    croner::Cron::from_str(schedule)
        .map(|_| ())
        .map_err(|error| format!("invalid cron expression '{schedule}': {error}"))
}

/// Whether the schedule string uses one-shot semantics.
pub fn is_one_shot_schedule(schedule: &str) -> bool {
    schedule.trim().starts_with(ONE_SHOT_PREFIX)
}

/// Context needed to execute a cron job (agent resources + messaging).
///
/// Prompts, identity, browser config, and skills are read from
/// `deps.runtime_config` on each job firing so changes propagate
/// without restarting the scheduler.
#[derive(Clone)]
pub struct CronContext {
    pub deps: AgentDeps,
    pub screenshot_dir: std::path::PathBuf,
    pub logs_dir: std::path::PathBuf,
    pub messaging_manager: Arc<MessagingManager>,
    pub store: Arc<CronStore>,
}

const MAX_CONSECUTIVE_FAILURES: u32 = 3;

/// Scheduler that manages cron job timers and execution.
pub struct Scheduler {
    jobs: Arc<RwLock<HashMap<String, CronJob>>>,
    timers: Arc<RwLock<HashMap<String, tokio::task::JoinHandle<()>>>>,
    context: CronContext,
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler").finish_non_exhaustive()
    }
}

impl Scheduler {
    pub fn new(context: CronContext) -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            timers: Arc::new(RwLock::new(HashMap::new())),
            context,
        }
    }

    /// Register and start a cron job from config.
    pub async fn register(&self, config: CronConfig) -> Result<()> {
        let delivery_target = DeliveryTarget::parse(&config.delivery_target).unwrap_or_else(|| {
            tracing::warn!(
                cron_id = %config.id,
                raw_target = %config.delivery_target,
                "invalid delivery target format, expected 'adapter:target'"
            );
            DeliveryTarget {
                adapter: "unknown".into(),
                target: config.delivery_target.clone(),
            }
        });

        let job = CronJob {
            id: config.id.clone(),
            prompt: config.prompt,
            schedule: config.schedule.clone(),
            delivery_target,
            enabled: config.enabled,
            run_once: config.run_once,
            consecutive_failures: 0,
            timeout_secs: config.timeout_secs,
        };

        {
            let mut jobs = self.jobs.write().await;
            jobs.insert(config.id.clone(), job);
        }

        if config.enabled {
            self.start_timer(&config.id).await;
        }

        tracing::info!(cron_id = %config.id, schedule = %config.schedule, "cron job registered");
        Ok(())
    }

    /// Start a timer loop for a cron job.
    async fn start_timer(&self, job_id: &str) {
        let job_id_for_map = job_id.to_string();
        let job_id = job_id.to_string();
        let jobs = self.jobs.clone();
        let context = self.context.clone();

        let handle = tokio::spawn(async move {
            loop {
                // Read the schedule expression from the current job state.
                let (schedule_expr, job_enabled) = {
                    let j = jobs.read().await;
                    match j.get(&job_id) {
                        Some(j) if !j.enabled => {
                            tracing::debug!(cron_id = %job_id, "cron job disabled, stopping timer");
                            break;
                        }
                        Some(j) => (j.schedule.clone(), j.enabled),
                        None => {
                            tracing::debug!(cron_id = %job_id, "cron job removed, stopping timer");
                            break;
                        }
                    }
                };

                if !job_enabled {
                    break;
                }

                let now = chrono::Local::now();

                // Parse schedule and compute next occurrence.
                let (next, is_one_shot) = match next_occurrence(&schedule_expr, now) {
                    Ok(value) => value,
                    Err(error) => {
                        tracing::error!(
                            cron_id = %job_id,
                            schedule = %schedule_expr,
                            error = %error,
                            "invalid schedule, stopping timer"
                        );
                        break;
                    }
                };

                let duration_until = (next - now).to_std().unwrap_or(Duration::from_secs(0));
                tracing::debug!(
                    cron_id = %job_id,
                    next_fire = %next.format("%Y-%m-%d %H:%M:%S %Z"),
                    secs_until = duration_until.as_secs(),
                    one_shot = is_one_shot,
                    "sleeping until next schedule occurrence"
                );

                tokio::time::sleep(duration_until).await;

                // Re-read job state after sleeping (it may have been disabled or removed).
                let job = {
                    let j = jobs.read().await;
                    match j.get(&job_id) {
                        Some(j) if !j.enabled => {
                            tracing::debug!(cron_id = %job_id, "cron job disabled during sleep, stopping timer");
                            break;
                        }
                        Some(j) => j.clone(),
                        None => {
                            tracing::debug!(cron_id = %job_id, "cron job removed during sleep, stopping timer");
                            break;
                        }
                    }
                };

                tracing::info!(cron_id = %job_id, "cron job firing");

                let run_result = run_cron_job(&job, &context).await;

                if is_one_shot || job.run_once {
                    tracing::info!(cron_id = %job_id, "one-shot/run-once cron job completed; disabling");

                    {
                        let mut j = jobs.write().await;
                        if let Some(job) = j.get_mut(&job_id) {
                            job.enabled = false;
                        }
                    }

                    if let Err(error) = context.store.update_enabled(&job_id, false).await {
                        tracing::error!(%error, cron_id = %job_id, "failed to persist one-shot disabled state");
                    }

                    break;
                }

                match run_result {
                    Ok(()) => {
                        // Reset failure count on success
                        let mut j = jobs.write().await;
                        if let Some(j) = j.get_mut(&job_id) {
                            j.consecutive_failures = 0;
                        }
                    }
                    Err(error) => {
                        tracing::error!(
                            cron_id = %job_id,
                            %error,
                            "cron job execution failed"
                        );

                        let should_disable = {
                            let mut j = jobs.write().await;
                            if let Some(j) = j.get_mut(&job_id) {
                                j.consecutive_failures += 1;
                                j.consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                            } else {
                                false
                            }
                        };

                        if should_disable {
                            tracing::warn!(
                                cron_id = %job_id,
                                "circuit breaker tripped after {MAX_CONSECUTIVE_FAILURES} consecutive failures, disabling"
                            );

                            {
                                let mut j = jobs.write().await;
                                if let Some(j) = j.get_mut(&job_id) {
                                    j.enabled = false;
                                }
                            }

                            // Persist the disabled state
                            if let Err(error) = context.store.update_enabled(&job_id, false).await {
                                tracing::error!(%error, "failed to persist cron job disabled state");
                            }

                            break;
                        }
                    }
                }
            }
        });

        let mut timers = self.timers.write().await;
        timers.insert(job_id_for_map, handle);
    }

    /// Shutdown all cron job timers and wait for them to finish.
    pub async fn shutdown(&self) {
        let handles: Vec<(String, tokio::task::JoinHandle<()>)> = {
            let mut timers = self.timers.write().await;
            timers.drain().collect()
        };

        for (id, handle) in handles {
            handle.abort();
            let _ = handle.await;
            tracing::debug!(cron_id = %id, "cron timer stopped");
        }
    }

    /// Unregister and stop a cron job.
    pub async fn unregister(&self, job_id: &str) {
        // Remove the timer handle and abort it
        let handle = {
            let mut timers = self.timers.write().await;
            timers.remove(job_id)
        };

        if let Some(handle) = handle {
            handle.abort();
            let _ = handle.await;
            tracing::debug!(cron_id = %job_id, "cron timer stopped");
        }

        // Remove the job from the jobs map
        let removed = {
            let mut jobs = self.jobs.write().await;
            jobs.remove(job_id).is_some()
        };

        if removed {
            tracing::info!(cron_id = %job_id, "cron job unregistered");
        }
    }

    /// Check if a job is currently registered.
    pub async fn is_registered(&self, job_id: &str) -> bool {
        let jobs = self.jobs.read().await;
        jobs.contains_key(job_id)
    }

    /// Trigger a cron job immediately, outside the timer loop.
    pub async fn trigger_now(&self, job_id: &str) -> Result<()> {
        let job = {
            let jobs = self.jobs.read().await;
            jobs.get(job_id).cloned()
        };

        if let Some(job) = job {
            if !job.enabled {
                return Err(crate::error::Error::Other(anyhow::anyhow!(
                    "cron job is disabled"
                )));
            }

            tracing::info!(cron_id = %job_id, "cron job triggered manually");
            let run_result = run_cron_job(&job, &self.context).await;

            if is_one_shot_schedule(&job.schedule) {
                tracing::info!(cron_id = %job_id, "one-shot cron job was manually triggered; disabling");
                {
                    let mut jobs = self.jobs.write().await;
                    if let Some(existing_job) = jobs.get_mut(job_id) {
                        existing_job.enabled = false;
                    }
                }
                if let Err(error) = self.context.store.update_enabled(job_id, false).await {
                    tracing::error!(%error, cron_id = %job_id, "failed to persist one-shot disabled state");
                }
            }

            run_result
        } else {
            Err(crate::error::Error::Other(anyhow::anyhow!(
                "cron job not found"
            )))
        }
    }

    /// Update a job's enabled state and manage its timer accordingly.
    pub async fn set_enabled(&self, job_id: &str, enabled: bool) -> Result<()> {
        // Try to find the job in the in-memory HashMap.
        let in_memory = {
            let jobs = self.jobs.read().await;
            jobs.contains_key(job_id)
        };

        if !in_memory {
            if !enabled {
                tracing::debug!(cron_id = %job_id, "set_enabled(false): job not in scheduler, nothing to do");
                return Ok(());
            }

            // Cold re-enable: job was disabled at startup so was never loaded into the scheduler.
            // Reload from the store, insert, then start the timer.
            tracing::info!(cron_id = %job_id, "cold re-enable: reloading config from store");
            let configs = self.context.store.load_all_unfiltered().await?;
            let config = configs
                .into_iter()
                .find(|c| c.id == job_id)
                .ok_or_else(|| {
                    crate::error::Error::Other(anyhow::anyhow!("cron job not found in store"))
                })?;

            let delivery_target =
                DeliveryTarget::parse(&config.delivery_target).unwrap_or_else(|| DeliveryTarget {
                    adapter: "unknown".into(),
                    target: config.delivery_target.clone(),
                });

            {
                let mut jobs = self.jobs.write().await;
                jobs.insert(
                    job_id.to_string(),
                    CronJob {
                        id: config.id.clone(),
                        prompt: config.prompt,
                        schedule: config.schedule,
                        delivery_target,
                        enabled: true,
                        run_once: config.run_once,
                        consecutive_failures: 0,
                        timeout_secs: config.timeout_secs,
                    },
                );
            }

            self.start_timer(job_id).await;
            tracing::info!(cron_id = %job_id, "cron job cold-re-enabled and timer started");
            return Ok(());
        }

        let was_enabled = {
            let mut jobs = self.jobs.write().await;
            if let Some(job) = jobs.get_mut(job_id) {
                let old = job.enabled;
                job.enabled = enabled;
                old
            } else {
                return Err(crate::error::Error::Other(anyhow::anyhow!(
                    "cron job not found"
                )));
            }
        };

        // If enabling and wasn't enabled before, start the timer
        if enabled && !was_enabled {
            self.start_timer(job_id).await;
            tracing::info!(cron_id = %job_id, "cron job enabled and timer started");
        }

        // If disabling, the timer loop will detect this and stop naturally
        if !enabled && was_enabled {
            tracing::info!(cron_id = %job_id, "cron job disabled, timer will stop on next tick");
        }

        Ok(())
    }
}

fn parse_one_shot_timestamp(
    schedule: &str,
) -> std::result::Result<Option<chrono::DateTime<chrono::Local>>, String> {
    let trimmed_schedule = schedule.trim();
    if !trimmed_schedule.starts_with(ONE_SHOT_PREFIX) {
        return Ok(None);
    }

    let raw_timestamp = trimmed_schedule
        .strip_prefix(ONE_SHOT_PREFIX)
        .unwrap_or_default()
        .trim();
    if raw_timestamp.is_empty() {
        return Err("missing RFC3339 timestamp after '@once:'".to_string());
    }

    let timestamp = chrono::DateTime::parse_from_rfc3339(raw_timestamp)
        .map_err(|error| format!("invalid one-shot timestamp '{raw_timestamp}': {error}"))?;

    Ok(Some(timestamp.with_timezone(&chrono::Local)))
}

fn next_occurrence(
    schedule: &str,
    now: chrono::DateTime<chrono::Local>,
) -> std::result::Result<(chrono::DateTime<chrono::Local>, bool), String> {
    if let Some(run_at) = parse_one_shot_timestamp(schedule)? {
        return Ok((run_at, true));
    }

    let cron = croner::Cron::from_str(schedule)
        .map_err(|error| format!("invalid cron expression '{schedule}': {error}"))?;
    let next = cron
        .find_next_occurrence(&now, false)
        .map_err(|error| format!("failed to compute next cron occurrence: {error}"))?;
    Ok((next, false))
}

/// Execute a single cron job: create a fresh channel, run the prompt, deliver the result.
#[tracing::instrument(skip(context), fields(cron_id = %job.id, agent_id = %context.deps.agent_id))]
async fn run_cron_job(job: &CronJob, context: &CronContext) -> Result<()> {
    let channel_id: crate::ChannelId = Arc::from(format!("cron:{}", job.id).as_str());

    // Create the outbound response channel to collect whatever the channel produces
    let (response_tx, mut response_rx) = tokio::sync::mpsc::channel::<OutboundResponse>(32);

    // Subscribe to the agent's event bus (the channel needs this for branch/worker events)
    let event_rx = context.deps.event_tx.subscribe();

    let (channel, channel_tx) = Channel::new(
        channel_id.clone(),
        context.deps.clone(),
        response_tx,
        event_rx,
        context.screenshot_dir.clone(),
        context.logs_dir.clone(),
    );

    // Spawn the channel's event loop
    let channel_handle = tokio::spawn(async move {
        if let Err(error) = channel.run().await {
            tracing::error!(%error, "cron channel failed");
        }
    });

    // Send the cron job prompt as a synthetic message
    let message = InboundMessage {
        id: uuid::Uuid::new_v4().to_string(),
        source: "cron".into(),
        conversation_id: format!("cron:{}", job.id),
        sender_id: "system".into(),
        agent_id: Some(context.deps.agent_id.clone()),
        content: MessageContent::Text(job.prompt.clone()),
        timestamp: chrono::Utc::now(),
        metadata: HashMap::new(),
        formatted_author: None,
    };

    channel_tx
        .send(message)
        .await
        .map_err(|error| anyhow::anyhow!("failed to send cron prompt to channel: {error}"))?;

    // Collect responses with a timeout. The channel may produce multiple messages
    // (e.g. status updates, then text). We only care about text responses.
    let mut collected_text = Vec::new();
    let timeout = Duration::from_secs(job.timeout_secs.unwrap_or(120));

    // Drop the sender so the channel knows no more messages are coming.
    // The channel will process the one message and then its event loop will end
    // when the sender is dropped (message_rx returns None).
    drop(channel_tx);

    loop {
        match tokio::time::timeout(timeout, response_rx.recv()).await {
            Ok(Some(OutboundResponse::Text(text))) => {
                collected_text.push(text);
            }
            Ok(Some(OutboundResponse::RichMessage { text, .. })) => {
                collected_text.push(text);
            }
            Ok(Some(_)) => {
                // Status updates, stream chunks, etc. — ignore for cron jobs
            }
            Ok(None) => {
                // Channel finished (response_tx dropped)
                break;
            }
            Err(_) => {
                tracing::warn!(cron_id = %job.id, "cron job timed out after {timeout:?}");
                channel_handle.abort();
                break;
            }
        }
    }

    // Wait for the channel task to finish (it should already be done since we dropped channel_tx)
    let _ = channel_handle.await;

    let result_text = collected_text.join("\n\n");
    let has_result = !result_text.trim().is_empty();

    // Log execution
    let summary = if has_result {
        Some(result_text.as_str())
    } else {
        None
    };
    if let Err(error) = context.store.log_execution(&job.id, true, summary).await {
        tracing::warn!(%error, "failed to log cron execution");
    }

    // Deliver result to target (only if there's something to say)
    if has_result {
        if let Err(error) = context
            .messaging_manager
            .broadcast(
                &job.delivery_target.adapter,
                &job.delivery_target.target,
                OutboundResponse::Text(result_text),
            )
            .await
        {
            tracing::error!(
                cron_id = %job.id,
                target = %job.delivery_target,
                %error,
                "failed to deliver cron result"
            );
            // Log the delivery failure
            let _ = context
                .store
                .log_execution(&job.id, false, Some(&error.to_string()))
                .await;
            return Err(error);
        }

        tracing::info!(
            cron_id = %job.id,
            target = %job.delivery_target,
            "cron result delivered"
        );
    } else {
        tracing::debug!(cron_id = %job.id, "cron job produced no output, skipping delivery");
    }

    Ok(())
}
