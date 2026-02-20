//! Messaging adapters (Discord, Slack, Telegram, Twitch, Webhook, WebChat).

#[cfg(feature = "discord")]
pub mod discord;
pub mod manager;
#[cfg(feature = "slack")]
pub mod slack;
pub mod telegram;
pub mod traits;
#[cfg(feature = "twitch")]
pub mod twitch;
pub mod webchat;
pub mod webhook;

pub use manager::MessagingManager;
pub use traits::Messaging;
