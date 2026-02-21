//! LLM provider management and routing.

pub mod anthropic;
pub(crate) mod google_antigravity;
pub mod manager;
pub mod model;
pub mod providers;
pub mod routing;

pub use manager::LlmManager;
pub use model::SpacebotModel;
pub use routing::RoutingConfig;
