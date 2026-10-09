pub mod config;
pub mod diagnostics;
pub mod files;
pub mod runtime;
pub mod security;
pub mod workspace;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid configuration: {0}")]
    Config(String),
    #[error("Operation not permitted: {0}")]
    Permission(String),
    #[error("Runtime error: {0}")]
    Runtime(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
