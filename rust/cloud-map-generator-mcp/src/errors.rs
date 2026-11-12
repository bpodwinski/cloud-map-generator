//! The errors of the server. Their text is what an AI reads, so each one says what was wrong and what to do.

use std::fmt;

/// Why a tool call or a job did not produce a result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppError {
    /// A request the caller can fix (a bad argument, a path outside the sandbox, a full queue): the text is
    /// returned to the AI verbatim. Python: `UserError` and `SandboxError`.
    User(String),
    /// A refusal of the engine behind the server, written for an AI reader: returned with every absolute path
    /// scrubbed out. Python: `ServiceError`.
    Service(String),
    /// The caller's `cancel` callback returned true.
    Cancelled,
    /// A crash. `kind` names it (`Panic`, `IoError`...); the details stay in the server log.
    Internal { kind: String, detail: String },
}

impl AppError {
    pub fn user(message: impl Into<String>) -> Self {
        AppError::User(message.into())
    }

    pub fn internal(kind: impl Into<String>, detail: impl Into<String>) -> Self {
        AppError::Internal { kind: kind.into(), detail: detail.into() }
    }
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::User(m) | AppError::Service(m) => f.write_str(m),
            AppError::Cancelled => f.write_str("cancelled"),
            AppError::Internal { kind, detail } => write!(f, "internal error ({kind}): {detail}"),
        }
    }
}

impl std::error::Error for AppError {}

pub type AppResult<T> = Result<T, AppError>;

/// A path or a name the sandbox refuses. The message says why and what to do instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxError(pub String);

impl fmt::Display for SandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SandboxError {}

impl From<SandboxError> for AppError {
    fn from(e: SandboxError) -> Self {
        AppError::User(e.0)
    }
}
