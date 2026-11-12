//! The one error type of the command line: a message for the person at the terminal (Python's
//! `raise SystemExit('message')`), printed as `cloudgen: <message>` with exit code 1.

use std::fmt;

#[derive(Debug)]
pub struct Fail(pub String);

pub type Result<T> = std::result::Result<T, Fail>;

impl fmt::Display for Fail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Fail {}

impl Fail {
    pub fn new(message: impl Into<String>) -> Fail {
        Fail(message.into())
    }
}

/// `fail!("...", args)` returns `Err(Fail)` from the current function.
#[macro_export]
macro_rules! fail {
    ($($arg:tt)*) => {
        return Err($crate::fail::Fail(format!($($arg)*)))
    };
}

/// Adds the path an I/O error happened on (`std::io::Error` does not carry it).
pub trait Context<T> {
    fn context(self, what: impl FnOnce() -> String) -> Result<T>;
}

impl<T, E: fmt::Display> Context<T> for std::result::Result<T, E> {
    fn context(self, what: impl FnOnce() -> String) -> Result<T> {
        self.map_err(|e| Fail(format!("{}: {e}", what())))
    }
}
