//! A tiny logger: stderr only, because stdout carries the protocol and is never written to by this package.

use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Debug = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
}

impl Level {
    pub fn parse(text: &str) -> Option<Level> {
        match text {
            "debug" => Some(Level::Debug),
            "info" => Some(Level::Info),
            "warning" => Some(Level::Warning),
            "error" => Some(Level::Error),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Level::Debug => "DEBUG",
            Level::Info => "INFO",
            Level::Warning => "WARNING",
            Level::Error => "ERROR",
        }
    }
}

static LEVEL: AtomicU8 = AtomicU8::new(Level::Warning as u8);

pub fn set_level(level: Level) {
    LEVEL.store(level as u8, Ordering::Relaxed);
}

/// Write one line to stderr when `level` is at or above the configured level.
pub fn log(level: Level, target: &str, message: fmt::Arguments<'_>) {
    if (level as u8) >= LEVEL.load(Ordering::Relaxed) {
        eprintln!("{target} {} {message}", level.name());
    }
}
