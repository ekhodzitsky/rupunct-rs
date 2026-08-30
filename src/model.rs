#[cfg(test)]
use std::path::Path;
use std::path::PathBuf;

/// Checkpoint file names expected in a model directory.
pub const MODEL_FILE: &str = "model.safetensors";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const CONFIG_FILE: &str = "config.json";

/// Directory name under `~/.rupunct/` for the default checkpoint (`RUPunct_big`).
pub const DEFAULT_CHECKPOINT_DIR: &str = "RUPunct_big";

/// `$RUPUNCT_MODEL` if set and non-empty, else `~/.rupunct/RUPunct_big`, else
/// `RUPunct_big`.
pub fn default_model_dir() -> PathBuf {
    model_dir_from(std::env::var("RUPUNCT_MODEL").ok(), home_dir())
}

fn home_dir() -> Option<PathBuf> {
    home_dir_from(std::env::var_os("HOME"), std::env::var_os("USERPROFILE"))
}

pub(crate) fn home_dir_from(
    home: Option<std::ffi::OsString>,
    userprofile: Option<std::ffi::OsString>,
) -> Option<PathBuf> {
    home.or(userprofile).map(PathBuf::from)
}

pub(crate) fn model_dir_from(env: Option<String>, home: Option<PathBuf>) -> PathBuf {
    if let Some(p) = env {
        let t = p.trim();
        if !t.is_empty() {
            return PathBuf::from(t);
        }
    }
    match home {
        Some(h) => h.join(".rupunct").join(DEFAULT_CHECKPOINT_DIR),
        None => PathBuf::from(DEFAULT_CHECKPOINT_DIR),
    }
}

/// Pinned `RUPunct_small` install path (`~/.rupunct/punct`) for live goldens.
#[cfg(test)]
pub(crate) fn live_small_model_dir() -> Option<PathBuf> {
    let punct = home_dir()?.join(".rupunct").join("punct");
    Path::new(&punct)
        .join(MODEL_FILE)
        .is_file()
        .then_some(punct)
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
