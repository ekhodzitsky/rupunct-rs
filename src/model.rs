#[cfg(test)]
use std::path::Path;
use std::path::PathBuf;

/// Checkpoint file names expected in a model directory.
pub const MODEL_FILE: &str = "model.safetensors";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const CONFIG_FILE: &str = "config.json";

/// `$RUPUNCT_MODEL` if set and non-empty, else `~/.rupunct/punct`, else `punct`.
pub fn default_model_dir() -> PathBuf {
    model_dir_from(std::env::var("RUPUNCT_MODEL").ok(), home_dir())
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

pub(crate) fn model_dir_from(env: Option<String>, home: Option<PathBuf>) -> PathBuf {
    if let Some(p) = env {
        let t = p.trim();
        if !t.is_empty() {
            return PathBuf::from(t);
        }
    }
    match home {
        Some(h) => h.join(".rupunct").join("punct"),
        None => PathBuf::from("punct"),
    }
}

#[cfg(test)]
pub(crate) fn live_model_dir() -> Option<PathBuf> {
    let dir = default_model_dir();
    Path::new(&dir).join(MODEL_FILE).is_file().then_some(dir)
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
