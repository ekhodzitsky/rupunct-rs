//! Hard caps so a crafted checkpoint cannot size-overflow GEMM or RAM.

use std::path::Path;

use crate::error::{Error, Result};

pub(crate) const MAX_HIDDEN: usize = 4096;
pub(crate) const MAX_LAYERS: usize = 48;
pub(crate) const MAX_HEADS: usize = 64;
pub(crate) const MAX_INTERMEDIATE: usize = 16_384;
pub(crate) const MAX_POSITIONS: usize = 8192;
pub(crate) const MIN_POSITIONS: usize = 3;
pub(crate) const MAX_VOCAB: usize = 500_000;
pub(crate) const MAX_LABELS: usize = 256;
pub(crate) const MAX_TYPE_VOCAB: usize = 32;
pub(crate) const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
pub(crate) const MAX_TOKENIZER_BYTES: u64 = 32 * 1024 * 1024;
pub(crate) const MAX_MODEL_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub(crate) const MAX_WORD_CHARS: usize = 256;
pub(crate) const MAX_ADDED_TOKENS: usize = 4096;
pub(crate) const MAX_RESTORE_BYTES: usize = 8 * 1024 * 1024;

pub(crate) fn read_capped_text(path: &Path, max: u64) -> Result<String> {
    let n = std::fs::metadata(path)
        .map_err(|e| Error::io(path, e))?
        .len();
    if n > max {
        return Err(Error::config(format!(
            "{} is {n} bytes (max {max})",
            path.display()
        )));
    }
    std::fs::read_to_string(path).map_err(|e| Error::io(path, e))
}

pub(crate) fn check_cap(name: &str, value: usize, max: usize) -> Result<()> {
    if value > max {
        return Err(Error::config(format!("{name} {value} exceeds max {max}")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_cap_and_read_capped_text() {
        assert!(check_cap("hidden_size", 4, MAX_HIDDEN).is_ok());
        let err = check_cap("hidden_size", MAX_HIDDEN + 1, MAX_HIDDEN)
            .unwrap_err()
            .to_string();
        assert!(err.contains("hidden_size"), "{err}");
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("f.txt");
        std::fs::write(&path, "hello").unwrap();
        assert_eq!(read_capped_text(&path, 8).unwrap(), "hello");
        let err = read_capped_text(&path, 4).unwrap_err().to_string();
        assert!(err.contains("bytes"), "{err}");
        assert!(read_capped_text(&tmp.path().join("missing"), 8).is_err());
    }
}
