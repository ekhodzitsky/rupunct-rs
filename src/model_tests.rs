use super::*;
use std::path::PathBuf;

#[test]
fn env_override_wins() {
    let dir = model_dir_from(
        Some("  /opt/punct  ".into()),
        Some(PathBuf::from("/home/user")),
    );
    assert_eq!(dir, PathBuf::from("/opt/punct"));
}

#[test]
fn blank_env_falls_back_to_home() {
    let dir = model_dir_from(Some("  ".into()), Some(PathBuf::from("/home/user")));
    assert_eq!(dir, PathBuf::from("/home/user/.rupunct/RUPunct_big"));
}

#[test]
fn missing_env_uses_home() {
    let dir = model_dir_from(None, Some(PathBuf::from("/home/user")));
    assert_eq!(dir, PathBuf::from("/home/user/.rupunct/RUPunct_big"));
}

#[test]
fn no_home_uses_relative_default_checkpoint() {
    assert_eq!(
        model_dir_from(None, None),
        PathBuf::from(DEFAULT_CHECKPOINT_DIR)
    );
    assert_eq!(
        model_dir_from(Some(String::new()), None),
        PathBuf::from(DEFAULT_CHECKPOINT_DIR)
    );
}

#[test]
fn file_name_constants() {
    assert_eq!(MODEL_FILE, "model.safetensors");
    assert_eq!(TOKENIZER_FILE, "tokenizer.json");
    assert_eq!(CONFIG_FILE, "config.json");
}

#[test]
fn home_dir_prefers_home_then_userprofile() {
    use std::ffi::OsString;
    assert_eq!(
        home_dir_from(
            Some(OsString::from("/home/u")),
            Some(OsString::from("C:\\Users\\u"))
        ),
        Some(PathBuf::from("/home/u"))
    );
    assert_eq!(
        home_dir_from(None, Some(OsString::from("C:\\Users\\u"))),
        Some(PathBuf::from("C:\\Users\\u"))
    );
    assert_eq!(home_dir_from(None, None), None);
}

#[test]
fn default_model_dir_is_absolute_or_checkpoint() {
    let dir = default_model_dir();
    assert!(
        dir.ends_with(DEFAULT_CHECKPOINT_DIR)
            || dir.as_os_str() == DEFAULT_CHECKPOINT_DIR
            || dir.is_absolute()
    );
}
