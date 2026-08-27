use super::*;
use crate::fixture::{TinySpec, write_tiny_model};
use crate::wordpiece::Tokenizer;

fn tokenizer() -> (tempfile::TempDir, Tokenizer) {
    let tmp = tempfile::tempdir().unwrap();
    write_tiny_model(tmp.path(), &TinySpec::default()).unwrap();
    let tok = Tokenizer::from_file(&tmp.path().join(crate::TOKENIZER_FILE)).unwrap();
    (tmp, tok)
}

#[test]
fn pack_windows_single_when_short() {
    let (_tmp, tok) = tokenizer();
    let words: Vec<String> = vec!["привет".into(), "мир".into()];
    let wins = pack_windows(&tok, &words, 12, 8);
    assert_eq!(
        wins,
        vec![Window {
            start: 0,
            end: 2,
            skip: 0
        }]
    );
}

#[test]
fn pack_windows_splits_when_over_budget() {
    let (_tmp, tok) = tokenizer();
    let words: Vec<String> = (0..15).map(|i| format!("w{i}")).collect();
    let wins = pack_windows(&tok, &words, 8, 2);
    assert!(wins.len() >= 2, "got {wins:?}");
    let committed: usize = wins.iter().map(|w| w.end - w.start - w.skip).sum();
    assert_eq!(committed, 15);
    assert_eq!(wins[0].skip, 0);
    assert!(wins[1].skip > 0);
}

#[test]
fn restore_opts_resolve_clamps() {
    let o = RestoreOpts {
        max_window_tokens: 10_000,
        overlap_tokens: 9_000,
    };
    assert_eq!(o.resolve(512), (512, 511));
    let o = RestoreOpts {
        max_window_tokens: 1,
        overlap_tokens: 0,
    };
    assert_eq!(o.resolve(2048), (2, 0));
}

#[test]
fn pack_windows_empty() {
    let (_tmp, tok) = tokenizer();
    assert!(pack_windows(&tok, &[], 8, 2).is_empty());
}

#[test]
fn fit_seq_truncates_keeping_sep() {
    let mut ids = vec![2, 5, 6, 7, 8, 3];
    let mut mask = vec![1; 6];
    let mut wids = vec![None, Some(0), Some(1), Some(2), Some(3), None];
    fit_seq(&mut ids, &mut mask, &mut wids, 4);
    assert_eq!(ids.len(), 4);
    assert_eq!(ids[0], 2);
    assert_eq!(*ids.last().unwrap(), 3);
    assert_eq!(mask.len(), 4);
    assert_eq!(wids.len(), 4);
    assert_eq!(wids[0], None);
    assert_eq!(wids[3], None);
}

#[test]
fn fit_seq_noop_when_short() {
    let mut ids = vec![2, 5, 3];
    let mut mask = vec![1; 3];
    let mut wids = vec![None, Some(0), None];
    fit_seq(&mut ids, &mut mask, &mut wids, 8);
    assert_eq!(ids, vec![2, 5, 3]);
}
