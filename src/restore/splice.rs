//! Patch the original string: replace each word span, delete stripped punct.

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Edit {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

/// Apply non-overlapping `[start, end)` replacements to `original`.
/// `start`/`end` are byte offsets and must be char boundaries.
pub(crate) fn splice(original: &str, edits: &mut [Edit]) -> Result<String> {
    edits.sort_by_key(|e| e.start);
    let mut out = String::with_capacity(original.len());
    let mut cur = 0usize;
    for e in edits.iter() {
        if e.start < cur
            || e.end < e.start
            || e.end > original.len()
            || !original.is_char_boundary(e.start)
            || !original.is_char_boundary(e.end)
        {
            return Err(Error::model(format!(
                "splice range {}..{} invalid (cur {cur}, len {})",
                e.start,
                e.end,
                original.len()
            )));
        }
        out.push_str(&original[cur..e.start]);
        out.push_str(&e.text);
        cur = e.end;
    }
    out.push_str(&original[cur..]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splice_keeps_gap_and_replaces_spans() {
        let mut eds = [
            Edit {
                start: 0,
                end: 2,
                text: "AB".into(),
            },
            Edit {
                start: 3,
                end: 5,
                text: "CD".into(),
            },
        ];
        assert_eq!(splice("xx-yy", &mut eds).unwrap(), "AB-CD");
    }

    #[test]
    fn splice_delete_span() {
        let mut eds = [Edit {
            start: 1,
            end: 2,
            text: String::new(),
        }];
        assert_eq!(splice("a,b", &mut eds).unwrap(), "ab");
    }

    #[test]
    fn splice_rejects_overlap() {
        let mut eds = [
            Edit {
                start: 0,
                end: 3,
                text: "a".into(),
            },
            Edit {
                start: 2,
                end: 4,
                text: "b".into(),
            },
        ];
        assert!(splice("xxxx", &mut eds).is_err());
    }
}
