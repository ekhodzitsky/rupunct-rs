use super::*;
use crate::error::Error;

#[test]
fn worker_count_caps() {
    assert_eq!(worker_count(0, 0), 1);
    assert_eq!(worker_count(8, 1), 1);
    assert_eq!(worker_count(1, 10), 1);
    assert_eq!(worker_count(8, 3), 3);
    let auto = worker_count(0, 64);
    assert!((1..=64).contains(&auto), "{auto}");
}

#[test]
fn split_budget_shares_pool() {
    assert_eq!(split_budget(1, 10), (1, 1));
    assert_eq!(split_budget(8, 2), (2, 4));
    assert_eq!(split_budget(8, 100), (8, 1));
    let (jobs, per) = split_budget(0, 1);
    assert_eq!(jobs, 1);
    assert!(per >= 1);
}

#[test]
fn parallel_map_preserves_order() {
    let xs: Vec<usize> = (0..20).collect();
    let out = parallel_map(&xs, 4, |x| Ok(x * 2)).unwrap();
    assert_eq!(out, (0..20).map(|x| x * 2).collect::<Vec<_>>());
}

#[test]
fn parallel_map_propagates_first_index_error() {
    let xs: Vec<usize> = (0..8).collect();
    let err = parallel_map(&xs, 4, |x| {
        if *x == 3 {
            Err(Error::model("boom"))
        } else {
            Ok(*x)
        }
    })
    .unwrap_err();
    assert!(err.to_string().contains("boom"), "{err}");
}
