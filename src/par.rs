//! Bounded `std::thread` fan-out. No rayon.

use crate::error::{Error, Result};

pub(crate) fn parallelism(requested: usize) -> usize {
    if requested == 0 {
        std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
    } else {
        requested.max(1)
    }
}

pub(crate) fn worker_count(requested: usize, n_work: usize) -> usize {
    if n_work <= 1 {
        return 1;
    }
    parallelism(requested).min(n_work).max(1)
}

/// `(item_jobs, threads_per_item)` so concurrent work stays within `requested`.
pub(crate) fn split_budget(requested: usize, n_items: usize) -> (usize, usize) {
    let total = parallelism(requested);
    if n_items <= 1 {
        return (1, total);
    }
    let item_jobs = total.min(n_items).max(1);
    let per = (total / item_jobs).max(1);
    (item_jobs, per)
}

pub(crate) fn parallel_map<T, R, F>(items: &[T], jobs: usize, f: F) -> Result<Vec<R>>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> Result<R> + Sync + Send,
{
    if items.is_empty() {
        return Ok(Vec::new());
    }
    if jobs <= 1 || items.len() == 1 {
        return items.iter().map(f).collect();
    }
    let jobs = jobs.min(items.len()).max(1);
    let f = &f;
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(jobs);
        for worker in 0..jobs {
            handles.push(scope.spawn(move || -> Result<Vec<(usize, R)>> {
                let mut out = Vec::new();
                let mut i = worker;
                while i < items.len() {
                    out.push((i, f(&items[i])?));
                    i += jobs;
                }
                Ok(out)
            }));
        }
        let mut slots: Vec<Option<R>> = (0..items.len()).map(|_| None).collect();
        for handle in handles {
            let chunk = match handle.join() {
                Ok(Ok(chunk)) => chunk,
                Ok(Err(err)) => return Err(err),
                Err(_) => return Err(Error::model("restore thread panicked")),
            };
            for (i, val) in chunk {
                if let Some(slot) = slots.get_mut(i) {
                    *slot = Some(val);
                }
            }
        }
        slots
            .into_iter()
            .map(|s| s.ok_or_else(|| Error::model("missing parallel result")))
            .collect()
    })
}

#[cfg(test)]
#[path = "par_tests.rs"]
mod tests;
