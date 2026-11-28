//! Opt-in physical backing samples. This is committed virtual address space, not a sum that
//! can be subtracted from process PrivateUsage or added to DXGI's local/non-local counters.

use super::{mem, page_home_in, SparseHome, AWAY, PHYS_BLOCKS, PHYS_PAGE};
use std::time::{Duration, Instant};

/// A bounded metadata snapshot, then Windows queries with no title mapping lock held.
pub fn line() -> Option<String> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("XCL_MEMORY_OWNERS").is_ok_and(|v| v == "1")) {
        return None;
    }
    let (mut ranges, mut private, live, reserved, deferred, mapped) = {
        let Ok(away) = AWAY.try_lock() else { return Some("physical memory owners: mapping snapshot busy".into()) };
        let Ok(blocks) = PHYS_BLOCKS.try_lock() else { return Some("physical memory owners: allocation snapshot busy".into()) };
        let mapped = away.as_ref().map_or(0, |a| a.at.len());
        let private_count = away.as_ref().map_or(0, |a| a.sparse_homes.len());
        if blocks.len().saturating_add(mapped).saturating_add(private_count) > 262_144 {
            return Some("physical memory owners: snapshot exceeds 262144 ranges; not sampled".into());
        }
        let mut ranges: Vec<_> = blocks.values().map(|b| (b.home, b.count * PHYS_PAGE)).collect();
        let live = blocks.values().map(|b| b.live * PHYS_PAGE).sum::<usize>();
        let reserved = blocks.values().map(|b| b.count * PHYS_PAGE).sum::<usize>();
        let deferred = blocks.values().filter(|b| b.live == 0).count();
        let mut private = Vec::new();
        if let Some(a) = away.as_ref() {
            ranges.extend(a.at.keys().map(|&at| (at, PHYS_PAGE)));
            private.extend(a.sparse_homes.iter().filter(|&(p, state)| *state == SparseHome::CpuOnly && !a.page.contains_key(p))
                .filter_map(|(&p, _)| page_home_in(&blocks, p)).filter(|home| !a.at.contains_key(home)).map(|home| (home, PHYS_PAGE)));
        }
        (ranges, private, live, reserved, deferred, mapped)
    };
    let started = Instant::now();
    let (private_bytes, private_queries, private_complete) = committed_ranges(&mut private, started, 16_384);
    let (committed, queries, complete) = committed_ranges(&mut ranges, started, 16_384 - private_queries);
    Some(format!(
        "physical memory owners: {} MB live page IDs; {} MB home address space ({} empty blocks held for foreign mappings), {} away mappings; {}{} MB committed CPU VA in known backing, including {}{} MB at private homes with GPU tiles deferred ({} queries, {} ms, concurrent snapshot; excludes old away bytes after free, not process/VRAM totals or measured VRAM savings)",
        live >> 20, reserved >> 20, deferred, mapped, if complete { "" } else { "at least " }, committed >> 20,
        if private_complete { "" } else { "at least " }, private_bytes >> 20, queries + private_queries, started.elapsed().as_millis()
    ))
}

fn committed_ranges(ranges: &mut [(usize, usize)], started: Instant, query_limit: usize) -> (usize, usize, bool) {
    ranges.sort_unstable();
    let mut runs = 0usize;
    for i in 0..ranges.len() {
        let (base, len) = ranges[i];
        let Some(end) = base.checked_add(len) else { return (0, 0, false) };
        if runs > 0 && base <= ranges[runs - 1].0 + ranges[runs - 1].1 {
            let prior = &mut ranges[runs - 1];
            prior.1 = prior.1.max(end - prior.0);
        } else {
            ranges[runs] = (base, len);
            runs += 1;
        }
    }
    let ranges = &ranges[..runs];
    let (mut bytes, mut queries, mut covered) = (0usize, 0usize, 0usize);
    for &(base, len) in ranges.iter() {
        let Some(end) = base.checked_add(len) else { return (bytes, queries, false) };
        let mut at = base.max(covered);
        while at < end {
            if queries >= query_limit || started.elapsed() >= Duration::from_millis(100) {
                return (bytes, queries, false);
            }
            let mut region: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            queries += 1;
            if unsafe { mem::VirtualQuery(at as *const _, &mut region, std::mem::size_of_val(&region)) } == 0 {
                return (bytes, queries, false);
            }
            let Some(next) = (region.BaseAddress as usize).checked_add(region.RegionSize).filter(|&n| n > at) else {
                return (bytes, queries, false);
            };
            let next = next.min(end);
            if region.State == mem::MEM_COMMIT {
                bytes += next - at;
            }
            at = next;
        }
        covered = covered.max(end);
    }
    (bytes, queries, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_backing_counts_overlapping_ranges_once() {
        unsafe {
            let p = mem::VirtualAlloc(std::ptr::null(), 2 * PHYS_PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE);
            assert!(!p.is_null());
            assert!(!mem::VirtualAlloc(p, PHYS_PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());
            let mut ranges = [(p as usize, 2 * PHYS_PAGE), (p as usize, PHYS_PAGE)];
            let (bytes, _, complete) = committed_ranges(&mut ranges, Instant::now(), 16_384);
            assert!(complete);
            assert_eq!(bytes, PHYS_PAGE);
            assert_ne!(mem::VirtualFree(p, 0, mem::MEM_RELEASE), 0);
        }
    }
}
