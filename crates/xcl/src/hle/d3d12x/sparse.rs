//! The title's own memory, seen by the GPU: sparse windows.
//!
//! A console title keeps GPU data in memory it reserves and commits itself (XDefiant: one
//! 8 GB reservation of 64 KiB pages it commits in steps and places buffers in) – on the
//! console all of it is GPU-visible at the same addresses. XCL gives such reservations
//! addresses in the *sparse windows* of the unified-memory block (`gpu::uma`: windows 0..2
//! in `for_runtime`), so translated shaders reach them through `mem`…`mem2` like any unified
//! memory. On the GPU each sparse window is a reserved (tiled) buffer of 4 GiB whose 64 KiB
//! tiles are mapped onto video-memory heaps as the title commits pages.
//! The CPU's view stays the title's memory: what the CPU wrote (write-watch) is copied into
//! the tiles before the title's next ExecuteCommandLists via an asynchronous ring-buffered
//! staging upload queue, and the title's queues GPU-wait on the upload fence (no CPU stall).
//! Only the bytes the CPU changed go (64-byte chunks against a hash of what was sent last,
//! `changed_chunks`): the rest of a written page may hold what the GPU wrote there (counters,
//! indirect arguments) and must not be overwritten with the CPU's stale view of it.
//! Buffers the title places there get reserved resources over the same tiles, so PC-side
//! copies, views and barriers see the same memory.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ffi::c_void;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_DECOMMIT, PAGE_GUARD,
    PAGE_NOACCESS, PAGE_READWRITE,
};

const TILE: usize = 0x1_0000;
/// tiles of a video-memory pool heap (4 MiB). A heap goes back only when every tile of it is
/// free (`State::retire_free_pools`), and a title frees its tiles in runs spread over all its
/// heaps: with heaps of 64 MiB a retail title's world load unmapped 1.5 GB of its 4.6 GB of
/// tiles and 3 of 78 heaps went back (0.2 GB); with heaps of 4 MiB 1.25 GB went back.
const POOL_TILES: u32 = 64;
/// tiles decommitted while no title work was submitted that stay mapped (`State::parked`) before
/// they are unmapped together: 256 MB of video memory
const PARKED_MAX: usize = 4096;
/// tiles waiting for their mapping call (`State::pending`) before they are sent at once: 256 MB
const PENDING_MAX: usize = 4096;
/// a pool tile freed is not reused before this (`State::cooling`): work the title submitted
/// before its unmap may still read it on the title's queues, and the sparse queue that maps,
/// clears and fills it for its next address does not wait for those
const COOL: std::time::Duration = std::time::Duration::from_secs(2);
/// bytes of the ring-buffered upload staging buffer
const STAGING: usize = 64 << 20;
/// alignment of sub-allocations inside the upload staging ring (write-combined cacheline)
const RING_ALIGN: usize = 64;
/// number of in-flight upload command allocators / lists
const UPLOAD_SLOTS: usize = 16;
/// bytes of the dedicated pre-zeroed buffer for tile/hole clears (1 MiB)
const ZERO_BUF: usize = 1 << 20;
/// bytes of the GPU→CPU readback staging buffer
const READBACK_STAGING: usize = 16 << 20;

/// changed runs of fewer bytes go up together in a compute dispatch (`scatter`) when there are
/// this many of them, of this many chunks at most (4 MiB, so the ring fills in pieces)
const SCATTER_RUN: usize = 4 << 10;
const SCATTER_FROM: usize = 16;
const SCATTER_CHUNKS: usize = 1 << 16;

const PAGE: usize = 0x1000;
const PAGES_PER_TILE: usize = TILE / PAGE;
const CHUNK: usize = super::scatter::CHUNK;
const CHUNKS: usize = PAGE / CHUNK;

/// Fast non-cryptographic integer hasher for page/tile lookup maps on the hot submit path.
#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        }
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn finish(&self) -> u64 {
        // Tile addresses are 64 KiB aligned. Move mixed bits into the low bucket-index bits.
        self.0.rotate_right(23)
    }
}

type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
type FxHashSet<K> = std::collections::HashSet<K, BuildHasherDefault<FxHasher>>;

struct TileEntry {
    pool: (u32, u32),
    /// bit `i` set => 4 KiB page `i` of this 64 KiB tile is uncommitted (a hole)
    holes: u16,
    /// bit `i` set => the title gave page `i` a protection other than writable
    not_rw: u16,
    /// per-4 KiB-page hashes of the 64-byte chunks last sent to the GPU (`None` = all zero)
    sent: [Option<Box<[u64; CHUNKS]>>; PAGES_PER_TILE],
    /// 0: the bytes are at the tile's own address. Else the 64 KiB-aligned CPU address they are
    /// read from and land at (a tile of a GPU view, `State::mirrors`); `holes` are then the
    /// uncommitted pages there. The tile's own address is no CPU memory: the title reserved it
    /// with no access, as the console's GPU-only view of the same pages
    src: usize,
    /// what the CPU memory of a tile over a CPU view is: new for each such entry (`next_gen`), and
    /// again whenever a page of it becomes a hole. A copy back or an immediate recorded with
    /// another lands nowhere: the memory it was for is gone (decommitted and committed again, or
    /// another CPU view shown). 0: a tile of the window's own, whose memory is its address's
    gen: u32,
}

/// The mapped tiles of the windows' own (not GPU views), and how many of them hold video memory
/// no CPU write was ever sent to: every committed page still has no `sent` hashes. Such a tile
/// holds zeros unless the GPU wrote to it.
fn never_sent(mapped: &FxHashMap<usize, TileEntry>) -> (usize, usize) {
    let (mut own, mut unsent) = (0, 0);
    for e in mapped.values().filter(|e| e.src == 0 && e.pool != LAZY) {
        own += 1;
        if e.sent.iter().all(|p| p.is_none()) {
            unsent += 1;
        }
    }
    (own, unsent)
}

/// `TileEntry::pool` of a tile the title committed that has no video memory yet
/// (`XCL_SPARSE_LAZY=1`): it gets a pool tile once the CPU's first bytes go up to it or a
/// placed buffer or a copy back names it (`State::materialize`). Until then the window reads
/// zeros there, as the title's freshly committed pages do.
const LAZY: (u32, u32) = (u32::MAX, u32::MAX);

/// `XCL_SPARSE_LAZY=1`: pages a title commits in a sparse window get their video memory at
/// their first use, not at the commit. Off by default: a compute program that writes through a
/// window to memory no buffer was placed on would write nothing there.
fn lazy_commits() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::LAZY_COMMITS_FOR_TEST.with(|c| c.get()) {
        return on;
    }
    lazy_asked() && !LAZY_REFUSED.load(Ordering::Relaxed)
}

/// `XCL_SPARSE_LAZY=1` was given
fn lazy_asked() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SPARSE_LAZY").is_ok_and(|v| v.trim() == "1"))
}

/// `XCL_SPARSE_LAZY=1` on a device whose unmapped tiles do not read zeros (`null_tiles_read_zero`):
/// commits map at once, as without it
static LAZY_REFUSED: AtomicBool = AtomicBool::new(false);

/// Whether a tile of a reserved buffer with no video memory reads zeros and takes writes that go
/// nowhere on `dev` (tiled resources tier 2). On tier 1 a touch of one is undefined.
unsafe fn null_tiles_read_zero(dev: &ID3D12Device) -> bool {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.TiledResourcesTier.0 >= D3D12_TILED_RESOURCES_TIER_2.0
}

/// The parts of `runs` (address, bytes) that lie in tiles `keep` holds for, cut where a tile
/// ends; neighbours left together are one run.
fn runs_where(runs: &[(usize, usize)], mut keep: impl FnMut(usize) -> bool) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for &(a, n) in runs {
        let end = a.saturating_add(n);
        let mut at = a;
        while at < end {
            let t = at & !(TILE - 1);
            let stop = t.saturating_add(TILE).min(end);
            if keep(t) {
                push_run(&mut out, at, stop - at);
            }
            at = stop;
        }
    }
    out
}

/// the last `TileEntry::gen` given out
static TILE_GEN: AtomicU32 = AtomicU32::new(0);

/// A `TileEntry::gen` for a tile over a CPU view (never 0)
fn next_gen() -> u32 {
    TILE_GEN.fetch_add(1, Ordering::Relaxed) % u32::MAX + 1
}

impl TileEntry {
    fn with_holes(pool: (u32, u32), holes: u16) -> Self {
        Self { pool, holes, not_rw: 0, sent: std::array::from_fn(|_| None), src: 0, gen: 0 }
    }

    /// The CPU address of the tile at `t` (this entry's tile)
    #[inline]
    fn cpu(&self, t: usize) -> usize {
        if self.src == 0 {
            t
        } else {
            self.src
        }
    }
}

/// The hole masks (`query_tile_holes`) of the 64 KiB tiles `tiles` of the title's CPU memory, in
/// their order. A page in the working set is committed: one query asks that of all their pages.
/// The others go to VirtualQuery, once for each region it reports: it walks a region to its end,
/// so a call a tile over a long committed range grew with the square of the range.
unsafe fn source_holes(tiles: &[usize]) -> Vec<u16> {
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    /// tiles a working-set query asks for
    const BATCH: usize = 256;
    let mut out = Vec::with_capacity(tiles.len());
    let mut ws: Vec<PSAPI_WORKING_SET_EX_INFORMATION> = Vec::with_capacity(BATCH.min(tiles.len()) * PAGES_PER_TILE);
    // the region the last VirtualQuery reported: [start, end), and whether its pages are accessible
    let mut region = (0usize, 0usize, false);
    for batch in tiles.chunks(BATCH) {
        ws.clear();
        for &t in batch {
            for i in 0..PAGES_PER_TILE {
                ws.push(PSAPI_WORKING_SET_EX_INFORMATION { VirtualAddress: (t + i * PAGE) as *mut c_void, ..std::mem::zeroed() });
            }
        }
        let asked = K32QueryWorkingSetEx(GetCurrentProcess(), ws.as_mut_ptr() as *mut c_void, std::mem::size_of_val(&ws[..]) as u32) != 0;
        for (j, &t) in batch.iter().enumerate() {
            let mut holes = 0xFFFFu16;
            for i in 0..PAGES_PER_TILE {
                let pg = t + i * PAGE;
                // PSAPI_WORKING_SET_EX_BLOCK: bit 0 Valid, bits 4-14 Win32Protection
                let flags = ws[j * PAGES_PER_TILE + i].VirtualAttributes.Flags;
                let protect = ((flags >> 4) & 0x7FF) as u32;
                let mut accessible = asked && flags & 1 != 0 && protect != 0 && protect & (PAGE_NOACCESS | PAGE_GUARD) == 0;
                if !accessible {
                    if !(region.0..region.1).contains(&pg) {
                        #[cfg(test)]
                        tests::VIRTUAL_QUERIES.with(|n| n.set(n.get() + 1));
                        let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                        region = if VirtualQuery(pg as *const c_void, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) == 0 || mbi.RegionSize == 0 {
                            (pg, pg + PAGE, false)
                        } else {
                            let ok = mbi.State == MEM_COMMIT && mbi.Protect != 0 && (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
                            (pg, (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize).max(pg + PAGE), ok)
                        };
                    }
                    accessible = region.2;
                }
                if accessible {
                    holes &= !(1u16 << i);
                }
            }
            out.push(holes);
        }
    }
    out
}

/// Query actual OS page commitment across the 64 KiB tile at `tile`.
/// Bit `i` in the returned mask is set whenever 4 KiB page `tile + i * PAGE` is not committed
/// and accessible (`MEM_RESERVE`, `MEM_FREE`, `PAGE_NOACCESS`, `PAGE_GUARD`, etc.).
unsafe fn query_tile_holes(tile: usize) -> u16 {
    let mut holes = 0xFFFFu16;
    let end = tile.saturating_add(TILE);
    let mut cur = tile;
    while cur < end {
        let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        // SAFETY: `VirtualQuery` inspects process virtual memory metadata without dereferencing `cur`.
        let n = VirtualQuery(cur as *const c_void, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>());
        if n == 0 || mbi.RegionSize == 0 {
            break;
        }
        let region_end = (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize).min(end);
        let accessible = mbi.State == MEM_COMMIT && mbi.Protect != 0 && (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
        if accessible {
            let mut pg = cur;
            while pg < region_end {
                holes &= !(1u16 << ((pg - tile) / PAGE));
                pg += PAGE;
            }
        }
        if region_end <= cur {
            break;
        }
        cur = region_end;
    }
    holes
}

/// Update `mapped`, `free`, and `to_zero` for a decommit of `[a, a + n)`.
/// Returns the base addresses of tiles that became completely uncommitted (`holes == 0xFFFF`)
/// and were removed from `mapped` (ready to be unmapped from their sparse window buffers).
fn decommit_tiles(
    mapped: &mut FxHashMap<usize, TileEntry>,
    pinned: &FxHashMap<(u32, u32), u32>,
    free: &mut Vec<(u32, u32)>,
    to_zero: &mut BTreeSet<(usize, usize)>,
    a: usize,
    n: usize,
) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let mut unmapped: Vec<usize> = Vec::new();
    let mut pg = a & !(PAGE - 1);
    let end = a.saturating_add(n);
    while pg < end {
        let t = pg & !(TILE - 1);
        let tile_end = t.saturating_add(TILE).min(end);
        if let Some(entry) = mapped.get_mut(&t) {
            let mut newly_holed = 0u16;
            let mut p = pg;
            while p < tile_end {
                let idx = (p - t) / PAGE;
                entry.sent[idx] = None;
                let bit = 1u16 << idx;
                if entry.holes & bit == 0 {
                    entry.holes |= bit;
                    newly_holed |= bit;
                }
                p += PAGE;
            }
            if newly_holed != 0 && entry.src != 0 {
                entry.gen = next_gen();
            }
            if entry.holes == 0xFFFF {
                unmapped.push(t);
            } else if newly_holed != 0 {
                for idx in 0..PAGES_PER_TILE {
                    if newly_holed & (1u16 << idx) != 0 {
                        to_zero.insert((t + idx * PAGE, PAGE));
                    }
                }
            }
        }
        if tile_end <= pg {
            break;
        }
        pg = tile_end;
    }
    for &t in &unmapped {
        if let Some(entry) = mapped.remove(&t) {
            if entry.pool != LAZY && !pinned.contains_key(&entry.pool) {
                free.push(entry.pool);
            }
        }
    }
    // (their pieces only: going over all of them for each page decommitted while a title loads,
    // with no submit to clear them, grew with the square of the pages)
    for &t in &unmapped {
        while let Some(&piece) = to_zero.range((t, 0)..(t + TILE, 0)).next() {
            to_zero.remove(&piece);
        }
    }
    unmapped
}

/// The pieces to clear (`State::to_zero`) as runs: neighbours and pieces inside another together.
fn zero_runs(pieces: BTreeSet<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (a, n) in pieces {
        match out.last_mut() {
            Some(l) if a <= l.0 + l.1 => l.1 = l.1.max(a + n - l.0),
            _ => out.push((a, n)),
        }
    }
    out
}

/// Free pool tiles partitioned by pool heap so `reuse_tile` packs the fullest live heap that
/// still has a free tile instead of scattering allocations across nearly-empty pools that are
/// draining towards `State::retire_free_pools`.
#[derive(Default)]
struct FreePools {
    by_heap: Vec<Vec<u32>>,
    total: usize,
}

impl FreePools {
    fn new() -> Self {
        Self::default()
    }

    fn len(&self) -> usize {
        self.total
    }

    fn is_empty(&self) -> bool {
        self.total == 0
    }

    fn heap_len(&self, h: usize) -> usize {
        self.by_heap.get(h).map_or(0, Vec::len)
    }

    fn has_empty_heap(&self) -> bool {
        self.by_heap.iter().any(|tiles| tiles.len() >= POOL_TILES as usize)
    }

    fn add_heap(&mut self, h: u32) {
        let idx = h as usize;
        if idx >= self.by_heap.len() {
            self.by_heap.resize_with(idx + 1, Vec::new);
        }
        debug_assert!(self.by_heap[idx].is_empty());
        self.by_heap[idx] = (1..POOL_TILES).rev().collect();
        self.total += (POOL_TILES - 1) as usize;
    }

    fn push(&mut self, (h, t): (u32, u32)) {
        let idx = h as usize;
        if idx >= self.by_heap.len() {
            self.by_heap.resize_with(idx + 1, Vec::new);
        }
        self.by_heap[idx].push(t);
        self.total += 1;
    }

    fn clear_heap(&mut self, h: usize) {
        if let Some(tiles) = self.by_heap.get_mut(h) {
            self.total -= tiles.len();
            tiles.clear();
            tiles.shrink_to_fit();
        }
    }

    /// Move every cooled tile (`>= COOL`) from `cooling` into its pool heap's free list.
    fn drain_cooling(&mut self, cooling: &mut VecDeque<((u32, u32), std::time::Instant)>, now: std::time::Instant) {
        while cooling.front().is_some_and(|&(_, at)| now.duration_since(at) >= COOL) {
            if let Some((tile, _)) = cooling.pop_front() {
                self.push(tile);
            }
        }
    }

    /// Take a tile from the fullest partially-used heap (`0 < len < POOL_TILES`, lowest heap index
    /// on ties) so nearly-empty heaps stay untouched and retire; only take from an all-free heap
    /// when no partially-used heap has a free tile.
    fn pop_packed(&mut self) -> Option<(u32, u32)> {
        if self.total == 0 {
            return None;
        }
        let h = self
            .by_heap
            .iter()
            .enumerate()
            .filter(|(_, tiles)| !tiles.is_empty() && tiles.len() < POOL_TILES as usize)
            .min_by_key(|(_, tiles)| tiles.len())
            .or_else(|| self.by_heap.iter().enumerate().find(|(_, tiles)| tiles.len() >= POOL_TILES as usize))
            .map(|(h, _)| h)?;
        let t = self.by_heap[h].pop()?;
        self.total -= 1;
        Some((h as u32, t))
    }

    #[cfg(test)]
    fn contains(&self, &(h, t): &(u32, u32)) -> bool {
        self.by_heap.get(h as usize).is_some_and(|tiles| tiles.contains(&t))
    }

    #[cfg(test)]
    fn clear(&mut self) {
        for tiles in &mut self.by_heap {
            tiles.clear();
        }
        self.total = 0;
    }

    #[cfg(test)]
    fn has_pool(&self, h: u32) -> bool {
        self.heap_len(h as usize) > 0
    }
}

/// Whether the oldest tile of `cooling` was freed `COOL` ago or more at `now`
fn cooled(cooling: &VecDeque<((u32, u32), std::time::Instant)>, now: std::time::Instant) -> bool {
    cooling.front().is_some_and(|&(_, at)| now.duration_since(at) >= COOL)
}

/// The pool tile to use next (`State::take_tile`): drain cooled tiles into their pool heaps and
/// take from the fullest partially-used heap (or an all-free heap if none is partially used).
/// None: neither, though tiles freed less than `COOL` ago may be cooling.
fn reuse_tile(free: &mut FreePools, cooling: &mut VecDeque<((u32, u32), std::time::Instant)>, now: std::time::Instant) -> Option<(u32, u32)> {
    free.drain_cooling(cooling, now);
    free.pop_packed()
}

/// The pool heaps (`pools` of them) all of whose tiles are in `free` or cooled in `cooling` at
/// `now` (`State::retire_free_pools`): a tile still cooling keeps its heap.
fn free_heaps(free: &FreePools, cooling: &VecDeque<((u32, u32), std::time::Instant)>, pools: usize, now: std::time::Instant) -> Vec<usize> {
    let mut count = vec![0u32; pools];
    for (h, c) in count.iter_mut().enumerate() {
        *c = free.heap_len(h) as u32;
    }
    for &((h, _), _) in cooling.iter().take_while(|&&(_, at)| now.duration_since(at) >= COOL) {
        if let Some(c) = count.get_mut(h as usize) {
            *c += 1;
        }
    }
    (0..pools).filter(|&h| count[h] == POOL_TILES).collect()
}

/// Whether a submit that freed no tile tries `State::retire_free_pools` at `now`: when a tile
/// of `cooling` cooled or a heap waits in `retired` (`waiting`), and `COOL` after the last try
/// (`tried`) at the earliest (a try counts the cooled tiles).
fn retire_due(cooling: &VecDeque<((u32, u32), std::time::Instant)>, waiting: bool, tried: std::time::Instant, now: std::time::Instant) -> bool {
    (waiting || cooled(cooling, now)) && now.duration_since(tried) >= COOL
}

fn retirement_reached(done: u64, value: u64) -> bool {
    done != u64::MAX && done >= value
}

fn next_retirement_value(value: u64) -> Option<u64> {
    value.checked_add(1).filter(|&next| next != u64::MAX)
}

fn retirement_marks_reached<F>(marks: &[(F, u64)], mut completed: impl FnMut(&F) -> u64) -> bool {
    !marks.is_empty() && marks.iter().all(|(f, v)| retirement_reached(completed(f), *v))
}

struct UploadSlot {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence_val: u64,
}

/// A copy back on `State::rbq`: its list, its readback buffer, the `rb_fence` value it waits
/// for, the `rbq` fence value that marks it done, and the copies into the title's memory
/// (window address, offset in `buf`, bytes, `TileEntry::gen` of the address's tile then:
/// `State::copy_back`).
struct RbJob {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    buf: ID3D12Resource,
    ptr: *mut u8,
    cap: usize,
    rb_v: u64,
    done_v: u64,
    copies: Vec<(usize, usize, usize, u32)>,
    /// the type of queue `list` is for (a copy back runs on the title's own queue)
    ty: D3D12_COMMAND_LIST_TYPE,
    /// the title queue's fences for it (`State::rb_queue_fences`): the first reaches `rb_v` when
    /// the lists it follows are done, the second `done_v` when the copy is. None: `rb_fence` and
    /// the readback queue's fence (the form before)
    fences: Option<(ID3D12Fence, ID3D12Fence)>,
    /// place among the submissions' jobs (`ReadbackQueue::job_seq`): copies and immediates land in it
    seq: u64,
}

// D3D12 objects and mapped readback staging pointers are free-threaded; mutated under mutexes.
unsafe impl Send for RbJob {}

/// In an immediate's byte count: the cell is a counter that the title's queues write in turn,
/// with values that only rise as they are recorded. A list that runs late – a retail title's
/// compute jobs, tens of frames after their recording when its submits are slow – writes an
/// earlier value over a later one; the title's CPU waits for "at least" the later one, which
/// was reached. The CPU's view keeps the larger value.
pub(super) const IMM_KEEP_MAX: u8 = 0x80;

/// Immediates the title's lists wrote into a window (`list_x::imm_by_cpu`): stored in the
/// title's memory once `fence` reaches `value` (the lists are done on the GPU).
struct ImmJob {
    /// place among the copies back (`RbJob::seq`)
    seq: u64,
    fence: ID3D12Fence,
    value: u64,
    /// (address, value, bytes, `TileEntry::gen` of the address's tile then)
    writes: Vec<(usize, u64, u8, u32)>,
}

unsafe impl Send for ImmJob {}

/// Async readback job queues and buffer pool under a pure leaf lock separate from `STATE`.
///
/// Strict 2-level DAG lock order (mathematically deadlock-free):
/// - `STATE` -> `ReadbackQueue` (mark `submitting` on plan prep, pop finished jobs, return recycled
///   `RbJob`s up to cap 32)
/// - `RbSubmit` -> `ReadbackQueue` (pop pooled `RbJob`, push newly submitted jobs & clear `submitting`)
/// - `STATE` and `RbSubmit` never nest; `ReadbackQueue` is never held across GPU fence spins,
///   `Condvar` waits (released while waiting), D3D12 COM calls, or `RbJob` drops.
struct ReadbackQueue {
    /// fallback fence signalled on title queues (when a job has `fences: None`)
    rb_fence: ID3D12Fence,
    /// fallback readback queue fence (when a job has `fences: None`)
    rbq_done: Option<ID3D12Fence>,
    /// submitted copies back, oldest first, and finished ones to reuse
    rb_jobs: VecDeque<RbJob>,
    rb_pool: Vec<RbJob>,
    /// immediates of the title's submissions, oldest first, and the count that orders them
    /// with `rb_jobs`
    imm_jobs: VecDeque<ImmJob>,
    job_seq: u64,
    /// count of async readback plans prepared under `STATE` whose jobs have not yet been
    /// published (or cancelled on error) into `rb_jobs`/`imm_jobs`
    submitting: usize,
    /// count of finished jobs currently popped and being landed into CPU memory under `STATE`
    /// (forces concurrent `settle_readbacks*` callers to synchronize on `STATE` before returning)
    landing: usize,
}

unsafe impl Send for ReadbackQueue {}

impl ReadbackQueue {
    fn new(rb_fence: ID3D12Fence, rbq_done: Option<ID3D12Fence>) -> Self {
        Self { rb_fence, rbq_done, rb_jobs: VecDeque::new(), rb_pool: Vec::new(), imm_jobs: VecDeque::new(), job_seq: 0, submitting: 0, landing: 0 }
    }
}

impl Drop for ReadbackQueue {
    fn drop(&mut self) {
        let left = self.rb_jobs.len() + self.imm_jobs.len();
        if left > 0 {
            RB_PENDING.fetch_sub(left, Ordering::Release);
        }
    }
}

/// Shared `ReadbackQueue` plus a `Condvar` notified whenever an in-flight `prepare_after_execute`
/// plan finishes publishing or cancels, so settling callers observe the transition within their
/// existing wait budget without polling or holding `STATE`.
struct SharedReadbacks {
    queue: Mutex<ReadbackQueue>,
    published: std::sync::Condvar,
}

impl SharedReadbacks {
    fn new(rb_fence: ID3D12Fence, rbq_done: Option<ID3D12Fence>) -> Self {
        Self {
            queue: Mutex::new(ReadbackQueue::new(rb_fence, rbq_done)),
            published: std::sync::Condvar::new(),
        }
    }

    fn lock(&self) -> std::sync::LockResult<std::sync::MutexGuard<'_, ReadbackQueue>> {
        self.queue.lock()
    }
}

/// Tracks one prepared `AsyncReadbackPlan` from `prepare_after_execute` (while `STATE` is still
/// held) until `submit_async_readback` either publishes its jobs or bails/drops. Holds a strong
/// `Arc<SharedReadbacks>` so `ReadbackQueue::drop` cannot run while a submission is in flight.
struct SubmittingGuard {
    readbacks: Arc<SharedReadbacks>,
    active: bool,
}

impl SubmittingGuard {
    fn begin(readbacks: &Arc<SharedReadbacks>) -> Self {
        let mut rq = readbacks.lock().unwrap();
        rq.submitting += 1;
        RB_PENDING.fetch_add(1, Ordering::Release);
        Self { readbacks: Arc::clone(readbacks), active: true }
    }

    /// Atomically replace the pending submission token with `added` queued jobs under `rq`.
    fn finish(mut self, rq: &mut ReadbackQueue, added: usize) {
        rq.submitting -= 1;
        self.active = false;
        if added > 1 {
            RB_PENDING.fetch_add(added - 1, Ordering::Release);
        } else if added == 0 {
            RB_PENDING.fetch_sub(1, Ordering::Release);
        }
    }
}

impl Drop for SubmittingGuard {
    fn drop(&mut self) {
        if self.active {
            let mut rq = self.readbacks.lock().unwrap();
            rq.submitting -= 1;
            RB_PENDING.fetch_sub(1, Ordering::Release);
            drop(rq);
            self.readbacks.published.notify_all();
        }
    }
}

/// Per-queue async readback submission state, locked outside `STATE` so `readback_after_execute`
/// records `CopyBufferRegion` / `pc_execute` / `Signal` without blocking `STATE`.
struct RbSubmit {
    dev: ID3D12Device,
    /// title queue -> the two fences it signals around a copy back, and the last value. One set
    /// a queue: with one fence for all, a queue that got to its Signal first let the copies of a
    /// queue still waiting run early
    rb_queue_fences: FxHashMap<usize, (ID3D12Fence, ID3D12Fence, u64)>,
}

unsafe impl Send for RbSubmit {}

impl RbSubmit {
    fn new(dev: ID3D12Device) -> Self {
        Self { dev, rb_queue_fences: FxHashMap::default() }
    }
}

enum ReadyJob {
    Imm(ImmJob),
    Rb(RbJob),
}

struct RbChunk {
    win: usize,
    va: usize,
    len: usize,
    gen: u32,
}

struct AsyncReadbackPlan {
    rb_submit: Arc<Mutex<RbSubmit>>,
    readbacks: Arc<SharedReadbacks>,
    windows: Vec<(usize, ID3D12Resource)>,
    batches: Vec<Vec<RbChunk>>,
    imm: Vec<(usize, u64, u8, u32)>,
    submitting: SubmittingGuard,
}

/// How far `drain_readbacks` goes
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Drain {
    /// only copies the GPU has finished
    #[cfg(test)]
    Finished,
    /// also those whose title work is done (waiting for the copy itself): before a CPU read
    /// the title made sure of, e.g. after its wait for the value a list wrote
    TitleDone,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SubmitResult {
    Submitted,
    NotSubmitted,
    Unordered,
}

enum UploadStep {
    Done(PreparationOutcome),
    Wait(WaitTicket),
}

enum BeginSlot {
    Ready(usize),
    Wait(WaitTicket),
    RetryRequired,
}

#[derive(Clone, Copy)]
enum WaitKind {
    Slot { index: usize, next_slot: usize, fence: u64 },
    Ring { front: (u64, u64), write: u64, submitted: u64, free: u64 },
}

struct WaitTicket {
    state_generation: u64,
    mapping_epoch: u64,
    dev: ID3D12Device,
    fence: ID3D12Fence,
    target: u64,
    kind: WaitKind,
}

/// Sources which can require one CPU page to be considered by a future immutable upload plan.
/// Multiple sources collapse into one page owner without losing why the page must be revisited.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct WorkReasons(u8);

impl WorkReasons {
    const ZERO: Self = Self(1);
    const RETRY: Self = Self(2);
    const DIRTY: Self = Self(4);

    fn insert(&mut self, other: Self) { self.0 |= other.0; }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct WorkClaimToken {
    state_generation: u64,
    serial: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkClaimResolution {
    Committed,
    Restored,
    AlreadyFinished,
    StaleGeneration,
}

struct ClaimedWork {
    token: WorkClaimToken,
    pages: Vec<(usize, WorkReasons)>,
}

/// Inert foundation for splitting sparse planning from its serialized commit phase. Work added
/// while a page is owned stays journalled for a later claim; cancelling an older claim merges its
/// exact reasons back without rewinding or duplicating a newer claim. Tokens include the sparse
/// state generation, so a late completion from replaced state cannot modify the replacement.
#[allow(dead_code)]
struct WorkClaimLedger {
    state_generation: u64,
    next_serial: u64,
    available: BTreeMap<usize, WorkReasons>,
    owners: BTreeMap<usize, u64>,
    claims: BTreeMap<u64, Vec<(usize, WorkReasons)>>,
}

#[allow(dead_code)]
impl WorkClaimLedger {
    fn new(state_generation: u64) -> Self {
        Self {
            state_generation,
            next_serial: 0,
            available: BTreeMap::new(),
            owners: BTreeMap::new(),
            claims: BTreeMap::new(),
        }
    }

    fn add_pages(&mut self, pages: impl IntoIterator<Item = usize>, reason: WorkReasons) {
        for page in pages {
            self.available.entry(page & !(PAGE - 1)).or_default().insert(reason);
        }
    }

    fn add_zero_ranges(&mut self, ranges: &[(usize, usize)]) {
        for &(start, bytes) in ranges {
            if bytes == 0 {
                continue;
            }
            let end = start.saturating_add(bytes);
            let mut page = start & !(PAGE - 1);
            while page < end {
                self.available.entry(page).or_default().insert(WorkReasons::ZERO);
                let Some(next) = page.checked_add(PAGE) else { break };
                page = next;
            }
        }
    }

    fn add_retry_pages(&mut self, pages: impl IntoIterator<Item = usize>) {
        self.add_pages(pages, WorkReasons::RETRY);
    }

    fn add_dirty_pages(&mut self, pages: impl IntoIterator<Item = usize>) {
        self.add_pages(pages, WorkReasons::DIRTY);
    }

    /// Claim at most `limit` pages in address order. A page has one owner even when zero, retry,
    /// and fresh dirty work overlap; reasons added after this snapshot remain for the next claim.
    fn claim(&mut self, limit: usize) -> Option<ClaimedWork> {
        if limit == 0 {
            return None;
        }
        let pages: Vec<_> = self.available
            .iter()
            .filter(|(page, _)| !self.owners.contains_key(page))
            .take(limit)
            .map(|(&page, &reasons)| (page, reasons))
            .collect();
        if pages.is_empty() {
            return None;
        }
        self.next_serial = self.next_serial.checked_add(1).expect("sparse work claim serial exhausted");
        let serial = self.next_serial;
        for &(page, _) in &pages {
            self.available.remove(&page);
            self.owners.insert(page, serial);
        }
        self.claims.insert(serial, pages.clone());
        Some(ClaimedWork {
            token: WorkClaimToken { state_generation: self.state_generation, serial },
            pages,
        })
    }

    fn commit(&mut self, token: WorkClaimToken) -> WorkClaimResolution {
        self.finish(token, false)
    }

    fn restore(&mut self, token: WorkClaimToken) -> WorkClaimResolution {
        self.finish(token, true)
    }

    fn cancel(&mut self, token: WorkClaimToken) -> WorkClaimResolution {
        self.restore(token)
    }

    fn finish(&mut self, token: WorkClaimToken, restore: bool) -> WorkClaimResolution {
        if token.state_generation != self.state_generation {
            return WorkClaimResolution::StaleGeneration;
        }
        let Some(pages) = self.claims.remove(&token.serial) else {
            return WorkClaimResolution::AlreadyFinished;
        };
        for (page, reasons) in pages {
            if self.owners.get(&page) != Some(&token.serial) {
                continue;
            }
            self.owners.remove(&page);
            if restore {
                self.available.entry(page).or_default().insert(reasons);
            }
        }
        if restore { WorkClaimResolution::Restored } else { WorkClaimResolution::Committed }
    }

    fn replace_generation(&mut self, state_generation: u64) {
        self.state_generation = state_generation;
        self.available.clear();
        self.owners.clear();
        self.claims.clear();
    }
}

/// Identity of one future sparse batch while it crosses planning and queue acceptance. `lease`
/// changes on takeover, so an old owner cannot publish or discard the new owner's batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct ClaimFrontierToken {
    state_generation: u64,
    epoch: u64,
    lease: u64,
    owner: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClaimFrontierState {
    Pending,
    Accepted,
    Receipted(u64),
    Obsolete,
}

#[allow(dead_code)]
struct ClaimFrontierRecord {
    lease: u64,
    owner: u64,
    state: ClaimFrontierState,
    prefix_fence: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum ClaimFrontierAction {
    Ready { through: u64, required_fence: Option<u64> },
    Progress(ClaimFrontierToken),
    TakeOver(ClaimFrontierToken),
    WaitForReceipt(ClaimFrontierToken),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum ClaimFrontierError {
    StaleGeneration,
    UnknownEpoch,
    StaleToken,
    WrongState,
    Fence,
    Exhausted,
}

/// Inert coordination model for future asynchronous sparse planning. Every observed epoch remains
/// visible until it is receipted or explicitly consumed as obsolete; a foreground queue therefore
/// cannot mistake work owned by another planner for an empty journal. There are no callers yet.
#[allow(dead_code)]
struct ClaimFrontier {
    state_generation: u64,
    next_epoch: u64,
    next_lease: u64,
    retired_through: u64,
    records: BTreeMap<u64, ClaimFrontierRecord>,
}

#[allow(dead_code)]
impl ClaimFrontier {
    fn new(state_generation: u64) -> Self {
        Self {
            state_generation,
            next_epoch: 0,
            next_lease: 0,
            retired_through: 0,
            records: BTreeMap::new(),
        }
    }

    fn claim(&mut self, owner: u64) -> Result<ClaimFrontierToken, ClaimFrontierError> {
        let epoch = self.next_epoch.checked_add(1).ok_or(ClaimFrontierError::Exhausted)?;
        let lease = self.next_lease.checked_add(1).ok_or(ClaimFrontierError::Exhausted)?;
        let token = ClaimFrontierToken { state_generation: self.state_generation, epoch, lease, owner };
        self.records.insert(epoch, ClaimFrontierRecord {
            lease,
            owner,
            state: ClaimFrontierState::Pending,
            prefix_fence: None,
        });
        self.next_epoch = epoch;
        self.next_lease = lease;
        Ok(token)
    }

    fn observe(
        &self, state_generation: u64, through: u64, requester: u64,
    ) -> Result<ClaimFrontierAction, ClaimFrontierError> {
        if state_generation != self.state_generation {
            return Err(ClaimFrontierError::StaleGeneration);
        }
        if through > self.next_epoch {
            return Err(ClaimFrontierError::UnknownEpoch);
        }
        if through <= self.retired_through {
            let required_fence = (through != 0)
                .then(|| self.records.get(&through).and_then(|record| record.prefix_fence))
                .flatten();
            return Ok(ClaimFrontierAction::Ready { through, required_fence });
        }
        let epoch = self.retired_through.checked_add(1).ok_or(ClaimFrontierError::Exhausted)?;
        let record = self.records.get(&epoch).ok_or(ClaimFrontierError::UnknownEpoch)?;
        let token = ClaimFrontierToken {
            state_generation: self.state_generation,
            epoch,
            lease: record.lease,
            owner: record.owner,
        };
        Ok(match record.state {
            ClaimFrontierState::Pending if record.owner == requester => ClaimFrontierAction::Progress(token),
            ClaimFrontierState::Pending => ClaimFrontierAction::TakeOver(token),
            ClaimFrontierState::Accepted => ClaimFrontierAction::WaitForReceipt(token),
            ClaimFrontierState::Receipted(_) | ClaimFrontierState::Obsolete => unreachable!("terminal frontier did not advance"),
        })
    }

    fn take_over(
        &mut self, token: ClaimFrontierToken, owner: u64,
    ) -> Result<ClaimFrontierToken, ClaimFrontierError> {
        self.validate(token, ClaimFrontierState::Pending)?;
        let lease = self.next_lease.checked_add(1).ok_or(ClaimFrontierError::Exhausted)?;
        let record = self.records.get_mut(&token.epoch).unwrap();
        record.lease = lease;
        record.owner = owner;
        self.next_lease = lease;
        Ok(ClaimFrontierToken { state_generation: self.state_generation, epoch: token.epoch, lease, owner })
    }

    fn accept(&mut self, token: ClaimFrontierToken) -> Result<(), ClaimFrontierError> {
        self.validate(token, ClaimFrontierState::Pending)?;
        self.records.get_mut(&token.epoch).unwrap().state = ClaimFrontierState::Accepted;
        Ok(())
    }

    fn receipt(&mut self, token: ClaimFrontierToken, fence: u64) -> Result<(), ClaimFrontierError> {
        self.validate(token, ClaimFrontierState::Accepted)?;
        if fence == 0 {
            return Err(ClaimFrontierError::Fence);
        }
        self.records.get_mut(&token.epoch).unwrap().state = ClaimFrontierState::Receipted(fence);
        self.advance();
        Ok(())
    }

    fn consume_obsolete(&mut self, token: ClaimFrontierToken) -> Result<(), ClaimFrontierError> {
        self.validate(token, ClaimFrontierState::Pending)?;
        self.records.get_mut(&token.epoch).unwrap().state = ClaimFrontierState::Obsolete;
        self.advance();
        Ok(())
    }

    fn validate(
        &self, token: ClaimFrontierToken, expected: ClaimFrontierState,
    ) -> Result<(), ClaimFrontierError> {
        if token.state_generation != self.state_generation {
            return Err(ClaimFrontierError::StaleGeneration);
        }
        let record = self.records.get(&token.epoch).ok_or(ClaimFrontierError::StaleToken)?;
        if record.lease != token.lease || record.owner != token.owner {
            return Err(ClaimFrontierError::StaleToken);
        }
        if record.state != expected {
            return Err(ClaimFrontierError::WrongState);
        }
        Ok(())
    }

    fn advance(&mut self) {
        let mut prefix_fence = if self.retired_through == 0 {
            0
        } else {
            self.records[&self.retired_through].prefix_fence.unwrap_or(0)
        };
        loop {
            let Some(next) = self.retired_through.checked_add(1) else { return };
            let Some(record) = self.records.get_mut(&next) else { return };
            match record.state {
                ClaimFrontierState::Receipted(fence) => prefix_fence = prefix_fence.max(fence),
                ClaimFrontierState::Obsolete => {}
                ClaimFrontierState::Pending | ClaimFrontierState::Accepted => return,
            }
            record.prefix_fence = (prefix_fence != 0).then_some(prefix_fence);
            self.retired_through = next;
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum SparseBatchError {
    StaleGeneration,
    Fence,
    Poisoned,
    Unsealed,
    Exhausted,
    Corrupt,
    Frontier(ClaimFrontierError),
}

/// One atomic owner for work claims and their visibility frontier. This remains inert: no live
/// sparse state, command list, queue or fence uses it yet. Keeping both ledgers under one mutex is
/// what makes a lease rotation revoke the old planner without restoring or duplicating its work.
#[allow(dead_code)]
struct SparseBatchLedger {
    work: WorkClaimLedger,
    frontier: ClaimFrontier,
    fence: Option<SparseFenceIdentity>,
    bindings: BTreeMap<u64, WorkClaimToken>,
    poisoned: bool,
}

#[allow(dead_code)]
impl SparseBatchLedger {
    fn new(state_generation: u64) -> Self {
        Self {
            work: WorkClaimLedger::new(state_generation),
            frontier: ClaimFrontier::new(state_generation),
            fence: None,
            bindings: BTreeMap::new(),
            poisoned: false,
        }
    }

    fn check(&self, state_generation: u64) -> Result<(), SparseBatchError> {
        if state_generation != self.work.state_generation {
            return Err(SparseBatchError::StaleGeneration);
        }
        if self.poisoned {
            return Err(SparseBatchError::Poisoned);
        }
        Ok(())
    }

    fn seal(
        &mut self, state_generation: u64, owner: u64, limit: usize,
    ) -> Result<Option<(ClaimFrontierToken, Vec<(usize, WorkReasons)>)>, SparseBatchError> {
        self.check(state_generation)?;
        if self.work.next_serial == u64::MAX {
            return Err(SparseBatchError::Exhausted);
        }
        let Some(claimed) = self.work.claim(limit) else { return Ok(None) };
        let token = match self.frontier.claim(owner) {
            Ok(token) => token,
            Err(error) => {
                let _ = self.work.restore(claimed.token);
                return Err(SparseBatchError::Frontier(error));
            }
        };
        self.bindings.insert(token.epoch, claimed.token);
        Ok(Some((token, claimed.pages)))
    }

    fn pages(&self, token: ClaimFrontierToken) -> Result<Vec<(usize, WorkReasons)>, SparseBatchError> {
        self.check(token.state_generation)?;
        self.frontier.validate(token, ClaimFrontierState::Pending).map_err(SparseBatchError::Frontier)?;
        let work = self.bindings.get(&token.epoch).ok_or(SparseBatchError::Corrupt)?;
        self.work.claims.get(&work.serial).cloned().ok_or(SparseBatchError::Corrupt)
    }

    fn take_over(
        &mut self, token: ClaimFrontierToken, owner: u64,
    ) -> Result<(ClaimFrontierToken, Vec<(usize, WorkReasons)>), SparseBatchError> {
        let pages = self.pages(token)?;
        let token = self.frontier.take_over(token, owner).map_err(SparseBatchError::Frontier)?;
        Ok((token, pages))
    }

    fn observe(
        &self, state_generation: u64, through: u64, requester: u64,
    ) -> Result<ClaimFrontierAction, SparseBatchError> {
        self.check(state_generation)?;
        if !self.work.available.is_empty() {
            return Err(SparseBatchError::Unsealed);
        }
        self.frontier.observe(state_generation, through, requester).map_err(SparseBatchError::Frontier)
    }

    fn through(&self, state_generation: u64) -> Result<u64, SparseBatchError> {
        self.check(state_generation)?;
        if !self.work.available.is_empty() {
            return Err(SparseBatchError::Unsealed);
        }
        Ok(self.frontier.next_epoch)
    }

    fn accept(&mut self, token: ClaimFrontierToken) -> Result<WorkClaimToken, SparseBatchError> {
        self.check(token.state_generation)?;
        let work = self.binding(token, ClaimFrontierState::Pending)?;
        self.frontier.accept(token).map_err(SparseBatchError::Frontier)?;
        Ok(work)
    }

    fn receipt(&mut self, token: ClaimFrontierToken, fence: u64) -> Result<(), SparseBatchError> {
        self.check(token.state_generation)?;
        if fence == 0 {
            return Err(SparseBatchError::Frontier(ClaimFrontierError::Fence));
        }
        let work = self.binding(token, ClaimFrontierState::Accepted)?;
        if self.work.commit(work) != WorkClaimResolution::Committed {
            self.poisoned = true;
            return Err(SparseBatchError::Corrupt);
        }
        if let Err(error) = self.frontier.receipt(token, fence) {
            self.poisoned = true;
            return Err(SparseBatchError::Frontier(error));
        }
        self.bindings.remove(&token.epoch);
        Ok(())
    }

    fn obsolete(&mut self, token: ClaimFrontierToken) -> Result<(), SparseBatchError> {
        self.check(token.state_generation)?;
        let work = self.binding(token, ClaimFrontierState::Pending)?;
        if self.work.commit(work) != WorkClaimResolution::Committed {
            self.poisoned = true;
            return Err(SparseBatchError::Corrupt);
        }
        if let Err(error) = self.frontier.consume_obsolete(token) {
            self.poisoned = true;
            return Err(SparseBatchError::Frontier(error));
        }
        self.bindings.remove(&token.epoch);
        Ok(())
    }

    fn retry(&mut self, token: ClaimFrontierToken) -> Result<(), SparseBatchError> {
        self.check(token.state_generation)?;
        let work = self.binding(token, ClaimFrontierState::Pending)?;
        if self.work.restore(work) != WorkClaimResolution::Restored {
            self.poisoned = true;
            return Err(SparseBatchError::Corrupt);
        }
        if let Err(error) = self.frontier.consume_obsolete(token) {
            self.poisoned = true;
            return Err(SparseBatchError::Frontier(error));
        }
        self.bindings.remove(&token.epoch);
        Ok(())
    }

    fn poison_exact(
        &mut self, token: ClaimFrontierToken, work: WorkClaimToken,
    ) -> Result<(), SparseBatchError> {
        if token.state_generation != self.work.state_generation || work.state_generation != token.state_generation {
            return Err(SparseBatchError::StaleGeneration);
        }
        let resolution = self.work.commit(work);
        if !matches!(resolution, WorkClaimResolution::Committed | WorkClaimResolution::AlreadyFinished) {
            return Err(SparseBatchError::Corrupt);
        }
        if self.bindings.get(&token.epoch) == Some(&work) { self.bindings.remove(&token.epoch); }
        self.poisoned = true;
        Ok(())
    }

    fn binding(
        &self, token: ClaimFrontierToken, state: ClaimFrontierState,
    ) -> Result<WorkClaimToken, SparseBatchError> {
        self.frontier.validate(token, state).map_err(SparseBatchError::Frontier)?;
        let work = *self.bindings.get(&token.epoch).ok_or(SparseBatchError::Corrupt)?;
        self.work.claims.contains_key(&work.serial).then_some(work).ok_or(SparseBatchError::Corrupt)
    }

    fn replace_generation(
        &mut self, current: u64, replacement: u64, fence_allocation: Option<u64>,
    ) -> Result<(), SparseBatchError> {
        if current != self.work.state_generation {
            return Err(SparseBatchError::StaleGeneration);
        }
        // Generations are identities, not labels: reuse could resurrect an old epoch/lease tuple.
        if replacement <= current {
            return Err(SparseBatchError::StaleGeneration);
        }
        self.fence = match (self.fence, fence_allocation) {
            (None, None) => None,
            (Some(fence), Some(allocation)) if allocation > fence.allocation => {
                Some(SparseFenceIdentity { state_generation: replacement, allocation })
            }
            _ => return Err(SparseBatchError::Fence),
        };
        self.work.replace_generation(replacement);
        self.frontier = ClaimFrontier::new(replacement);
        self.bindings.clear();
        self.poisoned = false;
        Ok(())
    }
}

#[derive(Clone)]
#[allow(dead_code)]
struct SparseBatches {
    ledger: Arc<Mutex<SparseBatchLedger>>,
    state_generation: u64,
}

#[allow(dead_code)]
impl SparseBatches {
    fn new(state_generation: u64) -> Self {
        Self { ledger: Arc::new(Mutex::new(SparseBatchLedger::new(state_generation))), state_generation }
    }

    fn with_fence(state_generation: u64, fence_allocation: u64) -> Option<Self> {
        if state_generation == 0 || fence_allocation == 0 { return None }
        let batches = Self::new(state_generation);
        batches.ledger.lock().unwrap().fence = Some(SparseFenceIdentity { state_generation, allocation: fence_allocation });
        Some(batches)
    }

    fn add_pages(&self, pages: impl IntoIterator<Item = usize>, reason: WorkReasons) -> bool {
        // Iterator code is caller code: finish it before taking the non-reentrant ledger mutex.
        let mut pages: Vec<_> = pages.into_iter().map(|page| page & !(PAGE - 1)).collect();
        pages.sort_unstable();
        pages.dedup();
        let mut ledger = self.ledger.lock().unwrap();
        if ledger.check(self.state_generation).is_err() { return false }
        ledger.work.add_pages(pages, reason);
        true
    }

    fn add_dirty_pages(&self, pages: impl IntoIterator<Item = usize>) -> bool {
        self.add_pages(pages, WorkReasons::DIRTY)
    }

    fn add_retry_pages(&self, pages: impl IntoIterator<Item = usize>) -> bool {
        self.add_pages(pages, WorkReasons::RETRY)
    }

    fn add_zero_ranges(&self, ranges: &[(usize, usize)]) -> bool {
        let mut ledger = self.ledger.lock().unwrap();
        if ledger.check(self.state_generation).is_err() { return false }
        ledger.work.add_zero_ranges(ranges);
        true
    }

    /// Mirror one legacy upload attempt into an exact-generation batch. All three reason sets are
    /// added and claimed under one ledger lock so a future unlocked planner cannot observe only a
    /// prefix of the legacy journal.
    fn seal_legacy(
        &self,
        owner: u64,
        zero: &[(usize, usize)],
        retry: &[usize],
        dirty: &[(usize, usize)],
    ) -> Result<Option<SparseBatchLease>, SparseBatchError> {
        let mut dirty_pages = Vec::new();
        for &(start, bytes) in dirty {
            let end = start.saturating_add(bytes);
            let mut page = start & !(PAGE - 1);
            while page < end {
                dirty_pages.push(page);
                let Some(next) = page.checked_add(PAGE) else { break };
                page = next;
            }
        }
        dirty_pages.sort_unstable();
        dirty_pages.dedup();

        let mut ledger = self.ledger.lock().unwrap();
        ledger.check(self.state_generation)?;
        ledger.work.add_zero_ranges(zero);
        ledger.work.add_retry_pages(retry.iter().copied());
        ledger.work.add_dirty_pages(dirty_pages);
        let result = ledger.seal(self.state_generation, owner, usize::MAX)?;
        Ok(result.map(|(token, pages)| SparseBatchLease { ledger: self.ledger.clone(), token, pages }))
    }

    fn seal(&self, owner: u64, limit: usize) -> Result<Option<SparseBatchLease>, SparseBatchError> {
        let result = self.ledger.lock().unwrap().seal(self.state_generation, owner, limit)?;
        Ok(result.map(|(token, pages)| SparseBatchLease { ledger: self.ledger.clone(), token, pages }))
    }

    fn observe(&self, through: u64, requester: u64) -> Result<ClaimFrontierAction, SparseBatchError> {
        self.ledger.lock().unwrap().observe(self.state_generation, through, requester)
    }

    fn ready_receipt(&self, through: u64, requester: u64) -> Result<SparseReadyReceipt, ForegroundWaitError> {
        let ledger = self.ledger.lock().unwrap();
        let fence = ledger.fence.ok_or(ForegroundWaitError::Fence)?;
        match ledger.observe(self.state_generation, through, requester).map_err(ForegroundWaitError::Batch)? {
            ClaimFrontierAction::Ready { through, required_fence } => Ok(SparseReadyReceipt {
                state_generation: self.state_generation, fence, through, required_fence,
            }),
            _ => Err(ForegroundWaitError::NotReady),
        }
    }

    fn through(&self) -> Result<u64, SparseBatchError> {
        self.ledger.lock().unwrap().through(self.state_generation)
    }

    fn take_over(
        &self, token: ClaimFrontierToken, owner: u64,
    ) -> Result<SparseBatchLease, SparseBatchError> {
        let (token, pages) = self.ledger.lock().unwrap().take_over(token, owner)?;
        Ok(SparseBatchLease { ledger: self.ledger.clone(), token, pages })
    }

    fn replace_generation(&self, replacement: u64) -> Result<Self, SparseBatchError> {
        self.ledger.lock().unwrap().replace_generation(self.state_generation, replacement, None)?;
        Ok(Self { ledger: self.ledger.clone(), state_generation: replacement })
    }

    fn replace_generation_with_fence(
        &self, replacement: u64, fence_allocation: u64,
    ) -> Result<Self, SparseBatchError> {
        self.ledger.lock().unwrap().replace_generation(
            self.state_generation, replacement, Some(fence_allocation),
        )?;
        Ok(Self { ledger: self.ledger.clone(), state_generation: replacement })
    }
}

#[allow(dead_code)]
struct SparseBatchLease {
    ledger: Arc<Mutex<SparseBatchLedger>>,
    token: ClaimFrontierToken,
    pages: Vec<(usize, WorkReasons)>,
}

#[allow(dead_code)]
impl SparseBatchLease {
    fn pages(&self) -> &[(usize, WorkReasons)] { &self.pages }

    fn take_over(&self, owner: u64) -> Result<Self, SparseBatchError> {
        let (token, pages) = self.ledger.lock().unwrap().take_over(self.token, owner)?;
        Ok(Self { ledger: self.ledger.clone(), token, pages })
    }

    fn accept(self) -> Result<AcceptedSparseBatch, SparseBatchError> {
        let work = self.ledger.lock().unwrap().accept(self.token)?;
        Ok(AcceptedSparseBatch { ledger: self.ledger.clone(), token: Some((self.token, work)) })
    }

    fn obsolete(self) -> Result<(), SparseBatchError> {
        self.ledger.lock().unwrap().obsolete(self.token)
    }

    fn retry(self) -> Result<(), SparseBatchError> {
        self.ledger.lock().unwrap().retry(self.token)
    }
}

/// Accepted work is fail-closed: losing its owner consumes the claim and poisons the generation.
#[allow(dead_code)]
struct AcceptedSparseBatch {
    ledger: Arc<Mutex<SparseBatchLedger>>,
    token: Option<(ClaimFrontierToken, WorkClaimToken)>,
}

#[allow(dead_code)]
impl AcceptedSparseBatch {
    fn receipt(mut self, fence: u64) -> Result<(), SparseBatchError> {
        let result = self.ledger.lock().unwrap().receipt(self.token.unwrap().0, fence);
        if result.is_ok() { self.token.take(); }
        result
    }
}

impl Drop for AcceptedSparseBatch {
    fn drop(&mut self) {
        if let Some((token, work)) = self.token.take() {
            let _ = self.ledger.lock().unwrap().poison_exact(token, work);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[allow(dead_code)]
struct SparseFenceIdentity {
    state_generation: u64, allocation: u64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[allow(dead_code)]
struct TitleQueueIdentity {
    raw: usize, generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct SparseReadyReceipt {
    state_generation: u64, fence: SparseFenceIdentity, through: u64, required_fence: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct ForegroundWaitToken {
    state_generation: u64, fence: SparseFenceIdentity, queue: TitleQueueIdentity, serial: u64, value: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ForegroundWaitState { Reserved, Accepted }

struct ForegroundWaitRecord {
    fence: SparseFenceIdentity, installed: u64,
    pending: Option<(ForegroundWaitToken, ForegroundWaitState)>, uncertain: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum ForegroundWaitError {
    StaleGeneration, Fence, Queue, NotReady, Pending(ForegroundWaitToken),
    Uncertain, StaleToken, Exhausted,
    Batch(SparseBatchError),
}

#[allow(dead_code)]
enum ForegroundWaitDecision {
    NoWait { through: u64 }, Installed { through: u64, value: u64 },
    Reserve(ForegroundWaitReservation),
}

#[allow(dead_code)]
struct ForegroundWaitLedger {
    state_generation: u64, fence: SparseFenceIdentity,
    next_queue_generation: u64, next_serial: u64,
    queues: BTreeMap<usize, u64>,
    waits: BTreeMap<TitleQueueIdentity, ForegroundWaitRecord>,
}

#[allow(dead_code)]
impl ForegroundWaitLedger {
    fn new(state_generation: u64, fence_allocation: u64) -> Option<Self> {
        if state_generation == 0 || fence_allocation == 0 { return None }
        Some(Self {
            state_generation,
            fence: SparseFenceIdentity { state_generation, allocation: fence_allocation },
            next_queue_generation: 0,
            next_serial: 0,
            queues: BTreeMap::new(),
            waits: BTreeMap::new(),
        })
    }

    fn register_queue(&mut self, raw: usize) -> Result<TitleQueueIdentity, ForegroundWaitError> {
        if raw == 0 { return Err(ForegroundWaitError::Queue) }
        if let Some(&generation) = self.queues.get(&raw) {
            return Ok(TitleQueueIdentity { raw, generation });
        }
        let generation = self.next_queue_generation.checked_add(1).ok_or(ForegroundWaitError::Exhausted)?;
        self.next_queue_generation = generation;
        self.queues.insert(raw, generation);
        Ok(TitleQueueIdentity { raw, generation })
    }

    fn retire_queue(&mut self, queue: TitleQueueIdentity) -> Result<(), ForegroundWaitError> {
        if self.queues.get(&queue.raw) != Some(&queue.generation) { return Err(ForegroundWaitError::Queue) }
        self.queues.remove(&queue.raw);
        self.waits.remove(&queue);
        Ok(())
    }

    fn reserve(
        &mut self, queue: TitleQueueIdentity, ready: SparseReadyReceipt,
    ) -> Result<ForegroundWaitDecision, ForegroundWaitError> {
        if ready.state_generation != self.state_generation { return Err(ForegroundWaitError::StaleGeneration) }
        if ready.fence != self.fence { return Err(ForegroundWaitError::Fence) }
        if self.queues.get(&queue.raw) != Some(&queue.generation) { return Err(ForegroundWaitError::Queue) }
        if let Some(record) = self.waits.get(&queue) {
            if record.fence != ready.fence { return Err(ForegroundWaitError::Fence) }
            if record.uncertain { return Err(ForegroundWaitError::Uncertain) }
            if let Some((token, _)) = record.pending { return Err(ForegroundWaitError::Pending(token)) }
        }
        let Some(value) = ready.required_fence else { return Ok(ForegroundWaitDecision::NoWait { through: ready.through }) };
        if value == 0 { return Err(ForegroundWaitError::Fence) }
        let record = self.waits.entry(queue).or_insert(ForegroundWaitRecord {
            fence: ready.fence,
            installed: 0,
            pending: None,
            uncertain: false,
        });
        if record.installed >= value { return Ok(ForegroundWaitDecision::Installed { through: ready.through, value }) }
        let serial = self.next_serial.checked_add(1).ok_or(ForegroundWaitError::Exhausted)?;
        self.next_serial = serial;
        let token = ForegroundWaitToken { state_generation: self.state_generation, fence: ready.fence, queue, serial, value };
        record.pending = Some((token, ForegroundWaitState::Reserved));
        Ok(ForegroundWaitDecision::Reserve(ForegroundWaitReservation {
            ledger: None,
            token: Some(token),
            through: ready.through,
        }))
    }

    fn transition(
        &mut self, token: ForegroundWaitToken, expected: ForegroundWaitState,
        replacement: Option<ForegroundWaitState>,
    ) -> Result<(), ForegroundWaitError> {
        if token.state_generation != self.state_generation { return Err(ForegroundWaitError::StaleGeneration) }
        if token.fence != self.fence { return Err(ForegroundWaitError::Fence) }
        if self.queues.get(&token.queue.raw) != Some(&token.queue.generation) {
            return Err(ForegroundWaitError::Queue);
        }
        let record = self.waits.get_mut(&token.queue).ok_or(ForegroundWaitError::StaleToken)?;
        if record.pending != Some((token, expected)) { return Err(ForegroundWaitError::StaleToken) }
        record.pending = replacement.map(|state| (token, state));
        Ok(())
    }

    fn publish(&mut self, token: ForegroundWaitToken) -> Result<(), ForegroundWaitError> {
        self.transition(token, ForegroundWaitState::Accepted, None)?;
        self.waits.get_mut(&token.queue).unwrap().installed = token.value;
        Ok(())
    }

    fn uncertain(&mut self, token: ForegroundWaitToken) {
        if self.transition(token, ForegroundWaitState::Accepted, None).is_ok() {
            self.waits.get_mut(&token.queue).unwrap().uncertain = true;
        }
    }

    fn replace_generation(
        &mut self, current: u64, replacement: u64, fence_allocation: u64,
    ) -> Result<(), ForegroundWaitError> {
        if current != self.state_generation || replacement <= current {
            return Err(ForegroundWaitError::StaleGeneration);
        }
        if fence_allocation <= self.fence.allocation { return Err(ForegroundWaitError::Fence) }
        self.state_generation = replacement;
        self.fence = SparseFenceIdentity { state_generation: replacement, allocation: fence_allocation };
        self.waits.clear();
        Ok(())
    }
}

#[derive(Clone)]
#[allow(dead_code)]
struct ForegroundWaits(Arc<Mutex<ForegroundWaitLedger>>);

#[allow(dead_code)]
impl ForegroundWaits {
    fn new(state_generation: u64, fence_allocation: u64) -> Option<Self> {
        Some(Self(Arc::new(Mutex::new(ForegroundWaitLedger::new(state_generation, fence_allocation)?))))
    }

    fn register_queue(&self, raw: usize) -> Result<TitleQueueIdentity, ForegroundWaitError> {
        self.0.lock().unwrap().register_queue(raw)
    }

    fn fence(&self) -> SparseFenceIdentity { self.0.lock().unwrap().fence }

    fn retire_queue(&self, queue: TitleQueueIdentity) -> Result<(), ForegroundWaitError> {
        self.0.lock().unwrap().retire_queue(queue)
    }

    fn replace_generation(
        &self, current: u64, replacement: u64, fence_allocation: u64,
    ) -> Result<(), ForegroundWaitError> {
        self.0.lock().unwrap().replace_generation(current, replacement, fence_allocation)
    }

    fn reserve(
        &self, queue: TitleQueueIdentity, ready: SparseReadyReceipt,
    ) -> Result<ForegroundWaitDecision, ForegroundWaitError> {
        let result = self.0.lock().unwrap().reserve(queue, ready)?;
        Ok(match result {
            ForegroundWaitDecision::Reserve(mut reservation) => {
                reservation.ledger = Some(self.0.clone());
                ForegroundWaitDecision::Reserve(reservation)
            }
            ready => ready,
        })
    }
}

#[allow(dead_code)]
struct ForegroundWaitReservation {
    ledger: Option<Arc<Mutex<ForegroundWaitLedger>>>, token: Option<ForegroundWaitToken>, through: u64,
}

#[allow(dead_code)]
impl ForegroundWaitReservation {
    fn arm(mut self) -> Result<AcceptedForegroundWait, ForegroundWaitError> {
        let token = self.token.unwrap();
        self.ledger.as_ref().unwrap().lock().unwrap().transition(
            token, ForegroundWaitState::Reserved, Some(ForegroundWaitState::Accepted),
        )?;
        self.token.take();
        Ok(AcceptedForegroundWait { ledger: self.ledger.take().unwrap(), token: Some(token), through: self.through })
    }
}

impl Drop for ForegroundWaitReservation {
    fn drop(&mut self) {
        if let (Some(ledger), Some(token)) = (&self.ledger, self.token.take()) {
            let _ = ledger.lock().unwrap().transition(token, ForegroundWaitState::Reserved, None);
        }
    }
}

#[allow(dead_code)]
struct AcceptedForegroundWait {
    ledger: Arc<Mutex<ForegroundWaitLedger>>, token: Option<ForegroundWaitToken>, through: u64,
}

#[allow(dead_code)]
impl AcceptedForegroundWait {
    fn publish(mut self) -> Result<(u64, u64), ForegroundWaitError> {
        let token = self.token.unwrap();
        self.ledger.lock().unwrap().publish(token)?;
        self.token.take();
        Ok((self.through, token.value))
    }

    fn rollback(mut self) -> Result<(), ForegroundWaitError> {
        let token = self.token.unwrap();
        self.ledger.lock().unwrap().transition(token, ForegroundWaitState::Accepted, None)?;
        self.token.take();
        Ok(())
    }
}

impl Drop for AcceptedForegroundWait {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() { self.ledger.lock().unwrap().uncertain(token) }
    }
}

/// Point-in-time identities needed to prove that planning and a later serialized upload commit
/// still refer to the same sparse page. `source` names the observed CPU page by chunk hash rather
/// than retaining a raw pointer across the unlocked interval. Title writes are outside sparse
/// state/upload exclusion, so a future commit must own or revalidate the bytes it actually copies.
#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct UploadPageObservation {
    page: usize,
    pool: (u32, u32),
    hole: bool,
    sent: Option<[u64; CHUNKS]>,
    source: [u64; CHUNKS],
}

/// Owned point-in-time input for building a coordinated upload plan after releasing `STATE`.
/// Capture is the only step that needs the live mapping and title bytes; plan construction uses
/// these immutable observations, then commit revalidates them against the current State.
#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct UploadPlanningSnapshot {
    state_generation: u64,
    mapping_epoch: u64,
    claimed: Vec<(usize, WorkReasons)>,
    observations: Vec<UploadPageObservation>,
}

#[allow(dead_code)]
impl UploadPlanningSnapshot {
    fn capture(
        state_generation: u64,
        mapping_epoch: u64,
        claimed: &[(usize, WorkReasons)],
        mut observe: impl FnMut(usize) -> Option<UploadPageObservation>,
    ) -> Option<Self> {
        if state_generation == 0 { return None }
        let observations = claimed.iter().map(|&(page, _)| observe(page)).collect::<Option<Vec<_>>>()?;
        Some(Self { state_generation, mapping_epoch, claimed: claimed.to_vec(), observations })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct PlannedUploadPage {
    page: usize,
    tile: usize,
    pool: (u32, u32),
    reasons: WorkReasons,
    zero: bool,
    dirty: bool,
    dirty_ranges: Vec<(usize, usize)>,
    sent: Option<[u64; CHUNKS]>,
    source: [u64; CHUNKS],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum UploadPlanInvalid {
    StateGeneration,
    MappingEpoch,
    MissingPage,
    Mapping,
    SentHashes,
    SourceHashes,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LegacyParityRevalidation {
    Ready,
    RetryCurrentGeneration,
    RestartForGeneration,
}

/// Private owned stand-in for a future mapped staging allocation. Keeping the copied bytes here
/// makes it impossible for the record/execute callback to retain a borrowed title-memory view.
#[allow(dead_code)]
struct OwnedUploadStaging {
    range: UploadRingRange,
    bytes: Box<[u8]>,
}

fn build_upload_pages(
    claimed: &[(usize, WorkReasons)], mut observe: impl FnMut(usize) -> Option<UploadPageObservation>,
) -> Option<Vec<PlannedUploadPage>> {
    let mut pages = Vec::with_capacity(claimed.len());
    for &(page, reasons) in claimed {
        let observation = observe(page)?;
        if observation.page != page || observation.hole { return None }
        let dirty = reasons.0 & (WorkReasons::DIRTY.0 | WorkReasons::RETRY.0) != 0;
        let mut dirty_ranges = Vec::new();
        if dirty {
            let sent = observation.sent.as_ref().unwrap_or(&ZERO_HASHES);
            for (chunk, (&source, &previous)) in observation.source.iter().zip(sent).enumerate() {
                if source != previous { push_run(&mut dirty_ranges, page + chunk * CHUNK, CHUNK) }
            }
        }
        pages.push(PlannedUploadPage {
            page,
            tile: page & !(TILE - 1),
            pool: observation.pool,
            reasons,
            zero: reasons.0 & WorkReasons::ZERO.0 != 0,
            dirty,
            dirty_ranges,
            sent: observation.sent,
            source: observation.source,
        });
    }
    Some(pages)
}

fn validate_upload_pages(
    pages: &[PlannedUploadPage], mut observe: impl FnMut(usize) -> Option<UploadPageObservation>,
) -> Result<(), UploadPlanInvalid> {
    for planned in pages {
        let current = observe(planned.page).ok_or(UploadPlanInvalid::MissingPage)?;
        if current.page != planned.page
            || current.page & !(TILE - 1) != planned.tile
            || current.pool != planned.pool
            || current.hole
        {
            return Err(UploadPlanInvalid::Mapping);
        }
        if current.sent != planned.sent { return Err(UploadPlanInvalid::SentHashes) }
        if current.source != planned.source { return Err(UploadPlanInvalid::SourceHashes) }
    }
    Ok(())
}

#[derive(Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum CoordinatedSparseUploadError<C, E, S> {
    Invalid(UploadPlanInvalid),
    Reserve(UploadReservationError),
    Copy(C),
    StagingSize,
    Prepare(E),
    Accept(SparseBatchError),
    Signal(S),
    PublishReservation(UploadReservationError),
    PublishBatch(SparseBatchError),
}

/// New inert composition path: the plan owns a revocable batch lease rather than an independently
/// restoring work guard. Dropping before acceptance leaves the batch visible for foreground help.
#[allow(dead_code)]
struct CoordinatedSparseUploadPlan {
    lease: Option<SparseBatchLease>,
    state_generation: u64,
    mapping_epoch: u64,
    pages: Vec<PlannedUploadPage>,
}

#[allow(dead_code)]
impl CoordinatedSparseUploadPlan {
    fn build(
        lease: SparseBatchLease,
        state_generation: u64,
        mapping_epoch: u64,
        observe: impl FnMut(usize) -> Option<UploadPageObservation>,
    ) -> Option<Self> {
        let snapshot = UploadPlanningSnapshot::capture(
            state_generation, mapping_epoch, lease.pages(), observe,
        )?;
        Self::from_snapshot(lease, snapshot)
    }

    /// Build only from owned observations. A live caller captures `snapshot` under `STATE`, drops
    /// that lock, and performs this allocation/scanning work before reacquiring for `validate`.
    fn from_snapshot(lease: SparseBatchLease, snapshot: UploadPlanningSnapshot) -> Option<Self> {
        if lease.token.state_generation != snapshot.state_generation
            || lease.pages() != snapshot.claimed
            || snapshot.observations.len() != snapshot.claimed.len()
        {
            return None;
        }
        let mut observations = snapshot.observations.into_iter();
        let pages = build_upload_pages(&snapshot.claimed, |page| {
            let observation = observations.next()?;
            (observation.page == page).then_some(observation)
        })?;
        Some(Self {
            lease: Some(lease),
            state_generation: snapshot.state_generation,
            mapping_epoch: snapshot.mapping_epoch,
            pages,
        })
    }

    fn validate(
        &self,
        state_generation: u64,
        mapping_epoch: u64,
        observe: impl FnMut(usize) -> Option<UploadPageObservation>,
    ) -> Result<(), UploadPlanInvalid> {
        if state_generation != self.state_generation { return Err(UploadPlanInvalid::StateGeneration) }
        if mapping_epoch != self.mapping_epoch { return Err(UploadPlanInvalid::MappingEpoch) }
        validate_upload_pages(&self.pages, observe)
    }

    fn obsolete(mut self) -> Result<(), SparseBatchError> {
        self.lease.take().expect("unfinished coordinated sparse upload plan").obsolete()
    }

    fn retry(mut self) -> Result<(), SparseBatchError> {
        self.lease.take().expect("unfinished coordinated sparse upload plan").retry()
    }

    fn commit_reserved<C, E, S>(
        mut self,
        state_generation: u64,
        mapping_epoch: u64,
        observe: impl FnMut(usize) -> Option<UploadPageObservation>,
        reservations: &UploadReservations,
        slot: usize,
        bytes: usize,
        copy: impl FnOnce(UploadRingRange, &[PlannedUploadPage]) -> Result<Vec<u8>, C>,
        prepare: impl FnOnce(&OwnedUploadStaging, &[PlannedUploadPage]) -> Result<(), E>,
        execute: impl FnOnce(&OwnedUploadStaging, &[PlannedUploadPage]),
        signal: impl FnOnce() -> Result<u64, S>,
    ) -> Result<UploadSubmissionReceipt, CoordinatedSparseUploadError<C, E, S>> {
        self.validate(state_generation, mapping_epoch, observe)
            .map_err(CoordinatedSparseUploadError::Invalid)?;
        let reservation = reservations.reserve_for(state_generation, slot, bytes)
            .map_err(CoordinatedSparseUploadError::Reserve)?;
        let range = reservation.range();
        let copied = copy(range, &self.pages).map_err(CoordinatedSparseUploadError::Copy)?;
        if copied.len() != bytes { return Err(CoordinatedSparseUploadError::StagingSize) }
        let staging = OwnedUploadStaging { range, bytes: copied.into_boxed_slice() };
        prepare(&staging, &self.pages).map_err(CoordinatedSparseUploadError::Prepare)?;

        let batch = self.lease.take().expect("finished coordinated sparse upload plan").accept()
            .map_err(CoordinatedSparseUploadError::Accept)?;
        let accepted = AcceptedCoordinatedUpload::new(reservation, batch);
        execute(&staging, &self.pages);
        let fence = signal().map_err(CoordinatedSparseUploadError::Signal)?;
        match accepted.publish(fence) {
            Ok(receipt) => Ok(receipt),
            Err(CoordinatedPublishError::Reservation(error)) => {
                Err(CoordinatedSparseUploadError::PublishReservation(error))
            }
            Err(CoordinatedPublishError::Batch(error)) => Err(CoordinatedSparseUploadError::PublishBatch(error)),
        }
    }
}

enum LegacyParityUpload {
    Lease { lease: SparseBatchLease, state_generation: u64, mapping_epoch: u64 },
    Planned(CoordinatedSparseUploadPlan),
}

impl LegacyParityUpload {
    fn revalidate(
        &self,
        state_generation: u64,
        mapping_epoch: u64,
        observe: impl FnMut(usize) -> Option<UploadPageObservation>,
    ) -> LegacyParityRevalidation {
        let plan = match self {
            Self::Lease { state_generation: planned_generation, mapping_epoch: planned_epoch, .. } => {
                return if state_generation != *planned_generation {
                    LegacyParityRevalidation::RestartForGeneration
                } else if mapping_epoch != *planned_epoch {
                    LegacyParityRevalidation::RetryCurrentGeneration
                } else {
                    LegacyParityRevalidation::Ready
                };
            }
            Self::Planned(plan) => plan,
        };
        match plan.validate(state_generation, mapping_epoch, observe) {
            Ok(()) => LegacyParityRevalidation::Ready,
            Err(UploadPlanInvalid::StateGeneration) => LegacyParityRevalidation::RestartForGeneration,
            Err(_) => LegacyParityRevalidation::RetryCurrentGeneration,
        }
    }

    fn obsolete(self) -> Result<(), SparseBatchError> {
        match self {
            Self::Lease { lease, .. } => lease.obsolete(),
            Self::Planned(plan) => plan.obsolete(),
        }
    }

    fn retry(self) -> Result<(), SparseBatchError> {
        match self {
            Self::Lease { lease, .. } => lease.retry(),
            Self::Planned(plan) => plan.retry(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct UploadReservationToken {
    state_generation: u64,
    serial: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct UploadRingRange {
    start: u64,
    end: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
struct UploadSubmissionReceipt {
    token: UploadReservationToken,
    fence: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum UploadReservationState {
    Reserved,
    Submitted { fence: u64, retired: bool },
    Cancelled,
    Poisoned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum UploadReservationError {
    Poisoned,
    StaleGeneration,
    Slot,
    SlotBusy,
    Empty,
    TooLarge,
    Full,
    Exhausted,
    Missing,
    State,
    Fence,
}

struct UploadReservationRecord {
    slot: usize,
    range: UploadRingRange,
    state: UploadReservationState,
}

/// Inert ownership model for a later short upload commit phase. Ring positions only advance;
/// cancellation leaves a retired interval in the ordered ledger instead of rewinding across a
/// newer reservation. Submitted intervals become reusable only after their exact fence retires and
/// every older interval has retired. A failed ordering signal poisons the whole generation.
#[allow(dead_code)]
struct UploadReservationLedger {
    state_generation: u64,
    capacity: u64,
    alignment: u64,
    next_serial: u64,
    cursor: u64,
    reclaimed: u64,
    poisoned: bool,
    slots: Vec<Option<u64>>,
    order: VecDeque<u64>,
    records: BTreeMap<u64, UploadReservationRecord>,
}

#[allow(dead_code)]
impl UploadReservationLedger {
    fn new(state_generation: u64, slots: usize, capacity: usize, alignment: usize) -> Self {
        assert!(slots > 0);
        assert!(capacity > 0);
        assert!(alignment > 0 && alignment.is_power_of_two() && capacity % alignment == 0);
        Self {
            state_generation,
            capacity: capacity as u64,
            alignment: alignment as u64,
            next_serial: 0,
            cursor: 0,
            reclaimed: 0,
            poisoned: false,
            slots: vec![None; slots],
            order: VecDeque::new(),
            records: BTreeMap::new(),
        }
    }

    fn reserve(
        &mut self, state_generation: u64, slot: usize, bytes: usize,
    ) -> Result<(UploadReservationToken, UploadRingRange), UploadReservationError> {
        if state_generation != self.state_generation {
            return Err(UploadReservationError::StaleGeneration);
        }
        if self.poisoned {
            return Err(UploadReservationError::Poisoned);
        }
        let owner = self.slots.get(slot).ok_or(UploadReservationError::Slot)?;
        if owner.is_some() {
            return Err(UploadReservationError::SlotBusy);
        }
        if bytes == 0 {
            return Err(UploadReservationError::Empty);
        }
        if bytes as u64 > self.capacity {
            return Err(UploadReservationError::TooLarge);
        }
        let (cursor, reclaimed) = if self.order.is_empty() {
            let offset = self.cursor % self.capacity;
            let boundary = if offset == 0 {
                self.cursor
            } else {
                self.cursor.checked_add(self.capacity - offset).ok_or(UploadReservationError::Exhausted)?
            };
            (boundary, boundary)
        } else {
            (self.cursor, self.reclaimed)
        };
        let mask = self.alignment - 1;
        let mut start = cursor.checked_add(mask).ok_or(UploadReservationError::Exhausted)? & !mask;
        let offset = start % self.capacity;
        if offset.checked_add(bytes as u64).ok_or(UploadReservationError::Exhausted)? > self.capacity {
            start = start.checked_add(self.capacity - offset).ok_or(UploadReservationError::Exhausted)?;
        }
        let end = start.checked_add(bytes as u64).ok_or(UploadReservationError::Exhausted)?;
        if end.checked_sub(reclaimed).ok_or(UploadReservationError::Exhausted)? > self.capacity {
            return Err(UploadReservationError::Full);
        }
        let serial = self.next_serial.checked_add(1).ok_or(UploadReservationError::Exhausted)?;
        self.next_serial = serial;
        self.cursor = end;
        self.reclaimed = reclaimed;
        self.slots[slot] = Some(serial);
        let range = UploadRingRange { start, end };
        self.order.push_back(serial);
        self.records.insert(serial, UploadReservationRecord { slot, range, state: UploadReservationState::Reserved });
        Ok((UploadReservationToken { state_generation, serial }, range))
    }

    fn submit(&mut self, token: UploadReservationToken, fence: u64) -> Result<(), UploadReservationError> {
        let record = self.record_mut(token)?;
        if record.state != UploadReservationState::Reserved || fence == 0 {
            return Err(UploadReservationError::State);
        }
        record.state = UploadReservationState::Submitted { fence, retired: false };
        Ok(())
    }

    fn complete(&mut self, receipt: UploadSubmissionReceipt) -> Result<u64, UploadReservationError> {
        let record = self.record_mut(receipt.token)?;
        match record.state {
            UploadReservationState::Submitted { fence: expected, retired: false } if expected == receipt.fence => {
                record.state = UploadReservationState::Submitted { fence: receipt.fence, retired: true };
                let slot = record.slot;
                self.slots[slot] = None;
                Ok(self.reclaim())
            }
            UploadReservationState::Submitted { fence: expected, .. } if expected != receipt.fence => {
                Err(UploadReservationError::Fence)
            }
            _ => Err(UploadReservationError::State),
        }
    }

    fn cancel(&mut self, token: UploadReservationToken) -> Result<u64, UploadReservationError> {
        let record = self.record_mut(token)?;
        if record.state != UploadReservationState::Reserved {
            return Err(UploadReservationError::State);
        }
        record.state = UploadReservationState::Cancelled;
        let slot = record.slot;
        self.slots[slot] = None;
        Ok(self.reclaim())
    }

    fn poison(&mut self, token: UploadReservationToken) -> Result<(), UploadReservationError> {
        let record = self.record_mut(token)?;
        if record.state != UploadReservationState::Reserved {
            return Err(UploadReservationError::State);
        }
        record.state = UploadReservationState::Poisoned;
        self.poisoned = true;
        Ok(())
    }

    /// Fail closed even after submission publication: no slot or ring byte in this generation may
    /// be reused when the matching batch receipt could not be published.
    fn poison_generation(&mut self, token: UploadReservationToken) {
        if token.state_generation != self.state_generation { return }
        if let Some(record) = self.records.get_mut(&token.serial) {
            if matches!(record.state, UploadReservationState::Reserved | UploadReservationState::Submitted { .. }) {
                record.state = UploadReservationState::Poisoned;
            }
        }
        self.poisoned = true;
    }

    fn record_mut(
        &mut self, token: UploadReservationToken,
    ) -> Result<&mut UploadReservationRecord, UploadReservationError> {
        if token.state_generation != self.state_generation {
            return Err(UploadReservationError::StaleGeneration);
        }
        self.records.get_mut(&token.serial).ok_or(UploadReservationError::Missing)
    }

    fn reclaim(&mut self) -> u64 {
        while let Some(&serial) = self.order.front() {
            let retired = self.records.get(&serial).is_some_and(|record| match record.state {
                UploadReservationState::Cancelled => true,
                UploadReservationState::Submitted { retired, .. } => retired,
                UploadReservationState::Reserved | UploadReservationState::Poisoned => false,
            });
            if !retired {
                break;
            }
            let record = self.records.remove(&serial).unwrap();
            self.reclaimed = record.range.end;
            self.order.pop_front();
        }
        self.reclaimed
    }
}

#[derive(Clone)]
#[allow(dead_code)]
struct UploadReservations(Arc<Mutex<UploadReservationLedger>>);

#[allow(dead_code)]
impl UploadReservations {
    fn new(state_generation: u64, slots: usize, capacity: usize, alignment: usize) -> Self {
        Self(Arc::new(Mutex::new(UploadReservationLedger::new(state_generation, slots, capacity, alignment))))
    }

    fn reserve(&self, slot: usize, bytes: usize) -> Result<UploadReservation, UploadReservationError> {
        let mut ledger = self.0.lock().unwrap();
        let generation = ledger.state_generation;
        let (token, range) = ledger.reserve(generation, slot, bytes)?;
        Ok(UploadReservation { ledger: self.0.clone(), token: Some(token), range })
    }

    fn reserve_for(
        &self, state_generation: u64, slot: usize, bytes: usize,
    ) -> Result<UploadReservation, UploadReservationError> {
        let (token, range) = self.0.lock().unwrap().reserve(state_generation, slot, bytes)?;
        Ok(UploadReservation { ledger: self.0.clone(), token: Some(token), range })
    }

    fn complete(&self, receipt: UploadSubmissionReceipt) -> Result<u64, UploadReservationError> {
        self.0.lock().unwrap().complete(receipt)
    }
}

#[allow(dead_code)]
struct UploadReservation {
    ledger: Arc<Mutex<UploadReservationLedger>>,
    token: Option<UploadReservationToken>,
    range: UploadRingRange,
}

#[allow(dead_code)]
impl UploadReservation {
    fn range(&self) -> UploadRingRange { self.range }

    fn submit(mut self, fence: u64) -> Result<UploadSubmissionReceipt, UploadReservationError> {
        let token = self.token.unwrap();
        self.ledger.lock().unwrap().submit(token, fence)?;
        self.token.take();
        Ok(UploadSubmissionReceipt { token, fence })
    }

    fn poison(mut self) -> Result<(), UploadReservationError> {
        let token = self.token.unwrap();
        self.ledger.lock().unwrap().poison(token)?;
        self.token.take();
        Ok(())
    }

    /// Publish after execute acceptance. Publication failure is fail-closed: the exact record and
    /// its entire generation are poisoned instead of being cancelled by this guard's destructor.
    fn publish_after_accept(mut self, fence: u64) -> Result<UploadSubmissionReceipt, UploadReservationError> {
        let token = self.token.take().expect("finished upload reservation");
        let mut ledger = self.ledger.lock().unwrap();
        if fence == 0 {
            let _ = ledger.poison(token);
            return Err(UploadReservationError::State);
        }
        match ledger.submit(token, fence) {
            Ok(()) => Ok(UploadSubmissionReceipt { token, fence }),
            Err(error) => {
                let _ = ledger.poison(token);
                Err(error)
            }
        }
    }
}

impl Drop for UploadReservation {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            let _ = self.ledger.lock().unwrap().cancel(token);
        }
    }
}

enum CoordinatedPublishError {
    Reservation(UploadReservationError),
    Batch(SparseBatchError),
}

/// Own both accepted ledgers. Any exit before both receipts are published poisons both generation
/// domains, including the narrow boundary after reservation submission but before frontier receipt.
struct AcceptedCoordinatedUpload {
    reservation: Option<UploadReservation>,
    reservation_token: UploadReservationToken,
    reservation_ledger: Arc<Mutex<UploadReservationLedger>>,
    batch: Option<AcceptedSparseBatch>,
    batch_token: ClaimFrontierToken,
    batch_ledger: Arc<Mutex<SparseBatchLedger>>,
    armed: bool,
}

impl AcceptedCoordinatedUpload {
    fn new(reservation: UploadReservation, batch: AcceptedSparseBatch) -> Self {
        Self {
            reservation_token: reservation.token.unwrap(),
            reservation_ledger: reservation.ledger.clone(),
            batch_token: batch.token.unwrap().0,
            batch_ledger: batch.ledger.clone(),
            reservation: Some(reservation),
            batch: Some(batch),
            armed: true,
        }
    }

    fn publish(mut self, fence: u64) -> Result<UploadSubmissionReceipt, CoordinatedPublishError> {
        let receipt = self.reservation.take().unwrap().publish_after_accept(fence)
            .map_err(CoordinatedPublishError::Reservation)?;
        self.batch.take().unwrap().receipt(fence).map_err(CoordinatedPublishError::Batch)?;
        self.armed = false;
        Ok(receipt)
    }
}

impl Drop for AcceptedCoordinatedUpload {
    fn drop(&mut self) {
        if !self.armed { return }
        if let Some(reservation) = self.reservation.take() { let _ = reservation.poison(); }
        if let Some(batch) = self.batch.take() { drop(batch); }
        self.reservation_ledger.lock().unwrap().poison_generation(self.reservation_token);
        let mut batches = self.batch_ledger.lock().unwrap();
        if batches.work.state_generation == self.batch_token.state_generation { batches.poisoned = true }
    }
}

/// Coordination ownership attached to one live sparse state generation. Keeping the three
/// ledgers together prevents a later activation from pairing work, waits, or ring reservations
/// from different device/fence generations.
#[allow(dead_code)]
#[derive(Clone)]
struct SparseCoordination {
    batches: SparseBatches,
    waits: ForegroundWaits,
    reservations: UploadReservations,
}

impl SparseCoordination {
    fn new(state_generation: u64, fence_allocation: u64) -> Option<Self> {
        Some(Self {
            batches: SparseBatches::with_fence(state_generation, fence_allocation)?,
            waits: ForegroundWaits::new(state_generation, fence_allocation)?,
            reservations: UploadReservations::new(state_generation, UPLOAD_SLOTS, STAGING, RING_ALIGN),
        })
    }

    fn seal_legacy(
        &self,
        owner: u64,
        zero: &[(usize, usize)],
        retry: &[usize],
        dirty: &[(usize, usize)],
    ) -> Result<Option<SparseBatchLease>, SparseBatchError> {
        self.batches.seal_legacy(owner, zero, retry, dirty)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PreparationOutcome {
    Ready,
    RetryRequired,
    ExecutedUntracked,
}

impl From<SubmitResult> for PreparationOutcome {
    fn from(result: SubmitResult) -> Self {
        match result {
            SubmitResult::Submitted => Self::Ready,
            SubmitResult::NotSubmitted => Self::RetryRequired,
            SubmitResult::Unordered => Self::ExecutedUntracked,
        }
    }
}

struct RetiredTiles {
    tiles: Vec<(u32, u32)>,
    /// Fence marks on `self.queue` and every known title queue (`None` = signal failed, keep held).
    marks: Option<Vec<(ID3D12Fence, u64)>>,
    /// when they were unmapped: they cool (`State::cooling`) from then on once the marks are reached
    at: std::time::Instant,
}

struct State {
    generation: u64,
    mapping_epoch: u64,
    /// Exact-generation ownership for the staged prepare/accept/publish sparse path. The legacy
    /// submission path remains authoritative until activation PRs journal and consume its work.
    #[allow(dead_code)]
    coordination: SparseCoordination,
    dev: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
    /// reusable auto-reset Win32 event for fence waits (no per-wait CreateEventW/CloseHandle)
    event: HANDLE,
    /// fence signalled on title queues after their lists, so a copy back waits for them
    rb_fence: ID3D12Fence,
    rb_value: u64,
    /// copies back after the title's submissions (`readback_after_execute`): on a queue of their
    /// own (`rbq`, None: on `queue`, waited for on the CPU), landing in the title's memory once
    /// done (`drain_readbacks`). The title's queues can wait for work the title submits after
    /// this one, from the same thread: waiting here for its lists stopped that submission – and,
    /// on `queue`, every upload behind it (`before_execute` waiting for ring room).
    rbq: Option<(ID3D12CommandQueue, ID3D12Fence)>,
    rbq_value: u64,
    /// async readback job queues and reusable staging pool (leaf lock separate from `STATE`)
    readbacks: Arc<SharedReadbacks>,
    /// async readback recorder and per-queue fence map (locked outside `STATE`)
    rb_submit: Arc<Mutex<RbSubmit>>,
    /// (window start, reserved buffer) of each sparse window
    windows: Vec<(usize, ID3D12Resource)>,
    /// video-memory pool heaps (None: given back, its index free for the next one)
    pools: Vec<Option<ID3D12Heap>>,
    /// pool tiles to use, partitioned by heap (`FreePools`): those of a pool that were never used,
    /// and cooled ones (`cooling`)
    free: FreePools,
    /// pool tiles freed (`flush_parked`, once `quarantined` lets them go, and `release_if_unused`),
    /// oldest first, with when: used again `COOL` after that at the earliest (`take_tile`).
    /// Before, a tile unmapped went to `free` and was the next one taken, mapped, cleared and
    /// filled for another address on the sparse queue while title work submitted before the
    /// unmap could still read it.
    cooling: VecDeque<((u32, u32), std::time::Instant)>,
    /// pools all of whose tiles came free: released once the GPU is past `value` and a moment
    /// went by (titles' queues may still read what was mapped there)
    retired: Vec<(ID3D12Heap, u64, std::time::Instant)>,
    /// when `retire_free_pools` last ran (`retire_due`)
    retire_tried: std::time::Instant,
    /// tile address → pool tile + uncommitted page hole mask + per-page sent chunk hashes
    mapped: FxHashMap<usize, TileEntry>,
    /// pool tiles also used by placed buffers (not freed while they are)
    pinned: FxHashMap<(u32, u32), u32>,
    /// tile address → pool tile of tiles decommitted since the last title submit, still mapped
    /// in their window: unmapped together (`flush_parked`) before the next submit, when the free
    /// list runs out, or past `PARKED_MAX`; a commit there before that takes its tile back with
    /// no new mapping. (A retail title moving physical pages one at a time while it loads
    /// decommits and commits again thousands of tiles a second, and each `UpdateTileMappings`
    /// took ~0.15 ms.)
    parked: FxHashMap<usize, (u32, u32)>,
    /// tile address → pool tile of decommitted tiles a placed buffer uses: they stay mapped in
    /// their window until the buffer is gone. Memory committed there again is then the same for
    /// the window (what shaders write) and the buffer (what an ExecuteIndirect or a copy of the
    /// title reads); with a new tile in the window the buffer kept the old one, with old bytes
    kept: FxHashMap<usize, (u32, u32)>,
    /// tile addresses given a pool tile in `mapped` whose `UpdateTileMappings` has not been sent
    /// yet: mapped together (`flush_pending`, one call per window) before the next title submit,
    /// before XCL's own GPU work on the windows, or past `PENDING_MAX`. (A retail title starting
    /// on a Series S profile maps 4.3 GB of physical pages in small pieces: one mapping call
    /// each, ~0.4 ms, held its loading thread ~25 s and its stall watchdog ended it.)
    pending: Vec<usize>,
    /// placed buffers → the tiles they use (address, pool tile)
    buffers: FxHashMap<usize, Vec<(usize, (u32, u32))>>,
    /// unmapped parked pool tiles awaiting `self.queue` and in-flight title queues before they
    /// cool (`cooling`)
    quarantined: VecDeque<RetiredTiles>,
    /// per-title-queue fences used to mark in-flight GPU work (`note_queue`, `quarantine_tiles_for`);
    /// raw keys are removed only by exact-generation queue retirement (`forget_queue`) before that
    /// queue's address can be reused
    queue_fences: FxHashMap<usize, (ID3D12Fence, u64)>,
    /// reused tiles / decommitted partial pages to clear: (address, bytes)
    to_zero: BTreeSet<(usize, usize)>,
    /// ring-buffered upload staging heap (`STAGING` bytes, persistently mapped)
    staging: Option<(ID3D12Resource, *mut u8)>,
    /// monotonic ring-buffer write/submitted/retired byte cursors
    ring_write: u64,
    ring_submitted: u64,
    ring_free: u64,
    /// submitted ring chunks awaiting GPU completion: `(fence_val, ring_end_pos)`
    inflight: VecDeque<(u64, u64)>,
    /// false after ExecuteCommandLists succeeded but its ordering Signal failed: no upload
    /// allocator or staging byte may be reused because completion can no longer be proved.
    uploads_ordered: bool,
    /// dedicated 1 MiB pre-zeroed buffer for tile/hole clears (zero staging bandwidth)
    zero_buf: Option<ID3D12Resource>,
    /// ring of async upload command allocators + lists
    slots: Vec<UploadSlot>,
    next_slot: usize,
    /// dedicated readback staging buffer + command allocator / list
    readback: Option<(ID3D12Resource, *mut u8)>,
    rb_alloc: ID3D12CommandAllocator,
    rb_list: ID3D12GraphicsCommandList,
    rb_last_fence: u64,
    /// the fence value each title queue last waited for
    waited: FxHashMap<usize, u64>,
    /// `mapped` as runs of neighbouring tiles (start -> end), kept with it: sorted out of all of
    /// them again (a retail title has 150,000) for each look behind a commit
    runs: BTreeMap<usize, usize>,
    /// reusable buffer for dirty runs in `before_execute` (avoids per-submit heap alloc)
    scratch_runs: Vec<(usize, usize)>,
    /// `XCL_LOOK_STATS=2`: the title's pages XCL wrote itself since the last look (copies back
    /// landed, immediates stored; `OWN_PAGES_MAX` at most), for `look_stats::own_writes`
    own_pages: Vec<usize>,
    /// CPU tile → the window tiles over it (tiles of a GPU view, `map_mirror`), with video memory
    /// or not: what the CPU writes there goes to each of them. Several can show one CPU tile
    mirrors: FxHashMap<usize, Vec<usize>>,
    /// window tile → the CPU tile under it, for each tile in `mirrors`
    mirror_src: FxHashMap<usize, usize>,
    /// the CPU tiles of `mirrors` as runs (start -> end): where a CPU write has to reach the GPU
    /// too (`mapped_ranges`). Tiles over a CPU view are never in `runs`: their own addresses are
    /// not the CPU's
    src_runs: BTreeMap<usize, usize>,
    /// `src_runs` for the threads that decommit the title's CPU memory, and their decommits under
    /// way (`SOURCES`; a test's state has one of its own)
    sources: &'static Mutex<Sources>,
    /// window tiles given video memory over their CPU view since the last submit: all of their
    /// committed pages go up with it (the CPU wrote them before the tile was there, and those
    /// writes were taken by a look before)
    fill: Vec<usize>,
    /// window tiles asked for (`map_mirror`) and not handled yet
    asks: Vec<usize>,
    /// window tiles whose GPU-view page changed (`remapped`), linked again at the next submit
    remaps: Vec<usize>,
    /// window tiles linked over their CPU tiles (`mirror_src`) that wait for their video memory
    /// and fill, oldest first (`fill_later`)
    later: VecDeque<(usize, usize)>,
    /// committed pages filled at a submit at most, for tiles given video memory there
    /// (`XCL_MIRROR_FILL_MB`): the rest wait for the next
    fill_pages: usize,
    /// tiles over a CPU view with video memory now, and the most there may be (`XCL_MIRROR_MB`)
    mirrored: usize,
    mirror_cap: usize,
    /// tiles over a CPU view left without video memory for `mirror_cap`, or as none was to be had:
    /// given it once there is room (`retry_refused`), or once their CPU view is written
    refused: FxHashSet<usize>,
    /// when a tile over a CPU view last found no video memory (`take_tile`): for a moment the
    /// others do not try to make a heap again (each tried, and logged it), and the refused ones
    /// wait
    no_memory: Option<std::time::Instant>,
    /// tiles listed with no video memory (`LAZY`) whose CPU bytes found none at an upload
    /// (`upload_dirty`): all that is committed there goes up at a submit once there is some
    starved: Vec<usize>,
    counts: MirrorCounts,
    diag: ViewDiag,
    /// `XCL_VIEW_POISON`: the tile the windows' tiles without video memory show (`poison_windows`)
    poison: Option<Poison>,
    /// Dirty CPU pages whose write-watch bits were reset before an upload failed.
    retry_pages: Vec<usize>,
}

/// The one 64 KiB of video memory the tiles of the windows without video memory of their own are
/// mapped to (`D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE`), a buffer placed over it, and the
/// pattern's upload copy it is filled from again (`refresh_poison`: what a shader wrote there
/// lands in the one tile).
struct Poison {
    heap: ID3D12Heap,
    buf: ID3D12Resource,
    src: ID3D12Resource,
    at: std::time::Instant,
}

/// What the tiles over CPU views did, for the timing line
#[derive(Default, Debug)]
struct MirrorCounts {
    /// the most tiles with video memory at once
    peak: usize,
    /// tiles filled whole from their CPU view, and the bytes of their committed pages
    fills: u64,
    fill_bytes: u64,
    /// tiles refused for `mirror_cap` or for want of video memory (each once until it has some)
    refused: u64,
    /// tiles whose video memory went back (their CPU view decommitted, or the view gone)
    given_back: u64,
    /// pages that were not committed in a CPU view, written since and asked about again
    rechecks: u64,
}

/// Diagnostics of the GPU-view tiles, all off unless set: `XCL_VIEW_KEEP=1` (a tile over a CPU
/// view keeps its video memory once filled, within `mirror_cap`), `XCL_VIEW_POISON=1` (or a
/// 32-bit pattern, `0x…`: what a tile of the windows without video memory reads instead of zeros,
/// `State::poison_windows`) and `XCL_VIEW_STATS=1`. Any of them logs `view_stats_line` every 2 s.
struct ViewSwitches {
    keep: bool,
    poison: Option<u32>,
    stats: bool,
}

fn view_switches() -> &'static ViewSwitches {
    static ON: std::sync::OnceLock<ViewSwitches> = std::sync::OnceLock::new();
    ON.get_or_init(|| {
        let var = |k: &str| std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty() && v != "0");
        let (keep, poison) = (var("XCL_VIEW_KEEP").is_some(), poison_pattern(var("XCL_VIEW_POISON").as_deref()));
        ViewSwitches { keep, poison, stats: keep || poison.is_some() || var("XCL_VIEW_STATS").is_some() }
    })
}

/// The poison pattern of `XCL_VIEW_POISON` (set, not "0"): a NaN as a float, a wild index or count
/// as an integer, or the pattern given in hex.
fn poison_pattern(v: Option<&str>) -> Option<u32> {
    let v = v?;
    match v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
        Some(hex) => u32::from_str_radix(hex, 16).ok().filter(|&p| p != 0),
        None => Some(0x7FC0_DEAD),
    }
}

/// Whether a tile over a CPU view whose video memory would go back keeps it mapped where it is
/// (`XCL_VIEW_KEEP`): while the tiles with video memory and kept, this one among them, are within
/// the cap.
fn keep_view_tile(keep: bool, held: usize, cap: usize) -> bool {
    keep && held <= cap
}

/// The bucket of a tile that got video memory again `submits` after it went: 0-1, 2-8, 9-64, more.
fn back_bucket(submits: u64) -> usize {
    match submits {
        0..=1 => 0,
        2..=8 => 1,
        9..=64 => 2,
        _ => 3,
    }
}

/// What the GPU-view diagnostics count (`ViewSwitches`; nothing while they are off but submits).
#[derive(Default, Debug)]
struct ViewDiag {
    keep: bool,
    stats: bool,
    /// submits so far (`before_execute_on`)
    submits: u64,
    /// tiles kept mapped that would have gone back (`XCL_VIEW_KEEP`)
    kept: u64,
    /// tile given back or kept → the submit then, until it gets video memory again
    left_at: FxHashMap<usize, u64>,
    /// tiles that got video memory again, by submits since they went (`back_bucket`), and the
    /// bytes of their committed pages filled again
    back: [u64; 4],
    back_bytes: u64,
    /// submits that left tiles asked for waiting (`later`, `asks`) for a later one
    deferred: u64,
}

/// Tiles of GPU views asked for (`map_mirror` queued them), while `ViewSwitches::stats`.
static VIEW_ASKED: AtomicU64 = AtomicU64::new(0);

/// `view_stats`: [asked, filled, given back, kept, back 0-1, 2-8, 9-64, more, bytes back, submits,
/// deferred] so far, then [with video memory, kept mapped, waiting, refused] now.
const VS: usize = 15;
const VS_NOW: usize = 11;

/// tiles over CPU views asked for taken in at a time, and refused ones tried again at a submit at
/// most (`State::map_mirrors`); the rest wait for the next one
const MIRROR_ASKS_MAX: usize = 4096;

/// Committed pages of the CPU views filled into tiles given video memory at a submit at most
/// (`XCL_MIRROR_FILL_MB`, 256 MB by default; at least a page; with 16 MB a close-up's mesh
/// data was asked for at a submit and drawn with part of it still unfilled, as shards and
/// slabs on the heads of a title's characters): the fills are hashed and go
/// through the staging ring on the title's submitting thread, with the state's lock held.
fn fill_pages() -> usize {
    static PAGES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PAGES.get_or_init(|| {
        let mb = std::env::var("XCL_MIRROR_FILL_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(256);
        (mb.saturating_mul(1 << 20) / PAGE).max(1)
    })
}

/// Video memory for tiles over CPU views at most, in tiles (`XCL_MIRROR_MB`, 4096 MB by default):
/// a GPU view covers the whole of a 24 GB CPU reservation, and the card is mostly in use already.
fn mirror_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let mb = std::env::var("XCL_MIRROR_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(4096);
        mb.saturating_mul(1 << 20) / TILE
    })
}

impl Drop for State {
    fn drop(&mut self) {
        if !self.event.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }
}

const K0: u64 = 0x9E37_79B9_7F4A_7C15;
const K1: u64 = 0xBF58_476D_1CE4_E5B9;
const K2: u64 = 0x94D0_49BB_1331_11EB;
const K3: u64 = 0x243F_6A88_85A3_08D3;

/// 4-way parallel hash over 8 `u64` words (64 bytes): breaks serial multiply latency chain.
#[inline(always)]
const fn hash_words(w: [u64; 8]) -> u64 {
    let a = (w[0] ^ K3).wrapping_mul(K0) ^ w[4].wrapping_mul(K1);
    let b = (w[1] ^ K0).wrapping_mul(K1) ^ w[5].wrapping_mul(K2);
    let c = (w[2] ^ K1).wrapping_mul(K2) ^ w[6].wrapping_mul(K3);
    let d = (w[3] ^ K2).wrapping_mul(K3) ^ w[7].wrapping_mul(K0);
    let h = a.rotate_left(27).wrapping_add(b.rotate_left(13)) ^ c.rotate_left(41).wrapping_add(d.rotate_left(53));
    h.wrapping_mul(K0) ^ (h >> 32)
}

const ZERO_HASH: u64 = hash_words([0; 8]);
const ZERO_HASHES: [u64; CHUNKS] = [ZERO_HASH; CHUNKS];

#[inline(always)]
fn read_chunk_words(c: &[u8]) -> [u64; 8] {
    debug_assert_eq!(c.len(), CHUNK);
    let p = c.as_ptr() as *const [u8; 8];
    unsafe {
        [
            u64::from_le_bytes(p.add(0).read_unaligned()),
            u64::from_le_bytes(p.add(1).read_unaligned()),
            u64::from_le_bytes(p.add(2).read_unaligned()),
            u64::from_le_bytes(p.add(3).read_unaligned()),
            u64::from_le_bytes(p.add(4).read_unaligned()),
            u64::from_le_bytes(p.add(5).read_unaligned()),
            u64::from_le_bytes(p.add(6).read_unaligned()),
            u64::from_le_bytes(p.add(7).read_unaligned()),
        ]
    }
}

/// hash of 64 bytes
#[inline(always)]
fn chunk_hash(c: &[u8]) -> u64 {
    hash_words(read_chunk_words(c))
}

#[inline(always)]
fn push_run(out: &mut Vec<(usize, usize)>, addr: usize, len: usize) {
    match out.last_mut() {
        Some(last) if last.0 + last.1 == addr => last.1 += len,
        _ => out.push((addr, len)),
    }
}

/// Land GPU bytes in one CPU page, 64-byte chunk by chunk. A chunk still matching `sent` has
/// no newer CPU write and may be replaced; a changed chunk keeps its CPU bytes and its old hash,
/// so the next write-watch look uploads it. With `gpu_written`, `src` is bytes the GPU's own
/// commands wrote: they land in a changed chunk too (the CPU wrote other bytes of it, such as the
/// cells a wait before the write watched), and its old hash still makes the next look upload the
/// whole chunk, the GPU's bytes with the CPU's. The hash check and copy are not atomic against a
/// concurrent CPU store: callers rely on the title's fence discipline to exclude stores to the
/// same bytes while its synchronous copy-back lands. Whether it wrote any (`State::own_pages`).
fn land_page(slot: &mut Option<Box<[u64; CHUNKS]>>, page: &mut [u8], at: usize, src: &[u8], gpu_written: bool) -> bool {
    debug_assert_eq!(page.len(), PAGE);
    debug_assert!(at + src.len() <= page.len());
    let sent = slot.get_or_insert_with(|| Box::new(ZERO_HASHES));
    let end = at + src.len();
    let mut offset = at;
    let mut wrote = false;
    while offset < end {
        let chunk = offset / CHUNK;
        let chunk_start = chunk * CHUNK;
        let copy_end = ((chunk + 1) * CHUNK).min(end);
        let source = offset - at;
        if chunk_hash(&page[chunk_start..chunk_start + CHUNK]) == sent[chunk] {
            page[offset..copy_end].copy_from_slice(&src[source..source + copy_end - offset]);
            sent[chunk] = chunk_hash(&page[chunk_start..chunk_start + CHUNK]);
            wrote = true;
        } else if gpu_written {
            page[offset..copy_end].copy_from_slice(&src[source..source + copy_end - offset]);
            wrote = true;
        }
        offset = copy_end;
    }
    wrote
}

/// `State::own_pages` keeps this many pages at most (a look's worth)
const OWN_PAGES_MAX: usize = 1 << 16;

thread_local! {
    /// `XCL_LOOK_STATS=2`: the next `before_execute` on this thread follows a look (`note_look`)
    static LOOKED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The submit about to call `before_execute` on this thread looked for CPU writes
/// (`XCL_LOOK_STATS=2`): XCL's own writes since the look before are counted against it.
pub(super) fn note_look() {
    LOOKED.with(|l| l.set(true));
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HashUpdate {
    at: usize,
    hash: u64,
}

const MAX_BATCH_HASH_UPDATES: usize = STAGING / CHUNK;

/// One synchronously-recorded dirty upload batch. The slot, ring rollback point, and unpublished
/// hash plan move together so a failed submission cannot publish bytes that have no fence receipt.
struct PreparedDirtyBatch {
    slot: usize,
    ring_start: u64,
    updates: Vec<HashUpdate>,
}

impl PreparedDirtyBatch {
    fn new(slot: usize, ring_start: u64) -> Self {
        Self { slot, ring_start, updates: Vec::new() }
    }

    fn reopen(mut self, slot: usize, ring_start: u64) -> Self {
        debug_assert!(self.updates.is_empty());
        self.slot = slot;
        self.ring_start = ring_start;
        self
    }

    fn hash_plan_would_overflow(&self, bytes: usize) -> bool {
        batch_hash_plan_would_overflow(self.updates.len(), bytes)
    }

    fn append_hashes(&mut self, at: usize, bytes: &[u8]) {
        append_hash_updates(at, bytes, &mut self.updates);
    }

    fn clear_after_submit(&mut self) {
        self.updates.clear();
    }
}

fn batch_hash_plan_would_overflow(current: usize, bytes: usize) -> bool {
    bytes / CHUNK > MAX_BATCH_HASH_UPDATES.saturating_sub(current)
}

/// Plan changed 64-byte chunk runs in a 4 KiB page without publishing their hashes. The hashes
/// become `sent` only after the corresponding upload has a durable queue-fence receipt.
/// This keeps the fresh-zero-page fast path allocation-free.
#[inline]
fn scan_page(
    page: &[u8],
    pg: usize,
    slot: &Option<Box<[u64; CHUNKS]>>,
    out: &mut Vec<(usize, usize)>,
) {
    if let Some(sent) = slot.as_deref() {
        for (k, c) in page.chunks_exact(CHUNK).enumerate() {
            let h = chunk_hash(c);
            if h == sent[k] {
                continue;
            }
            push_run(out, pg + k * CHUNK, CHUNK);
        }
        return;
    }
    // Page has never had non-zero data sent: skip zero chunks with a cheap OR-reduction and
    // allocate the hash array only if at least one chunk is non-zero.
    let mut chunks = page.chunks_exact(CHUNK).enumerate();
    while let Some((k, c)) = chunks.next() {
        let w = read_chunk_words(c);
        if (w[0] | w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7]) == 0 {
            continue;
        }
        let h = hash_words(w);
        if h == ZERO_HASH {
            continue;
        }
        push_run(out, pg + k * CHUNK, CHUNK);
        for (k2, c2) in chunks {
            let h2 = chunk_hash(c2);
            if h2 == ZERO_HASH {
                continue;
            }
            push_run(out, pg + k2 * CHUNK, CHUNK);
        }
        break;
    }
}

fn append_hash_updates(at: usize, bytes: &[u8], updates: &mut Vec<HashUpdate>) {
    debug_assert_eq!(at % CHUNK, 0);
    debug_assert_eq!(bytes.len() % CHUNK, 0);
    debug_assert!(updates.len() + bytes.len() / CHUNK <= MAX_BATCH_HASH_UPDATES);
    updates.extend(bytes.chunks_exact(CHUNK).enumerate().map(|(i, chunk)| HashUpdate {
        at: at + i * CHUNK,
        hash: chunk_hash(chunk),
    }));
}

fn publish_hash_updates(mapped: &mut FxHashMap<usize, TileEntry>, updates: &[HashUpdate]) {
    for update in updates {
        let tile = update.at & !(TILE - 1);
        let Some(entry) = mapped.get_mut(&tile) else { continue };
        let page = update.at & !(PAGE - 1);
        let index = (page - tile) / PAGE;
        let chunk = (update.at - page) / CHUNK;
        if entry.holes & (1u16 << index) != 0 {
            continue;
        }
        if update.hash == ZERO_HASH && entry.sent[index].is_none() {
            continue;
        }
        entry.sent[index].get_or_insert_with(|| Box::new(ZERO_HASHES))[chunk] = update.hash;
    }
}

/// The byte ranges (offset, length) of `page` whose 64-byte chunks differ from `sent`
/// (updated to the page's chunks).
pub fn changed_chunks(page: &[u8], sent: &mut [u64; CHUNKS]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (k, c) in page.chunks_exact(CHUNK).enumerate() {
        let h = chunk_hash(c);
        if h == sent[k] {
            continue;
        }
        sent[k] = h;
        push_run(&mut out, k * CHUNK, CHUNK);
    }
    out
}

/// Plan a contiguous sub-allocation `(aligned_pos, chunk_len)` inside a ring buffer of `cap`
/// bytes starting at monotonic cursor `ring_write`, keeping each chunk `<= cap / 2` and
/// avoiding mid-copy wrap-around without wasting large tail gaps.
#[inline]
fn plan_ring_chunk(ring_write: u64, want: usize, cap: usize) -> (u64, usize) {
    let mut chunk = want.min(cap / 2);
    let mut aligned = (ring_write + (RING_ALIGN as u64 - 1)) & !(RING_ALIGN as u64 - 1);
    let off = (aligned as usize) % cap;
    if off + chunk > cap {
        let tail = cap - off;
        if tail >= PAGE {
            chunk = tail;
        } else {
            aligned += tail as u64;
        }
    }
    (aligned, chunk)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PreparedDirtyCopy {
    ring_start: u64,
    staging_offset: usize,
    window_offset: usize,
    bytes: usize,
}

/// Bound one dirty copy to both its sparse window and its contiguous staging-ring allocation.
/// Keeping this arithmetic pure lets planning and recording use exactly the same split points.
fn prepare_dirty_copy(
    at: usize,
    remaining: usize,
    window_base: usize,
    window_size: usize,
    ring_write: u64,
    staging_cap: usize,
) -> PreparedDirtyCopy {
    debug_assert!(at >= window_base && at < window_base + window_size);
    let to_window_end = window_base + window_size - at;
    let (ring_start, bytes) = plan_ring_chunk(ring_write, remaining.min(to_window_end), staging_cap);
    PreparedDirtyCopy {
        ring_start,
        staging_offset: ring_start as usize % staging_cap,
        window_offset: at - window_base,
        bytes,
    }
}

// D3D12 objects are free-threaded; staging pointers and event handle are only used under STATE lock
unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static NEXT_STATE_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Serialize mutation of upload slots, staging cursors and retry journals. Fence waits release this
/// gate while a shared generation lifetime keeps their exact native backing. Lock order is
/// `WAIT_GATES` → `UPLOAD_STATE_LIFETIME` (shared) → `UPLOAD_GATE` → `STATE`;
/// initialization takes `PLACEMENT` then the lifetime lock exclusively before `UPLOAD_GATE`.
static UPLOAD_GATE: Mutex<()> = Mutex::new(());

/// A waiter keeps its exact `State` generation and all of that generation's native backing alive
/// while `UPLOAD_GATE` is released. Other uploaders take shared ownership and can cross the wait;
/// setup takes exclusive ownership before `UPLOAD_GATE` and cannot replace `STATE` meanwhile.
static UPLOAD_STATE_LIFETIME: std::sync::RwLock<()> = std::sync::RwLock::new(());

type UploadGuard = std::sync::MutexGuard<'static, ()>;

#[derive(Clone, Copy, Default)]
struct UploadGateTimes {
    wait_us: u64,
    hold_us: u64,
    released_us: u64,
}

struct UploadGateTiming {
    enabled: bool,
    held_since: Option<std::time::Instant>,
    times: UploadGateTimes,
}

impl UploadGateTiming {
    fn acquire(enabled: bool) -> (Option<UploadGuard>, Self) {
        let waiting = enabled.then(std::time::Instant::now);
        let guard = UPLOAD_GATE.lock().unwrap();
        let acquired = enabled.then(std::time::Instant::now);
        let wait_us = waiting.zip(acquired).map_or(0, |(waiting, acquired)| acquired.duration_since(waiting).as_micros() as u64);
        (Some(guard), Self { enabled, held_since: acquired, times: UploadGateTimes { wait_us, ..Default::default() } })
    }

    fn with_released<T>(&mut self, guard: &mut Option<UploadGuard>, wait: impl FnOnce() -> T) -> T {
        let released = self.enabled.then(std::time::Instant::now);
        if let (Some(held_since), Some(released)) = (self.held_since.take(), released) {
            self.times.hold_us = self.times.hold_us.wrapping_add(released.duration_since(held_since).as_micros() as u64);
        }
        drop(guard.take());
        let result = wait();
        *guard = Some(UPLOAD_GATE.lock().unwrap());
        let acquired = self.enabled.then(std::time::Instant::now);
        if let (Some(released), Some(acquired)) = (released, acquired) {
            self.times.released_us = self.times.released_us.wrapping_add(acquired.duration_since(released).as_micros() as u64);
        }
        self.held_since = acquired;
        result
    }

    fn finish(mut self) -> UploadGateTimes {
        if let Some(held_since) = self.held_since.take() {
            self.times.hold_us = self.times.hold_us.wrapping_add(held_since.elapsed().as_micros() as u64);
        }
        self.times
    }
}

/// A placement may release `STATE` while the driver creates its reserved resource. Keep sparse
/// state replacement and another placement from crossing that exact-state transaction.
static PLACEMENT: Mutex<()> = Mutex::new(());

/// Serializes sparse preparation through wait enqueue for one title queue. Without this, a
/// concurrent submission could observe a reserved `waited` value and execute before the thread
/// which reserved it has called the native queue `Wait`.
static WAIT_GATES: Mutex<FxHashMap<usize, Arc<Mutex<()>>>> = Mutex::new(FxHashMap::with_hasher(BuildHasherDefault::new()));

/// The state's queue again, outside `STATE` (`queue`). `STATE` is held around D3D12 calls (an
/// upload's allocator `Reset` in `begin_slot`), and `late_release` asks for the queue from a
/// resource's last `Release`, which D3D12 can make inside a call of its own on another thread
/// (a list's `Reset`, with `XCL_DRED=1` on every start): the two then waited for each other.
static QUEUE: Mutex<Option<ID3D12CommandQueue>> = Mutex::new(None);

/// The active state's async readback queue, reachable without locking `STATE` (`settle_readbacks*`,
/// `start_readback_thread`). Only locked briefly to clone the `Arc` (never nested with other locks).
static READBACKS: Mutex<Option<Arc<SharedReadbacks>>> = Mutex::new(None);

fn readbacks_arc() -> Option<Arc<SharedReadbacks>> {
    READBACKS.lock().unwrap().clone()
}

fn remove_queue_fence<F>(fences: &mut FxHashMap<usize, (F, u64)>, raw: usize) -> Option<(F, u64)> {
    fences.remove(&raw)
}

fn initial_quarantine_marks<F: Clone>(own: (F, u64), retiring: &[(F, u64)], blocked: bool) -> Option<Vec<(F, u64)>> {
    (!blocked).then(|| std::iter::once(own).chain(retiring.iter().cloned()).collect())
}

/// Remove raw queue-address state before a retired queue's final owner is released. Retired tile
/// batches own fence clones independently; the cached fence drops after unlocking.
pub(super) fn forget_queue(id: super::present::QueueId) {
    let (gone, rb_submit) = {
        let mut state = STATE.lock().unwrap();
        let _held = holding(Holder::Other);
        match state.as_mut() {
            Some(state) => {
                state.waited.remove(&id.raw());
                (remove_queue_fence(&mut state.queue_fences, id.raw()), Some(Arc::clone(&state.rb_submit)))
            }
            None => (None, None),
        }
    };
    // (the copy-back fences of the queue: `RbSubmit` never nests with `STATE`; a copy back still
    // in flight keeps its own clones of them)
    let rb_gone = rb_submit.and_then(|sub| sub.lock().unwrap().rb_queue_fences.remove(&id.raw()));
    let gate = WAIT_GATES.lock().unwrap().remove(&id.raw());
    drop(gone);
    drop(rb_gone);
    drop(gate);
}

/// (bytes of video memory in pool heaps, of those mapped) – `XCL_ACTIVITY`
pub fn pool_bytes() -> (usize, usize) {
    let g = lock_state();
    let _held = holding(Holder::Other);
    let Some(s) = g.as_ref() else { return (0, 0) };
    let pools = s.pools.iter().filter(|p| p.is_some()).count() * POOL_TILES as usize * TILE;
    let quarantined: usize = s.quarantined.iter().map(|b| b.tiles.len()).sum();
    (pools, pools.saturating_sub((s.free.len() + s.cooling.len() + quarantined) * TILE))
}

/// Active pool bytes, occupied tile bytes, retired pool bytes, and committed CPU page bytes for
/// `device`, from one state lock so a pool moving to the retired list cannot be counted twice.
pub(super) fn memory_bytes_for(device: usize) -> (usize, usize, usize, usize) {
    let g = lock_state();
    let _held = holding(Holder::Other);
    let Some(s) = g.as_ref() else { return (0, 0, 0, 0) };
    memory_byte_counts(
        device,
        s.dev.as_raw() as usize,
        s.pools.iter().filter(|p| p.is_some()).count(),
        s.free.len(),
        s.retired.len(),
        s.mapped.values().filter(|e| e.src == 0).map(|e| e.holes),
    )
}

fn memory_byte_counts(
    device: usize,
    owner: usize,
    active_pools: usize,
    free_tiles: usize,
    retired_pools: usize,
    mapped_holes: impl Iterator<Item = u16>,
) -> (usize, usize, usize, usize) {
    if device != owner {
        return (0, 0, 0, 0);
    }
    let pool_bytes = (POOL_TILES as usize).saturating_mul(TILE);
    let pools = active_pools.saturating_mul(pool_bytes);
    let occupied = pools.saturating_sub(free_tiles.saturating_mul(TILE));
    let retired = retired_pools.saturating_mul(pool_bytes);
    let committed = mapped_holes.fold(0usize, |bytes, holes| {
        let pages = PAGES_PER_TILE.saturating_sub(holes.count_ones() as usize);
        bytes.saturating_add(pages.saturating_mul(PAGE))
    });
    (pools, occupied, retired, committed)
}
/// commits before the device existed: the pages committed, as disjoint page ranges (start →
/// end) merged with their neighbours, so a commit or decommit costs O(log n) and not a scan of
/// every page committed so far. Like `committed` / `decommitted` with a state, a range counts
/// every page it touches; `init` commits them once the windows exist.
static EARLY: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());
/// Before the sparse device exists: pages the title protected other than writable. Commits clear
/// these ranges, and `init` replays what remains after the early commits.
static EARLY_NOT_RW: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

fn early_protection(early: &mut BTreeMap<usize, usize>, a: usize, n: usize, writable: bool) {
    let (start, end) = page_span(a, n);
    if writable {
        early_remove(early, start, end);
    } else {
        early_add(early, start, end);
    }
}

/// The pages `[a, a + n)` touches, as `[start, end)`.
fn page_span(a: usize, n: usize) -> (usize, usize) {
    (a & !(PAGE - 1), a.saturating_add(n).saturating_add(PAGE - 1) & !(PAGE - 1))
}

/// Add the pages `[a, b)` to `early` (ranges as start -> end), merged with the ranges they overlap
/// or touch.
fn early_add(early: &mut BTreeMap<usize, usize>, mut a: usize, mut b: usize) {
    if a >= b {
        return;
    }
    if let Some((&s, &e)) = early.range(..=a).next_back() {
        if e >= a {
            a = s;
            b = b.max(e);
            early.remove(&s);
        }
    }
    loop {
        let Some((s, e)) = early.range(a..=b).next().map(|(&s, &e)| (s, e)) else { break };
        b = b.max(e);
        early.remove(&s);
    }
    early.insert(a, b);
}

/// Take the pages `[a, b)` out of `early`; a range that overlaps it keeps the rest.
fn early_remove(early: &mut BTreeMap<usize, usize>, a: usize, b: usize) {
    if a >= b {
        return;
    }
    if let Some((&s, &e)) = early.range(..a).next_back() {
        if e > a {
            early.insert(s, a);
            if e > b {
                early.insert(b, e);
                return;
            }
        }
    }
    loop {
        let Some((s, e)) = early.range(a..b).next().map(|(&s, &e)| (s, e)) else { break };
        early.remove(&s);
        if e > b {
            early.insert(b, e);
            break;
        }
    }
}
/// generation counter incremented whenever the set of mapped tiles changes (for watch_ranges cache)
static MAPPED_GEN: AtomicU64 = AtomicU64::new(0);

/// Generation counter of `mapped_ranges()` (changes only when tiles are mapped or unmapped).
pub fn mapped_gen() -> u64 {
    MAPPED_GEN.load(Ordering::Acquire)
}

fn buffer_desc(bytes: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    }
}

unsafe fn new_state(dev: &ID3D12Device, windows: Vec<(usize, ID3D12Resource)>) -> windows::core::Result<State> {
    if lazy_asked() && !null_tiles_read_zero(dev) && !LAZY_REFUSED.swap(true, Ordering::Relaxed) {
        crate::xlog_warn!("sparse windows: XCL_SPARSE_LAZY=1 is left out - on this device a tile with no video memory does not read zeros (tiled resources tier under 2)");
    }
    let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())?;
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
    let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
    let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None)?;
    let rbq = (|| -> windows::core::Result<(ID3D12CommandQueue, ID3D12Fence)> {
        let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())?;
        let _ = q.SetName(windows::core::w!("xcl sparse readback queue"));
        Ok((q, dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?))
    })()
    .ok();
    let mut slots = Vec::with_capacity(UPLOAD_SLOTS);
    for _ in 0..UPLOAD_SLOTS {
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
        list.Close()?;
        let _ = list.SetName(windows::core::w!("xcl sparse upload list"));
        let _ = alloc.SetName(windows::core::w!("xcl sparse upload allocator"));
        slots.push(UploadSlot { alloc, list, fence_val: 0 });
    }
    let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
    let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None)?;
    rb_list.Close()?;
    let _ = rb_list.SetName(windows::core::w!("xcl sparse readback list"));
    let _ = rb_alloc.SetName(windows::core::w!("xcl sparse readback allocator"));
    let _ = queue.SetName(windows::core::w!("xcl sparse queue"));
    let readbacks = Arc::new(SharedReadbacks::new(rb_fence.clone(), rbq.as_ref().map(|(_, f)| f.clone())));
    let rb_submit = Arc::new(Mutex::new(RbSubmit::new(dev.clone())));
    let generation = NEXT_STATE_GENERATION.fetch_add(1, Ordering::Relaxed);
    let fence_allocation = fence.as_raw() as usize as u64;
    let coordination = SparseCoordination::new(generation, fence_allocation)
        .expect("valid sparse state generation and fence identity");
    Ok(State {
        generation,
        mapping_epoch: 0,
        coordination,
        dev: dev.clone(),
        queue,
        fence,
        value: 0,
        event,
        rb_fence,
        rb_value: 0,
        rbq,
        rbq_value: 0,
        readbacks,
        rb_submit,
        windows,
        pools: Vec::new(),
        free: FreePools::new(),
        cooling: VecDeque::new(),
        retired: Vec::new(),
        retire_tried: std::time::Instant::now(),
        mapped: FxHashMap::default(),
        pinned: FxHashMap::default(),
        parked: FxHashMap::default(),
        kept: FxHashMap::default(),
        pending: Vec::new(),
        buffers: FxHashMap::default(),
        quarantined: VecDeque::new(),
        queue_fences: FxHashMap::default(),
        to_zero: BTreeSet::new(),
        staging: None,
        ring_write: 0,
        ring_submitted: 0,
        ring_free: 0,
        inflight: VecDeque::new(),
        uploads_ordered: true,
        zero_buf: None,
        slots,
        next_slot: 0,
        readback: None,
        rb_alloc,
        rb_list,
        rb_last_fence: 0,
        waited: FxHashMap::default(),
        runs: BTreeMap::new(),
        scratch_runs: Vec::new(),
        own_pages: Vec::new(),
        mirrors: FxHashMap::default(),
        mirror_src: FxHashMap::default(),
        src_runs: BTreeMap::new(),
        sources: &SOURCES,
        fill: Vec::new(),
        asks: Vec::new(),
        remaps: Vec::new(),
        later: VecDeque::new(),
        fill_pages: fill_pages(),
        mirrored: 0,
        mirror_cap: mirror_cap(),
        refused: FxHashSet::default(),
        no_memory: None,
        starved: Vec::new(),
        counts: MirrorCounts::default(),
        diag: ViewDiag { keep: view_switches().keep, stats: view_switches().stats, ..Default::default() },
        poison: None,
        retry_pages: Vec::new(),
    })
}

/// pool heap number of the tiles of a shadowed window: they are the window's own buffer, not
/// pool tiles (never mapped, freed or pinned)
const SHADOW_POOL: u32 = u32::MAX;

/// A committed window the GPU could not open as a heap (`real::open_uma`): `buf` in video
/// memory stands for `[base, base + size)` and is kept in step like the tiles of a sparse
/// window -- the CPU's writes (write-watch, changed 64-byte chunks) go to it before each
/// ExecuteCommandLists, GPU writes come back on Map and after the lists' memory operations.
/// All of it counts as mapped from the start: the window is committed. False: no queue for
/// the copies (the buffer then never sees the CPU's writes).
pub unsafe fn shadow_window(dev: &ID3D12Device, base: usize, size: usize, buf: &ID3D12Resource) -> bool {
    let mut g = STATE.lock().unwrap();
    if g.is_none() {
        // no sparse window was made (`init`): the copies still need the queue and the lists
        match new_state(dev, Vec::new()) {
            Ok(s) => *g = Some(s),
            Err(e) => {
                crate::xlog_warn!("shadowed window at {base:#x}: {e}");
                return false;
            }
        }
    }
    let Some(s) = g.as_mut() else { return false };
    s.windows.push((base, buf.clone()));
    for (i, t) in (base..base + size).step_by(TILE).enumerate() {
        s.mapped.insert(t, TileEntry::with_holes((SHADOW_POOL, i as u32), 0));
    }
    early_add(&mut s.runs, base, base + size.next_multiple_of(TILE));
    MAPPED_GEN.fetch_add(1, Ordering::Release);
    true
}

/// The sparse windows' buffers on `dev`: `(window_index, buffer_raw, gpu_va)` for each
/// sparse window in `gpu::uma` and each view window, made the same way (empty: no sparse
/// windows). Nothing of a view window is mapped here: its tiles get video memory when asked for.
pub unsafe fn init(dev: &ID3D12Device) -> Vec<(usize, usize, u64)> {
    let Some(u) = crate::gpu::uma::get() else { return Vec::new() };
    let mut out = Vec::new();
    let mut windows = Vec::new();
    for k in 0..u.shader_windows() {
        if !u.is_sparse_window(k) && !u.is_view_window(k) {
            continue;
        }
        let (base, size) = u.window(k);
        let mut r: Option<ID3D12Resource> = None;
        if let Err(e) = dev.CreateReservedResource(&buffer_desc(size as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut r) {
            crate::xlog_warn!("sparse window {k}: CreateReservedResource: {e}");
            return Vec::new();
        }
        let Some(r) = r else { return Vec::new() };
        // (a name for what a device removal reports about an address, `real::log_device_removed`)
        let _ = r.SetName(&windows::core::HSTRING::from(format!("xcl memory window {k}{}", if u.is_view_window(k) { " (GPU views)" } else { "" })));
        out.push((k, r.clone().into_raw() as usize, r.GetGPUVirtualAddress()));
        windows.push((base, r));
    }
    if out.is_empty() {
        return out;
    }
    let made = new_state(dev, windows);
    match made {
        Ok(mut s) => {
            let views = out.iter().filter(|w| u.is_view_window(w.0)).count();
            crate::xlog_info!("D3D12.X: {} sparse window(s) for the title's own memory and {views} for GPU views of it (tiled buffers on video memory)", out.len() - views);
            let sw = view_switches();
            if let Some(pattern) = sw.poison {
                s.poison_windows(pattern);
            }
            if sw.stats {
                crate::xlog_info!(
                    "D3D12.X: GPU-view diagnostics: keep {}, poison {}, a line every 2 s",
                    if sw.keep { format!("on ({} tiles at most, XCL_MIRROR_MB)", s.mirror_cap) } else { "off".into() },
                    match (sw.poison, s.poison.is_some()) {
                        (Some(p), true) => format!("{p:#010x}"),
                        (Some(_), false) => "failed (zeros)".into(),
                        (None, _) => "off".into(),
                    }
                );
                start_view_stats();
            }
            // a state this replaces (the windows set up again) goes after the lock: the release of
            // its last buffers reaches `forget_buffer`, which takes the lock (#344)
            // (the queue first: `queue()` never answers with one older than the state's)
            let _placement = PLACEMENT.lock().unwrap();
            let _upload_lifetime = UPLOAD_STATE_LIFETIME.write().unwrap();
            let _uploads = UPLOAD_GATE.lock().unwrap();
            let old_queue = QUEUE.lock().unwrap().replace(s.queue.clone());
            let _old_rb = READBACKS.lock().unwrap().replace(Arc::clone(&s.readbacks));
            let mut state = STATE.lock().unwrap();
            let old = state.replace(s);
            let s = state.as_mut().unwrap();
            for (a, b) in std::mem::take(&mut *EARLY.lock().unwrap()) {
                super::title_textures::mark_hot(a, b - a);
                s.commit_range(a, b - a);
            }
            for (a, b) in std::mem::take(&mut *EARLY_NOT_RW.lock().unwrap()) {
                mark_protection(&mut s.mapped, a, b - a, false);
            }
            drop(state);
            if old.as_ref().is_some_and(|state| !state.uploads_ordered) {
                // ExecuteCommandLists was accepted but no fence can prove when its allocator,
                // staging source, or destination windows become unused. Keep that generation
                // alive until process exit rather than release memory the GPU may still touch.
                crate::xlog_warn!("sparse: retaining an unordered upload generation after its queue Signal failed");
                std::mem::forget(old);
            } else {
                drop(old);
            }
            drop(old_queue);
            out
        }
        Err(e) => {
            crate::xlog_warn!("sparse windows: {e}");
            Vec::new()
        }
    }
}

impl State {
    #[inline]
    fn window_index(&self, a: usize) -> Option<usize> {
        self.windows.iter().position(|(b, _)| a >= *b && a < *b + crate::gpu::uma::WINDOW)
    }

    /// Capture every identity used by unlocked sparse planning while this State is authoritative.
    /// The returned value owns hashes and mapping metadata; it retains no title-memory borrow.
    /// (A tile over a CPU view, `TileEntry::src`, is read at its CPU address: its own is no memory.)
    unsafe fn upload_page_observation(&self, page: usize) -> Option<UploadPageObservation> {
        let tile = page & !(TILE - 1);
        let entry = self.mapped.get(&tile)?;
        let index = (page - tile) / PAGE;
        if index >= PAGES_PER_TILE { return None }
        let hole = entry.holes & (1u16 << index) != 0;
        let sent = entry.sent[index].as_deref().copied();
        let source = if hole {
            ZERO_HASHES
        } else {
            let mut bytes = [0u8; PAGE];
            if !crate::hle::dstorage::read_own(entry.cpu(tile) + index * PAGE, bytes.as_mut_ptr(), bytes.len()) {
                return None;
            }
            std::array::from_fn(|chunk| {
                let start = chunk * CHUNK;
                chunk_hash(&bytes[start..start + CHUNK])
            })
        };
        Some(UploadPageObservation { page, pool: entry.pool, hole, sent, source })
    }

    /// A pool tile (true: used before, to be cleared). One freed less than `COOL` ago, or one
    /// still `quarantined`, is not taken: a retired pool is taken back or a heap made instead.
    /// (Parked tiles are not flushed for it: freed now, they would wait and cool all the same.)
    unsafe fn take_tile(&mut self) -> Option<((u32, u32), bool)> {
        #[cfg(test)]
        if tests::NO_TILES.with(|c| c.get()) {
            return None;
        }
        if super::real::fence_completed(&self.fence) == u64::MAX {
            return None;
        }
        self.drain_quarantined();
        if let Some(t) = reuse_tile(&mut self.free, &mut self.cooling, std::time::Instant::now()) {
            return Some((t, true));
        }
        // a pool retired a moment ago (its last tile came free) is taken back before a new
        // heap is made: freeing and allocating a tile in turn made a new pool each time,
        // up to 18 GB of retired heaps in 2 s (#913). Its tiles are reused as freed tiles are
        // (zeroed), and they all cooled before it was retired.
        if let Some((heap, _, _)) = self.retired.pop() {
            let h = self.add_pool(heap);
            return Some(((h, 0), true));
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: POOL_TILES as u64 * TILE as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
        };
        // (a heap takes its time, with the state locked: what waits for its mapping goes first, a
        // CPU signal that asks meanwhile gives up after 2 ms)
        self.flush_pending();
        let mut heap: Option<ID3D12Heap> = None;
        if let Err(e) = super::title_textures::own_work(|| self.dev.CreateHeap(&desc, &mut heap)) {
            crate::xlog_warn!("sparse windows: no video memory for more tiles ({e})");
            return None;
        }
        let heap = heap?;
        let _ = heap.SetName(windows::core::w!("xcl sparse tile pool"));
        let h = self.add_pool(heap);
        Some(((h, 0), false))
    }

    /// `heap` becomes a pool (in a given-back pool's index if there is one); its tiles but the
    /// first, which the caller takes, go to the free list.
    fn add_pool(&mut self, heap: ID3D12Heap) -> u32 {
        let h = match self.pools.iter().position(|p| p.is_none()) {
            Some(i) => {
                self.pools[i] = Some(heap);
                i as u32
            }
            None => {
                self.pools.push(Some(heap));
                self.pools.len() as u32 - 1
            }
        };
        self.free.add_heap(h);
        h
    }

    /// Pools whose every tile is free again (cooled, `COOL`) go (their video memory – and as much
    /// of the system's commit: a retail game's homes of physical pages left 0.75 GB of pools for
    /// 0.29 GB of tiles in use). Call after their tiles are unmapped and signalled. `done`: the
    /// value the sparse queue's fence has reached (a retired heap goes once it is past its mark).
    fn retire_free_pools_with(&mut self, done: u64) {
        let now = std::time::Instant::now();
        self.retire_tried = now;
        if done == u64::MAX {
            return;
        }
        self.retired.retain(|(_, v, at)| !(retirement_reached(done, *v) && now.duration_since(*at).as_secs() >= 2));
        self.free.drain_cooling(&mut self.cooling, now);
        let gone: Vec<usize> = free_heaps(&self.free, &self.cooling, self.pools.len(), now).into_iter().filter(|&h| self.pools[h].is_some()).collect();
        if gone.is_empty() {
            return;
        }
        for h in gone {
            self.free.clear_heap(h);
            if let Some(heap) = self.pools[h].take() {
                self.retired.push((heap, self.value, now));
            }
        }
    }

    fn retire_free_pools(&mut self) {
        self.retire_free_pools_with(unsafe { super::real::fence_completed(&self.fence) });
    }

    /// Map `tiles` (tile addresses) of `res` (starting at `res_base`) onto pool tiles,
    /// batched per pool heap with contiguous runs merged into single regions.
    unsafe fn map_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[(usize, (u32, u32))]) {
        if tiles.is_empty() {
            return;
        }
        let mut by_heap: FxHashMap<u32, Vec<(u32, u32)>> = FxHashMap::default();
        debug_assert!(tiles.iter().all(|t| t.1 != LAZY), "a tile with no video memory is mapped nowhere");
        for &(a, (h, t)) in tiles {
            by_heap.entry(h).or_default().push((((a - res_base) / TILE) as u32, t));
        }
        count_tile_calls(res_base, false, by_heap.len(), tiles.len());
        for (h, mut pairs) in by_heap {
            pairs.sort_unstable_by_key(|&(x, _)| x);
            let mut coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = Vec::with_capacity(pairs.len());
            let mut regions: Vec<D3D12_TILE_REGION_SIZE> = Vec::with_capacity(pairs.len());
            let mut flags: Vec<D3D12_TILE_RANGE_FLAGS> = Vec::with_capacity(pairs.len());
            let mut offsets: Vec<u32> = Vec::with_capacity(pairs.len());
            let mut counts: Vec<u32> = Vec::with_capacity(pairs.len());
            let mut i = 0;
            while i < pairs.len() {
                let (x0, t0) = pairs[i];
                let mut run = 1u32;
                while i + (run as usize) < pairs.len() {
                    let (nx, nt) = pairs[i + run as usize];
                    if nx == x0 + run && nt == t0 + run {
                        run += 1;
                    } else {
                        break;
                    }
                }
                coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: x0, ..Default::default() });
                regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
                flags.push(D3D12_TILE_RANGE_FLAG_NONE);
                offsets.push(t0);
                counts.push(run);
                i += run as usize;
            }
            self.queue.UpdateTileMappings(
                res,
                coords.len() as u32,
                Some(coords.as_ptr()),
                Some(regions.as_ptr()),
                self.pools[h as usize].as_ref().expect("a pool tile of a live pool"),
                flags.len() as u32,
                Some(flags.as_ptr()),
                Some(offsets.as_ptr()),
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
    }

    unsafe fn unmap_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[usize]) {
        if tiles.is_empty() {
            return;
        }
        let mut xs: Vec<u32> = tiles.iter().map(|&a| ((a - res_base) / TILE) as u32).collect();
        xs.sort_unstable();
        let mut coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = Vec::with_capacity(xs.len());
        let mut regions: Vec<D3D12_TILE_REGION_SIZE> = Vec::with_capacity(xs.len());
        let mut flags: Vec<D3D12_TILE_RANGE_FLAGS> = Vec::with_capacity(xs.len());
        let mut counts: Vec<u32> = Vec::with_capacity(xs.len());
        // (`XCL_VIEW_POISON`: onto the poison tile, every tile of a run on its one tile)
        let poison = self.poison.as_ref().map(|p| &p.heap);
        let mut i = 0;
        while i < xs.len() {
            let x0 = xs[i];
            let mut run = 1u32;
            while i + (run as usize) < xs.len() && xs[i + run as usize] == x0 + run {
                run += 1;
            }
            coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: x0, ..Default::default() });
            regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
            flags.push(if poison.is_some() { D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE } else { D3D12_TILE_RANGE_FLAG_NULL });
            counts.push(run);
            i += run as usize;
        }
        let offsets = vec![0u32; if poison.is_some() { counts.len() } else { 0 }];
        count_tile_calls(res_base, true, 1, tiles.len());
        self.queue.UpdateTileMappings(
            res,
            coords.len() as u32,
            Some(coords.as_ptr()),
            Some(regions.as_ptr()),
            poison,
            flags.len() as u32,
            Some(flags.as_ptr()),
            poison.map(|_| offsets.as_ptr()),
            Some(counts.as_ptr()),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
    }

    /// `XCL_VIEW_POISON`: every tile of `res` (`tiles` of them) onto the poison tile, before the
    /// tiles with video memory are mapped over it.
    unsafe fn poison_all(&self, res: &ID3D12Resource, tiles: u32) {
        let Some(p) = &self.poison else { return };
        if tiles == 0 {
            return;
        }
        let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
        let region = D3D12_TILE_REGION_SIZE { NumTiles: tiles, UseBox: false.into(), ..Default::default() };
        let (flag, off) = (D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE, 0u32);
        self.queue.UpdateTileMappings(res, 1, Some(&coord), Some(&region), &p.heap, 1, Some(&flag), Some(&off), Some(&tiles), D3D12_TILE_MAPPING_FLAG_NONE);
    }

    /// `XCL_VIEW_POISON`: a tile of video memory filled with `pattern`, and every tile of the
    /// windows mapped onto it (one call each, before any has video memory of its own); unmapped
    /// tiles go back to it from here on (`unmap_tiles`), and a buffer placed in a window starts
    /// on it (`place_buffer`). Many tiles of reserved buffers on one tile of a heap is what
    /// `REUSE_SINGLE_TILE` is for. Logged and left off when the tile cannot be made.
    unsafe fn poison_windows(&mut self, pattern: u32) {
        let made = (|| -> windows::core::Result<Poison> {
            let desc = D3D12_HEAP_DESC {
                SizeInBytes: TILE as u64,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
            };
            let mut heap: Option<ID3D12Heap> = None;
            self.dev.CreateHeap(&desc, &mut heap)?;
            let heap = heap.ok_or_else(|| windows::core::Error::from(windows::Win32::Foundation::E_FAIL))?;
            let _ = heap.SetName(windows::core::w!("xcl poison tile"));
            let mut d = buffer_desc(TILE as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            let mut buf: Option<ID3D12Resource> = None;
            self.dev.CreatePlacedResource(&heap, 0, &d, D3D12_RESOURCE_STATE_COMMON, None, &mut buf)?;
            let mut src: Option<ID3D12Resource> = None;
            self.dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut src)?;
            let (Some(buf), Some(src)) = (buf, src) else { return Err(windows::Win32::Foundation::E_FAIL.into()) };
            let mut p = std::ptr::null_mut();
            src.Map(0, None, Some(&mut p))?;
            std::slice::from_raw_parts_mut(p as *mut u32, TILE / 4).fill(pattern);
            src.Unmap(0, None);
            Ok(Poison { heap, buf, src, at: std::time::Instant::now() })
        })();
        match made {
            Ok(p) => self.poison = Some(p),
            Err(e) => {
                crate::xlog_warn!("XCL_VIEW_POISON: no poison tile ({e}) – tiles without video memory read zeros");
                return;
            }
        }
        let windows: Vec<ID3D12Resource> = self.windows.iter().map(|w| w.1.clone()).collect();
        let mut tiles = 0u64;
        for r in &windows {
            let n = (r.GetDesc().Width as usize).div_ceil(TILE) as u32;
            self.poison_all(r, n);
            tiles += n as u64;
        }
        self.refresh_poison();
        crate::xlog_info!("XCL_VIEW_POISON: {tiles} tiles of {} window(s) show {pattern:#010x} where they have no video memory", windows.len());
    }

    /// The poison tile filled with its pattern again (at most once every 16 ms: at a submit),
    /// on the sparse queue the title's queues wait for.
    unsafe fn refresh_poison(&mut self) {
        let Some((buf, src)) = self.poison.as_mut().map(|p| {
            p.at = std::time::Instant::now();
            (p.buf.clone(), p.src.clone())
        }) else {
            return;
        };
        // (no slot free without a wait, or none could be reset: the pattern goes back at a later submit)
        let BeginSlot::Ready(idx) = self.begin_slot() else { return };
        self.slots[idx].list.CopyBufferRegion(&buf, 0, &src, 0, TILE as u64);
        let _ = self.submit_slot(idx);
    }

    /// `ensure_mapped` for a commit with `lazy_commits`: the tiles of `[a, a + n)` are listed
    /// (watched, scanned, their holes kept) with no video memory (`LAZY`). A tile still mapped
    /// from before (parked, or kept for a buffer) is taken back as it is.
    unsafe fn ensure_listed(&mut self, a: usize, n: usize) {
        let end = a.saturating_add(n);
        let mut t = a & !(TILE - 1);
        let mut new = false;
        while t < end {
            if !self.mapped.contains_key(&t) && !self.mirror_src.contains_key(&t) && self.window_index(t).is_some() {
                if self.parked.contains_key(&t) || self.kept.contains_key(&t) {
                    self.ensure_mapped(t, TILE, |_| (0xFFFF, 0));
                } else {
                    self.mapped.insert(t, TileEntry::with_holes(LAZY, 0xFFFF));
                    early_add(&mut self.runs, t, t + TILE);
                    new = true;
                }
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        if new {
            MAPPED_GEN.fetch_add(1, Ordering::Release);
        }
    }

    /// The `LAZY` tiles of `[a, a + n)` get their video memory now (their mapping is pending,
    /// as `ensure_mapped` leaves it): (the tiles that did, those left without). A tile used
    /// before goes to `reused` as the run to clear, which the caller sees to before anything is
    /// written there. A moment after no video memory was found (`no_memory`) only tiles at hand
    /// are taken, as in `new_mirror`, but with `force` (a placed buffer maps its tiles once).
    unsafe fn materialize(&mut self, a: usize, n: usize, force: bool, reused: &mut Vec<(usize, usize)>) -> (usize, usize) {
        let end = a.saturating_add(n);
        let mut t = a & !(TILE - 1);
        let (mut got, mut left) = (0, 0);
        while t < end {
            if self.mapped.get(&t).is_some_and(|e| e.pool == LAZY) {
                let at_hand = !self.free.is_empty() || cooled(&self.cooling, std::time::Instant::now()) || !self.retired.is_empty();
                let wait = left > 0 || (!force && self.short_of_memory() && !at_hand);
                match if wait { None } else { self.take_tile() } {
                    Some((tile, used)) => {
                        if let Some(e) = self.mapped.get_mut(&t) {
                            e.pool = tile;
                        }
                        if used {
                            reused.push((t, TILE));
                        }
                        self.pending.push(t);
                        got += 1;
                    }
                    None => {
                        if !wait {
                            self.no_memory = Some(std::time::Instant::now());
                        }
                        left += 1;
                    }
                }
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        (got, left)
    }

    /// The title's lists about to be submitted write `ranges` on the GPU (`will_write`): their
    /// tiles listed with no video memory get it, cleared at this submit before the lists run.
    unsafe fn give_memory(&mut self, ranges: &[(usize, usize)]) {
        let mut reused = Vec::new();
        for &(a, n) in ranges {
            self.materialize(a, n, false, &mut reused);
        }
        self.to_zero.extend(reused);
    }

    /// tiles of `[a, a + n)` get video memory (mapped in their window); `initial_state` supplies
    /// the initial `(holes, untrusted protection)` masks for each newly mapped 64 KiB tile. Tiles
    /// over a CPU view (`mirror_src`) are left to `new_mirror`.
    unsafe fn ensure_mapped(&mut self, a: usize, n: usize, mut initial_state: impl FnMut(usize) -> (u16, u16)) {
        if n == 0 {
            return;
        }
        let first = a & !(TILE - 1);
        let end = a.saturating_add(n);
        let mut new = false;
        let mut unparked = false;
        let mut t = first;
        while t < end {
            if !self.mapped.contains_key(&t) && !self.mirror_src.contains_key(&t) && self.window_index(t).is_some() {
                // still mapped there: no new mapping, what it held is cleared
                if let Some(tile) = self.parked.remove(&t).or_else(|| self.kept.remove(&t)) {
                    let (holes, not_rw) = initial_state(t);
                    let mut entry = TileEntry::with_holes(tile, holes);
                    entry.not_rw = not_rw;
                    self.mapped.insert(t, entry);
                    early_add(&mut self.runs, t, t + TILE);
                    self.to_zero.insert((t, TILE));
                    unparked = true;
                } else {
                    let Some((tile, reused)) = self.take_tile() else { break };
                    let (holes, not_rw) = initial_state(t);
                    let mut entry = TileEntry::with_holes(tile, holes);
                    entry.not_rw = not_rw;
                    self.mapped.insert(t, entry);
                    early_add(&mut self.runs, t, t + TILE);
                    if reused {
                        self.to_zero.insert((t, TILE));
                    }
                    self.pending.push(t);
                    new = true;
                }
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        if new || unparked {
            MAPPED_GEN.fetch_add(1, Ordering::Release);
        }
        if self.pending.len() >= PENDING_MAX {
            self.flush_pending();
        } else if !self.pending.is_empty() {
            pending_soon();
        }
    }

    /// The pending tiles (`pending`) mapped onto their pool tiles, one call per window, then the
    /// fence signalled (title queues wait for it in `before_execute`). An address decommitted
    /// since has no `mapped` entry and is skipped; one committed again maps its current tile. One
    /// decommitted and parked (or kept for a placed buffer) stays pending: its mapping was never
    /// sent, and a commit that takes it back from there sends none.
    unsafe fn flush_pending(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let mut addrs = std::mem::take(&mut self.pending);
        addrs.sort_unstable();
        addrs.dedup();
        let mut by_window: FxHashMap<usize, Vec<(usize, (u32, u32))>> = FxHashMap::default();
        for t in addrs {
            if let (Some(e), Some(wi)) = (self.mapped.get(&t).filter(|e| e.pool != LAZY), self.window_index(t)) {
                by_window.entry(wi).or_default().push((t, e.pool));
            } else if self.parked.contains_key(&t) || self.kept.contains_key(&t) {
                self.pending.push(t);
            }
        }
        if by_window.is_empty() {
            return;
        }
        for (wi, tiles) in by_window {
            let (b, r) = (self.windows[wi].0, self.windows[wi].1.clone());
            TILE_WORK[0].fetch_add(tiles.len() as u64, Ordering::Relaxed);
            TILE_WORK[1].fetch_add(1, Ordering::Relaxed);
            self.map_tiles(&r, b, &tiles);
        }
        self.signal();
    }

    #[inline]
    unsafe fn signal(&mut self) {
        self.value += 1;
        let _ = super::real::native_queue_signal(&self.queue, &self.fence, self.value);
    }

    #[inline]
    unsafe fn wait_for(&self, target: u64) -> bool {
        self.wait_for_ms(target, 5000)
    }

    unsafe fn wait_for_ms(&self, target: u64, timeout_ms: u32) -> bool {
        if target == 0 || super::real::fence_completed(&self.fence) >= target {
            return true;
        }
        // The event is one for all waits here, and a wait that gave up leaves its registration
        // behind: that one sets the event later, and the next wait would return at once with the
        // queue not there (10 such "after 5 s" lines within 100 ms in one run, each followed by a
        // Reset of an allocator the GPU was still in). So: until the fence says so, or `timeout_ms`.
        let t0 = std::time::Instant::now();
        let wait_stall = crate::stall::span(crate::stall::StallKind::GpuWait);
        loop {
            let _ = self.fence.SetEventOnCompletion(target, self.event);
            let left = (timeout_ms as u64).saturating_sub(t0.elapsed().as_millis() as u64).max(1) as u32;
            windows::Win32::System::Threading::WaitForSingleObject(self.event, left);
            if super::real::fence_completed(&self.fence) >= target {
                return true;
            }
            if t0.elapsed().as_millis() >= timeout_ms as u128 {
                break;
            }
        }
        drop(wait_stall);
        crate::xlog_warn!("sparse: the queue has not reached fence value {target} after {timeout_ms} ms (at {})", super::real::fence_completed(&self.fence));
        false
    }

    #[inline]
    unsafe fn reclaim_ring(&mut self) {
        if self.inflight.is_empty() {
            return;
        }
        let completed = super::real::fence_completed(&self.fence);
        while let Some(&(fv, end_pos)) = self.inflight.front() {
            if fv <= completed {
                self.ring_free = end_pos;
                self.inflight.pop_front();
            } else {
                break;
            }
        }
    }

    /// Acquire the next upload command slot, or return a ticket to wait for without `STATE`.
    unsafe fn begin_slot(&mut self) -> BeginSlot {
        let next_slot = self.next_slot;
        let completed = super::real::fence_completed(&self.fence);
        let mut waiting = None;
        let mut ready = None;
        for offset in 0..self.slots.len() {
            let idx = (next_slot + offset) % self.slots.len();
            let fv = self.slots[idx].fence_val;
            if fv == 0 || fv <= completed {
                ready = Some(idx);
                break;
            }
            if waiting.is_none_or(|(_, earliest)| fv < earliest) {
                waiting = Some((idx, fv));
            }
        }
        let Some(idx) = ready else {
            let (idx, fv) = waiting.expect("sparse upload slots are not empty");
            return BeginSlot::Wait(self.wait_ticket(fv, WaitKind::Slot { index: idx, next_slot, fence: fv }));
        };
        let slot = &self.slots[idx];
        if slot.alloc.Reset().is_err() || slot.list.Reset(&slot.alloc, None).is_err() {
            return BeginSlot::RetryRequired;
        }
        self.next_slot = (idx + 1) % self.slots.len();
        BeginSlot::Ready(idx)
    }

    fn wait_ticket(&self, target: u64, kind: WaitKind) -> WaitTicket {
        WaitTicket {
            state_generation: self.generation,
            mapping_epoch: self.mapping_epoch,
            dev: self.dev.clone(),
            fence: self.fence.clone(),
            target,
            kind,
        }
    }

    fn wait_ticket_owns_upload(&self, ticket: &WaitTicket) -> bool {
        if !self.wait_ticket_generation(ticket) {
            return false;
        }
        match ticket.kind {
            WaitKind::Slot { index, next_slot, fence } => {
                self.next_slot == next_slot && self.slots.get(index).is_some_and(|slot| slot.fence_val == fence)
            }
            WaitKind::Ring { front, write, submitted, free } => {
                self.inflight.front().copied() == Some(front)
                    && self.ring_write == write && self.ring_submitted == submitted && self.ring_free == free
            }
        }
    }

    fn wait_ticket_generation(&self, ticket: &WaitTicket) -> bool {
        self.generation == ticket.state_generation
            && self.fence.as_raw() == ticket.fence.as_raw()
            && self.uploads_ordered
    }

    fn wait_ticket_valid(&self, ticket: &WaitTicket) -> bool {
        self.mapping_epoch == ticket.mapping_epoch && self.wait_ticket_owns_upload(ticket)
    }

    /// Close and submit upload slot `idx` asynchronously (NO CPU wait — callers order GPU work
    /// via `queue.Wait(&self.fence, self.value)`).
    unsafe fn submit_slot(&mut self, idx: usize) -> SubmitResult {
        // the copies on `queue` see every tile mapped (`pending` goes first on the same queue)
        self.flush_pending();
        if self.slots[idx].list.Close().is_err() {
            return SubmitResult::NotSubmitted;
        }
        let Ok(l) = self.slots[idx].list.cast::<ID3D12CommandList>() else {
            return SubmitResult::NotSubmitted;
        };
        super::title_textures::own_work(|| {
            super::real::native_queue_execute(&self.queue, &l);
        });
        let value = self.value + 1;
        if super::real::native_queue_signal(&self.queue, &self.fence, value).is_err() {
            self.uploads_ordered = false;
            self.slots[idx].fence_val = u64::MAX;
            return SubmitResult::Unordered;
        }
        self.value = value;
        self.slots[idx].fence_val = value;
        if self.ring_write > self.ring_submitted {
            self.inflight.push_back((value, self.ring_write));
            self.ring_submitted = self.ring_write;
        }
        SubmitResult::Submitted
    }

    unsafe fn staging_ptr(&mut self) -> Option<*mut u8> {
        if self.staging.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(STAGING as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut r,
                )
                .ok()?;
            let r = r?;
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).ok()?;
            self.staging = Some((r, p as *mut u8));
        }
        self.staging.as_ref().map(|&(_, p)| p)
    }

    unsafe fn ensure_zero_buf(&mut self) -> bool {
        if self.zero_buf.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(ZERO_BUF as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            if self
                .dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut r,
                )
                .is_err()
            {
                return false;
            }
            let Some(r) = r else { return false };
            let mut p = std::ptr::null_mut();
            if r.Map(0, None, Some(&mut p)).is_err() {
                return false;
            }
            std::ptr::write_bytes(p as *mut u8, 0, ZERO_BUF);
            r.Unmap(0, None);
            self.zero_buf = Some(r);
        }
        true
    }

    unsafe fn readback_buf(&mut self) -> Option<(ID3D12Resource, *mut u8)> {
        if self.readback.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(READBACK_STAGING as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut r,
                )
                .ok()?;
            let r = r?;
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).ok()?;
            self.readback = Some((r, p as *mut u8));
        }
        self.readback.clone()
    }

    /// Clear `sent` hashes for every page overlapping `[a, a + n)`.
    fn clear_sent_range(&mut self, a: usize, n: usize) {
        let mut pg = a & !(PAGE - 1);
        let end = a + n;
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = (t + TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    entry.sent[(p - t) / PAGE] = None;
                    p += PAGE;
                }
            }
            pg = tile_end;
        }
    }

    /// Update `sent` hashes for all 64-byte chunks overlapping `[va, va + len)` so the CPU
    /// copy-back we just wrote to `va` is not re-uploaded by the next `before_execute`.
    unsafe fn sync_sent_range(&mut self, va: usize, len: usize) {
        if len == 0 {
            return;
        }
        let mut pg = va & !(PAGE - 1);
        let end = va + len;
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = (t + TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let cpu = entry.cpu(t);
                let mut p = pg;
                while p < tile_end {
                    let idx = (p - t) / PAGE;
                    if entry.holes & (1u16 << idx) == 0 {
                        let bytes = std::slice::from_raw_parts((cpu + (p - t)) as *const u8, PAGE);
                        let sent = entry.sent[idx].get_or_insert_with(|| Box::new(ZERO_HASHES));
                        let lo = va.saturating_sub(p) / CHUNK;
                        let hi = (end.min(p + PAGE) - p).div_ceil(CHUNK).min(CHUNKS);
                        for k in lo..hi {
                            sent[k] = chunk_hash(&bytes[k * CHUNK..(k + 1) * CHUNK]);
                        }
                    }
                    p += PAGE;
                }
            }
            pg = tile_end;
        }
    }

    fn publish_hashes(&mut self, updates: &[HashUpdate]) {
        publish_hash_updates(&mut self.mapped, updates);
    }

    /// Submit one exact dirty batch. Until submission succeeds its ring bytes and hash plan are
    /// provisional; a close/cast failure rolls the ring cursor back to this batch's start.
    unsafe fn submit_dirty_batch(
        &mut self,
        mut batch: PreparedDirtyBatch,
        reclaim: bool,
    ) -> Result<PreparedDirtyBatch, PreparationOutcome> {
        match self.submit_slot(batch.slot) {
            SubmitResult::Submitted => {
                self.publish_hashes(&batch.updates);
                batch.clear_after_submit();
                if reclaim {
                    self.reclaim_ring();
                }
                Ok(batch)
            }
            SubmitResult::NotSubmitted => {
                self.ring_write = batch.ring_start;
                Err(PreparationOutcome::RetryRequired)
            }
            SubmitResult::Unordered => Err(PreparationOutcome::ExecutedUntracked),
        }
    }

    /// Whether nobody wrote the 64-byte chunk holding `va` since it was last sent to the GPU (its
    /// page has video memory and its hash is the one known).
    unsafe fn chunk_clean(&self, va: usize) -> bool {
        let pg = va & !(PAGE - 1);
        let t = pg & !(TILE - 1);
        let idx = (pg - t) / PAGE;
        let Some(entry) = self.mapped.get(&t).filter(|e| e.pool != LAZY) else { return false };
        if entry.holes & (1u16 << idx) != 0 {
            return false;
        }
        let page = std::slice::from_raw_parts((entry.cpu(t) + (pg - t)) as *const u8, PAGE);
        let k = (va - pg) / CHUNK;
        let h = chunk_hash(&page[k * CHUNK..(k + 1) * CHUNK]);
        entry.sent[idx].as_deref().map_or(h == ZERO_HASH, |sent| h == sent[k])
    }

    /// `len` bytes from readback memory `p + off` into the title's `dst` (committed pages only, in
    /// one tile), a window address: they land at its tile's CPU address, if the tile's memory is
    /// what it was when the copy was recorded (`gen`, `TileEntry::gen`). One shown over another
    /// CPU view since, or with a page decommitted (and maybe committed again), is left out: what
    /// the GPU had there before is not that memory's. (A tile of the window's own lands as before.)
    /// `gpu_written`: the bytes are what the GPU's commands wrote (`land_page`): they land in a
    /// chunk the CPU changed too.
    unsafe fn copy_back(&mut self, p: *mut u8, dst: usize, off: usize, len: usize, gen: u32, gpu_written: bool) {
        let mut at = dst;
        let mut src_off = off;
        let end = dst + len;
        while at < end {
            let pg = at & !(PAGE - 1);
            let t = pg & !(TILE - 1);
            let idx = (pg - t) / PAGE;
            let step = (pg + PAGE).min(end) - at;
            // (a tile with no video memory holds nothing the GPU wrote)
            if let Some(entry) = self.mapped.get_mut(&t).filter(|e| e.pool != LAZY) {
                let here = entry.cpu(t);
                if entry.holes & (1u16 << idx) == 0 && entry.gen == gen {
                    let page = std::slice::from_raw_parts_mut((here + (pg - t)) as *mut u8, PAGE);
                    let source = std::slice::from_raw_parts(p.add(src_off), step);
                    let wrote = land_page(&mut entry.sent[idx], page, at - pg, source, gpu_written);
                    if wrote && self.own_pages.len() < OWN_PAGES_MAX && super::look_stats::deep() {
                        self.own_pages.push(here + (pg - t));
                    }
                }
            }
            at += step;
            src_off += step;
        }
    }

    /// An immediate the GPU wrote at `va` (`bytes`: 4 or 8), into the title's memory there (at the
    /// tile's CPU address), if it is the memory it was when the immediate was recorded (`gen`, as
    /// in `copy_back`)
    unsafe fn store_imm(&mut self, va: usize, value: u64, bytes: u8, gen: u32) {
        let pg = va & !(PAGE - 1);
        let t = pg & !(TILE - 1);
        let Some(cpu) = self.mapped.get(&t).filter(|e| e.holes & (1u16 << ((pg - t) / PAGE)) == 0 && e.gen == gen).map(|e| e.cpu(t) + (va - t)) else { return };
        // (a chunk the title's CPU wrote since it was last sent keeps its state: it goes up with the
        // next look, with this value in it)
        let clean = self.chunk_clean(va);
        match bytes & 0x7f {
            4 => (*(cpu as *const std::sync::atomic::AtomicU32)).store(value as u32, Ordering::Release),
            8 if bytes & IMM_KEEP_MAX != 0 => {
                if (*(cpu as *const AtomicU64)).fetch_max(value, Ordering::AcqRel) > value {
                    static LATE: AtomicU64 = AtomicU64::new(0);
                    let k = LATE.fetch_add(1, Ordering::Relaxed) + 1;
                    if k.is_power_of_two() {
                        crate::xlog_info!("sparse: {k} immediates so far left out – their cell is a counter that held a later value ({value:#x} for {va:#x})");
                    }
                    return;
                }
            }
            8 => (*(cpu as *const AtomicU64)).store(value, Ordering::Release),
            _ => return,
        }
        if self.own_pages.len() < OWN_PAGES_MAX && super::look_stats::deep() {
            self.own_pages.push(cpu & !(PAGE - 1));
        }
        if clean {
            self.sync_sent_range(va, (bytes & 0x7f) as usize);
        }
    }

    /// A copy back's list and a readback buffer of at least `need` bytes (reused when one fits).
    #[cfg(test)]
    unsafe fn rb_job(&self, need: usize, ty: D3D12_COMMAND_LIST_TYPE) -> Option<RbJob> {
        take_or_alloc_rb_job(&self.dev, &self.readbacks, need, ty)
    }

    /// Pop every finished `imm_job` and `rb_job` from `self.readbacks` in strict ascending `seq`
    /// order, land them into CPU memory under `&mut self` (`STATE`), recycle finished `RbJob`s up
    /// to cap 32, and return any excess `RbJob`s so the caller drops them outside all locks.
    unsafe fn land_finished_readbacks(&mut self) -> Vec<RbJob> {
        let ready: Vec<ReadyJob> = {
            let mut rq = self.readbacks.lock().unwrap();
            let rbq_done = rq.rbq_done.clone();
            let mut out = Vec::new();
            let mut i = 0;
            let mut k = 0;
            loop {
                // the immediates of submissions before the next copy back first
                while k < rq.imm_jobs.len() && rq.rb_jobs.get(i).is_none_or(|j| rq.imm_jobs[k].seq < j.seq) {
                    if super::real::fence_completed(&rq.imm_jobs[k].fence) < rq.imm_jobs[k].value {
                        k += 1;
                        continue;
                    }
                    let Some(job) = rq.imm_jobs.remove(k) else { break };
                    out.push(ReadyJob::Imm(job));
                }
                if i >= rq.rb_jobs.len() {
                    break;
                }
                let done = {
                    let j = &rq.rb_jobs[i];
                    match (&j.fences, &rbq_done) {
                        (Some((_, b)), _) => super::real::fence_completed(b) >= j.done_v,
                        (None, Some(d)) => super::real::fence_completed(d) >= j.done_v,
                        (None, None) => false,
                    }
                };
                if !done {
                    i += 1;
                    continue;
                }
                let Some(job) = rq.rb_jobs.remove(i) else { break };
                out.push(ReadyJob::Rb(job));
            }
            if !out.is_empty() {
                rq.landing += out.len();
            }
            out
        };
        if ready.is_empty() {
            return Vec::new();
        }
        let landed_n = ready.len();
        let mut recycled: Vec<RbJob> = Vec::new();
        for item in ready {
            match item {
                ReadyJob::Imm(job) => {
                    for &(va, value, bytes, gen) in &job.writes {
                        self.store_imm(va, value, bytes, gen);
                    }
                    RB_PENDING.fetch_sub(1, Ordering::Release);
                }
                ReadyJob::Rb(mut job) => {
                    let copies = std::mem::take(&mut job.copies);
                    for &(dst, off, len, gen) in &copies {
                        // (the lists' own GPU writes, `readback_after_execute`)
                        self.copy_back(job.ptr, dst, off, len, gen, true);
                    }
                    TILE_WORK[3].fetch_add(copies.iter().map(|c| c.2 as u64).sum(), Ordering::Relaxed);
                    RB_PENDING.fetch_sub(1, Ordering::Release);
                    job.copies = copies;
                    job.copies.clear();
                    recycled.push(job);
                }
            }
        }
        let mut excess = Vec::new();
        {
            let mut rq = self.readbacks.lock().unwrap();
            rq.landing -= landed_n;
            for job in recycled {
                // (a burst of copies back does not keep its buffers for good: each is up to 16 MiB of
                // host memory; excess jobs drop outside all locks)
                if rq.rb_pool.len() < 32 {
                    rq.rb_pool.push(job);
                } else {
                    excess.push(job);
                }
            }
        }
        excess
    }

    /// Land submitted copies back in the title's memory as far as `how` says (test helper on a
    /// standalone `State` without global `STATE`).
    #[cfg(test)]
    unsafe fn drain_readbacks(&mut self, how: Drain) -> Vec<RbJob> {
        self.drain_readbacks_spin(how, 2000)
    }

    #[cfg(test)]
    unsafe fn drain_readbacks_spin(&mut self, how: Drain, spin_us: u128) -> Vec<RbJob> {
        if wait_and_check_ready(&self.readbacks, how, spin_us, None) != ReadyCheck::LandNow {
            return Vec::new();
        }
        self.land_finished_readbacks()
    }

    /// Copy `ranges` (address, bytes) from sparse VRAM back into the CPU view and refresh
    /// `sent` hashes. Caller has already ordered `self.queue` after any GPU writes if needed.
    /// `gpu_written`: the ranges are bytes the GPU's commands wrote (`land_page`).
    /// Returns any excess recycled `RbJob`s for the caller to drop outside `STATE`.
    unsafe fn readback_ranges(&mut self, ranges: &[(usize, usize)], gpu_written: bool) -> Vec<RbJob> {
        self.flush_pending();
        // copies back of earlier submissions first: their data is older than this one's
        let excess = self.land_finished_readbacks();
        let Some((rb, p)) = self.readback_buf() else { return excess };
        let mut pending: Vec<(usize, usize, usize, u32)> = Vec::new(); // (va, rb_off, chunk, gen of its tile)
        let mut used = 0usize;
        let mut open = false;
        let flush_and_copy = |s: &mut State, open: &mut bool, pending: &mut Vec<(usize, usize, usize, u32)>, used: &mut usize, p: *mut u8| -> bool {
            if *open {
                *open = false;
                let Ok(()) = s.rb_list.Close() else { return false };
                let Ok(l) = s.rb_list.cast::<ID3D12CommandList>() else { return false };
                super::title_textures::own_work(|| {
                    super::real::native_queue_execute(&s.queue, &l);
                });
                s.signal();
                s.rb_last_fence = s.value;
                if !s.wait_for(s.value) {
                    pending.clear();
                    *used = 0;
                    return false;
                }
                s.reclaim_ring();
            }
            for &(dst, off, len, gen) in pending.iter() {
                s.copy_back(p, dst, off, len, gen, gpu_written);
            }
            pending.clear();
            *used = 0;
            true
        };
        for &(a, n) in ranges {
            if n == 0 {
                continue;
            }
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let tile = cur & !(TILE - 1);
                let Some(wi) = self.window_index(cur) else { break };
                let b = self.windows[wi].0;
                let to_tile_end = (tile + TILE) - cur;
                // (a tile with no video memory has nothing to copy back: the GPU's writes there went
                // nowhere, and a tile it got now would hold what another address left in it)
                let Some(gen) = self.mapped.get(&tile).filter(|e| e.pool != LAZY).map(|e| e.gen) else {
                    done += (n - done).min(to_tile_end);
                    continue;
                };
                if used == READBACK_STAGING {
                    if !flush_and_copy(self, &mut open, &mut pending, &mut used, p) {
                        return excess;
                    }
                }
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let chunk = (n - done).min(READBACK_STAGING - used).min(to_tile_end).min(to_win_end);
                if !open {
                    if !self.wait_for(self.rb_last_fence) || self.rb_alloc.Reset().is_err() || self.rb_list.Reset(&self.rb_alloc, None).is_err() {
                        return excess;
                    }
                    open = true;
                }
                self.rb_list.CopyBufferRegion(&rb, used as u64, &self.windows[wi].1, (cur - b) as u64, chunk as u64);
                pending.push((cur, used, chunk, gen));
                used += chunk;
                done += chunk;
            }
        }
        if open {
            let _ = flush_and_copy(self, &mut open, &mut pending, &mut used, p);
        }
        excess
    }

    /// Asynchronously zero `runs` (address, bytes) in sparse VRAM using the pre-zeroed `zero_buf`
    /// (zero staging ring bytes consumed, zero CPU stall).
    unsafe fn upload_zeros(&mut self, runs: &[(usize, usize)]) -> UploadStep {
        // (a tile with no video memory reads zeros already; a run goes on into its neighbours)
        let kept: Vec<(usize, usize)>;
        let runs = if lazy_commits() {
            kept = runs_where(runs, |t| self.mapped.get(&t).is_none_or(|e| e.pool != LAZY));
            &kept[..]
        } else {
            runs
        };
        if runs.is_empty() {
            return UploadStep::Done(PreparationOutcome::Ready);
        }
        if !self.uploads_ordered {
            return UploadStep::Done(PreparationOutcome::ExecutedUntracked);
        }
        if !self.ensure_zero_buf() {
            return UploadStep::Done(PreparationOutcome::RetryRequired);
        }
        let slot_idx = match self.begin_slot() {
            BeginSlot::Ready(index) => index,
            BeginSlot::Wait(ticket) => return UploadStep::Wait(ticket),
            BeginSlot::RetryRequired => return UploadStep::Done(PreparationOutcome::RetryRequired),
        };
        let mut last_wi: Option<usize> = None;
        let zero_buf = self.zero_buf.as_ref().unwrap();
        let list = &self.slots[slot_idx].list;
        for &(a, n) in runs {
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let wi = match last_wi {
                    Some(i) if cur >= self.windows[i].0 && cur < self.windows[i].0 + crate::gpu::uma::WINDOW => i,
                    _ => match self.window_index(cur) {
                        Some(i) => {
                            last_wi = Some(i);
                            i
                        }
                        None => break,
                    },
                };
                let b = self.windows[wi].0;
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let chunk = (n - done).min(ZERO_BUF).min(to_win_end);
                list.CopyBufferRegion(&self.windows[wi].1, (cur - b) as u64, zero_buf, 0, chunk as u64);
                done += chunk;
            }
        }
        UploadStep::Done(self.submit_slot(slot_idx).into())
    }

    /// Asynchronously copy `runs` (address, bytes) from the CPU view into sparse VRAM tiles via
    /// the 64 MiB ring staging buffer (zero CPU stall on the common path).
    unsafe fn upload_dirty(&mut self, runs: &[(usize, usize)]) -> UploadStep {
        self.upload_dirty_with_timeout(runs, 5000)
    }

    unsafe fn upload_dirty_with_timeout(&mut self, runs: &[(usize, usize)], _wait_ms: u32) -> UploadStep {
        if runs.is_empty() {
            return UploadStep::Done(PreparationOutcome::Ready);
        }
        if !self.uploads_ordered {
            return UploadStep::Done(PreparationOutcome::ExecutedUntracked);
        }
        // the CPU's first bytes for a tile listed with no video memory: it gets it now, and its
        // mapping is sent before the copies that fill it. One used before is cleared first, here
        // and not at the next submit (`to_zero`), which would clear these bytes with it
        let kept: Vec<(usize, usize)>;
        let mut runs = runs;
        if lazy_commits() {
            let mut reused = Vec::new();
            let (mut got, mut left) = (0, 0);
            for &(a, n) in runs {
                let (g, l) = self.materialize(a, n, false, &mut reused);
                got += g;
                left += l;
            }
            if got > 0 {
                self.flush_pending();
                match self.upload_zeros(&reused) {
                    UploadStep::Done(PreparationOutcome::Ready) => {}
                    step => {
                        // not cleared now: cleared before anything else at the next try
                        // (`to_zero`), and all of their bytes go up after that (no hashes known)
                        for &(t, n) in &reused {
                            self.clear_sent_range(t, n);
                        }
                        self.to_zero.extend(reused.iter().copied());
                        return step;
                    }
                }
            }
            // none to be had: the bytes are not sent, whatever the scan noted, and all of the tile
            // goes up once there is some (`starved`)
            if left > 0 {
                let mut starved: Vec<usize> = Vec::new();
                kept = runs_where(runs, |t| {
                    let has = self.mapped.get(&t).is_none_or(|e| e.pool != LAZY);
                    if !has && starved.last() != Some(&t) {
                        starved.push(t);
                    }
                    has
                });
                for &t in &starved {
                    self.clear_sent_range(t, TILE);
                }
                self.starved.extend(starved);
                runs = &kept[..];
                if runs.is_empty() {
                    return UploadStep::Done(PreparationOutcome::Ready);
                }
            }
        }
        let Some(p) = self.staging_ptr() else {
            return UploadStep::Done(PreparationOutcome::RetryRequired);
        };
        // the open upload: its slot, where its ring bytes start, and the hashes it publishes once
        // submitted (`submit_upload`); a submitted one's vector is used again
        let mut batch: Option<PreparedDirtyBatch> = None;
        let mut recycled_batch: Option<PreparedDirtyBatch> = None;
        // what the open upload's copies take (window address, CPU address, bytes): their chunks'
        // hashes, published once it is submitted (`submit_upload`)
        let mut staged: Vec<(usize, usize, usize)> = Vec::new();
        let mut last_wi: Option<usize> = None;
        // the copies into the ring (source address, ring offset, bytes), done before their slot is
        // submitted, by the worker threads when there are many
        let mut jobs: Vec<(usize, usize, usize)> = Vec::new();
        // the many small ones first, together (the windows the open slot's dispatches write are
        // UAVs in it until their first copy)
        let mut uavs: Vec<usize> = Vec::new();
        let single = match self.scatter_small(runs, p, &mut batch, &mut recycled_batch, &mut jobs, &mut staged, &mut uavs) {
            Ok(single) => single,
            Err(step) => return step,
        };
        for &(a, n) in single.as_deref().unwrap_or(runs) {
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let wi = match last_wi {
                    Some(i) if cur >= self.windows[i].0 && cur < self.windows[i].0 + crate::gpu::uma::WINDOW => i,
                    _ => match self.window_index(cur) {
                        Some(i) => {
                            last_wi = Some(i);
                            i
                        }
                        None => {
                            if let Some(batch) = batch {
                                self.ring_write = batch.ring_start;
                            }
                            return UploadStep::Done(PreparationOutcome::RetryRequired);
                        }
                    },
                };
                let b = self.windows[wi].0;
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let (src, len) = self.cpu_span(cur, (n - done).min(to_win_end));
                let copy = prepare_dirty_copy(cur, len, b, crate::gpu::uma::WINDOW, self.ring_write, STAGING);
                if let Err(step) = self.ring_room(copy.ring_start + copy.bytes as u64, p, &mut batch, &mut recycled_batch, &mut jobs, &mut staged) {
                    return step;
                }
                let slot_idx = match batch.as_ref() {
                    Some(batch) => batch.slot,
                    None => {
                        uavs.clear();
                        let idx = match self.begin_slot() {
                            BeginSlot::Ready(index) => index,
                            BeginSlot::Wait(ticket) => return UploadStep::Wait(ticket),
                            BeginSlot::RetryRequired => return UploadStep::Done(PreparationOutcome::RetryRequired),
                        };
                        batch = Some(match recycled_batch.take() {
                            Some(recycled) => recycled.reopen(idx, self.ring_write),
                            None => PreparedDirtyBatch::new(idx, self.ring_write),
                        });
                        idx
                    }
                };
                if let Some(k) = uavs.iter().position(|&w| w == wi) {
                    uavs.swap_remove(k);
                    self.slots[slot_idx].list.ResourceBarrier(&[uav_to_copy(&self.windows[wi].1)]);
                }
                jobs.push((src, copy.staging_offset, copy.bytes));
                staged.push((cur, src, copy.bytes));
                let staging_res = &self.staging.as_ref().unwrap().0;
                self.slots[slot_idx].list.CopyBufferRegion(
                    &self.windows[wi].1,
                    copy.window_offset as u64,
                    staging_res,
                    copy.staging_offset as u64,
                    copy.bytes as u64,
                );
                self.ring_write = copy.ring_start + copy.bytes as u64;
                done += copy.bytes;
            }
        }
        if let Some(batch) = batch {
            if let Err(outcome) = self.submit_upload(batch, &mut jobs, &mut staged, p, false) {
                return UploadStep::Done(outcome);
            }
        }
        UploadStep::Done(PreparationOutcome::Ready)
    }

    /// Room in the upload ring up to its position `end`: what the GPU is done with is taken back;
    /// when that is not enough, the open upload (`batch`) is submitted (a submitted one's vector
    /// goes to `recycled`). Still not enough: a ticket for the oldest upload in the ring, which the
    /// caller waits for with the state's lock and the upload gate given up, then starts again.
    unsafe fn ring_room(
        &mut self,
        end: u64,
        p: *mut u8,
        batch: &mut Option<PreparedDirtyBatch>,
        recycled: &mut Option<PreparedDirtyBatch>,
        jobs: &mut Vec<(usize, usize, usize)>,
        staged: &mut Vec<(usize, usize, usize)>,
    ) -> Result<(), UploadStep> {
        if end.saturating_sub(self.ring_free) <= STAGING as u64 {
            return Ok(());
        }
        self.reclaim_ring();
        if end.saturating_sub(self.ring_free) <= STAGING as u64 {
            return Ok(());
        }
        if let Some(ready) = batch.take() {
            match self.submit_upload(ready, jobs, staged, p, true) {
                Ok(done) => *recycled = Some(done),
                Err(outcome) => return Err(UploadStep::Done(outcome)),
            }
            if end.saturating_sub(self.ring_free) <= STAGING as u64 {
                return Ok(());
            }
        }
        let Some(&front) = self.inflight.front() else {
            crate::xlog_warn!("sparse: the staging ring is full without a retirement fence; the unsubmitted CPU writes will be retried");
            return Err(UploadStep::Done(PreparationOutcome::RetryRequired));
        };
        Err(UploadStep::Wait(self.wait_ticket(
            front.0,
            WaitKind::Ring { front, write: self.ring_write, submitted: self.ring_submitted, free: self.ring_free },
        )))
    }

    /// Submit the open upload `batch`: the hashes of the chunks its copies take (`staged`: window
    /// address, CPU address, bytes), read from the title's memory before the copies into the ring
    /// (`run_copies`; a CPU write in between leaves the chunk changed, and it goes up again), then
    /// the submit. They become `sent` only once it has its fence (`submit_dirty_batch`). A copy's
    /// partial chunks at either end get none: they stay changed and go up with the next look.
    unsafe fn submit_upload(
        &mut self,
        mut batch: PreparedDirtyBatch,
        jobs: &mut Vec<(usize, usize, usize)>,
        staged: &mut Vec<(usize, usize, usize)>,
        p: *mut u8,
        reclaim: bool,
    ) -> Result<PreparedDirtyBatch, PreparationOutcome> {
        for (at, src, n) in staged.drain(..) {
            let mut c = at.next_multiple_of(CHUNK);
            let hi = (at + n) & !(CHUNK - 1);
            while c < hi {
                let bytes = std::slice::from_raw_parts((src + (c - at)) as *const u8, CHUNK);
                batch.updates.push(HashUpdate { at: c, hash: chunk_hash(bytes) });
                c += CHUNK;
            }
        }
        run_copies(jobs, p);
        self.submit_dirty_batch(batch, reclaim)
    }

    /// Where the CPU bytes of window addresses `[a, a + n)` are: (CPU address of `a`, bytes from
    /// `a` that lie in one piece there). The piece ends where a tile's CPU address does not follow
    /// the one before it: a GPU view shown over another part of its CPU view, or a tile of the
    /// window's own next to one over a CPU view (whose own address the CPU cannot read).
    fn cpu_span(&self, a: usize, n: usize) -> (usize, usize) {
        if self.mirrors.is_empty() {
            return (a, n);
        }
        let src = |t: usize| self.mapped.get(&t).map_or(t, |e| e.cpu(t));
        let t0 = a & !(TILE - 1);
        let s0 = src(t0);
        let mut end = (t0 + TILE).min(a + n);
        while end < a + n && src(end) == s0 + (end - t0) {
            end = (end + TILE).min(a + n);
        }
        (s0 + (a - t0), end - a)
    }

    /// Record the small ones of `runs` as dispatches (`scatter`) that put many of them in their
    /// places at once, in the open upload (`batch`, opened here if none is): the runs left for a
    /// copy each (None: all of them). `uavs` are the windows the open slot's dispatches write.
    /// Err: what the caller returns (a ticket to wait for, or the outcome of a failed submit).
    #[allow(clippy::too_many_arguments)]
    unsafe fn scatter_small(
        &mut self,
        runs: &[(usize, usize)],
        p: *mut u8,
        batch: &mut Option<PreparedDirtyBatch>,
        recycled: &mut Option<PreparedDirtyBatch>,
        jobs: &mut Vec<(usize, usize, usize)>,
        staged: &mut Vec<(usize, usize, usize)>,
        uavs: &mut Vec<usize>,
    ) -> Result<Option<Vec<(usize, usize)>>, UploadStep> {
        if runs.len() < SCATTER_FROM || !super::scatter::on() {
            return Ok(None);
        }
        // (a run that ends in the next window is cut where the window ends, by the copies; one
        // whose CPU bytes are not in one piece is cut where they break, by the copies too)
        let small = |&(a, n): &(usize, usize)| n < SCATTER_RUN && a % CHUNK == 0 && n % CHUNK == 0 && self.window_index(a).is_some_and(|w| a + n <= self.windows[w].0 + crate::gpu::uma::WINDOW) && self.cpu_span(a, n).1 == n;
        let (small, single): (Vec<_>, Vec<_>) = runs.iter().copied().partition(small);
        if small.len() < SCATTER_FROM {
            return Ok(None);
        }
        let Some(pass) = super::scatter::of(&self.dev) else { return Ok(None) };
        let Some(ring_va) = self.staging.as_ref().map(|s| s.0.GetGPUVirtualAddress()) else { return Ok(None) };
        let pieces = super::scatter::pieces(&small, SCATTER_CHUNKS, |a| self.window_index(a).map(|w| (w, self.windows[w].0)));
        let mut bound = None;
        for piece in &pieces {
            // the table of the piece and its chunks in one part of the ring
            let bytes = super::scatter::Scatter::bytes(piece.chunks);
            let mut at = self.ring_write.next_multiple_of(RING_ALIGN as u64);
            let to_end = STAGING - at as usize % STAGING;
            if bytes > to_end {
                at += to_end as u64;
            }
            self.ring_room(at + bytes as u64, p, batch, recycled, jobs, staged)?;
            let slot_idx = match batch.as_ref() {
                Some(open) => open.slot,
                None => {
                    uavs.clear();
                    let idx = match self.begin_slot() {
                        BeginSlot::Ready(index) => index,
                        BeginSlot::Wait(ticket) => return Err(UploadStep::Wait(ticket)),
                        BeginSlot::RetryRequired => return Err(UploadStep::Done(PreparationOutcome::RetryRequired)),
                    };
                    *batch = Some(match recycled.take() {
                        Some(done) => done.reopen(idx, self.ring_write),
                        None => PreparedDirtyBatch::new(idx, self.ring_write),
                    });
                    idx
                }
            };
            let list = self.slots[slot_idx].list.clone();
            if bound != Some((slot_idx, self.slots[slot_idx].fence_val)) {
                bound = Some((slot_idx, self.slots[slot_idx].fence_val));
                pass.bind(&list);
            }
            let off = at as usize % STAGING;
            let mut place = p.add(off) as *mut u32;
            let mut data = off + super::scatter::Scatter::data_at(piece.chunks);
            for &(a, n) in &small[piece.runs.clone()] {
                for k in 0..n / CHUNK {
                    place.write((a - piece.base + k * CHUNK) as u32);
                    place = place.add(1);
                }
                let src = self.cpu_span(a, n).0;
                jobs.push((src, data, n));
                staged.push((a, src, n));
                data += n;
            }
            let (base, buf) = &self.windows[piece.window];
            pass.dispatch(&list, piece.chunks, ring_va + off as u64, buf.GetGPUVirtualAddress() + (piece.base - base) as u64);
            if !uavs.contains(&piece.window) {
                uavs.push(piece.window);
            }
            self.ring_write = at + bytes as u64;
            SCATTERED[0].fetch_add(1, Ordering::Relaxed);
        }
        SCATTERED[1].fetch_add(small.len() as u64, Ordering::Relaxed);
        Ok(Some(single))
    }
}

/// dispatches of `scatter_small` so far, and the runs in them
static SCATTERED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// `r`, written by a dispatch of the list, becomes the destination of a copy.
fn uav_to_copy(r: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: the list keeps no reference and the window outlives it
                pResource: unsafe { std::mem::transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                StateAfter: D3D12_RESOURCE_STATE_COPY_DEST,
            }),
        },
    }
}

/// Allocate a zeroed, write-watched block in the sparse VRAM windows (never straddling a
/// 4 GiB window boundary when `size <= 4 GiB`). Commits the CPU pages and maps VRAM tiles
/// (or queues them until the D3D12 device exists). Returns `(address, rounded_size)`.
pub unsafe fn alloc(size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc(size, align)?;
    if VirtualAlloc(addr as *const c_void, rounded, MEM_COMMIT, PAGE_READWRITE).is_null() {
        u.sparse_release(addr);
        return None;
    }
    committed(addr, rounded);
    Some((addr, rounded))
}

/// Allocate a zeroed, write-watched block strictly inside sparse window `window` (e.g.
/// window 0 for 32-bit-addressed descriptor heaps). Returns `(address, rounded_size)`.
pub unsafe fn alloc_in(window: usize, size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc_in(window, size, align)?;
    if VirtualAlloc(addr as *const c_void, rounded, MEM_COMMIT, PAGE_READWRITE).is_null() {
        u.sparse_release(addr);
        return None;
    }
    committed(addr, rounded);
    Some((addr, rounded))
}

/// Release a sparse allocation from `alloc` / `alloc_in` (false: not a sparse allocation).
pub unsafe fn free(p: usize) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    if !u.is_sparse(p) {
        return false;
    }
    // decommitted before the address goes back: once released, another thread may get it,
    // commit it and write there
    let Some((start, len)) = u.sparse_reservation(p).filter(|&(start, _)| start == p) else { return false };
    decommitted(start, len);
    VirtualFree(p as *mut c_void, len, MEM_DECOMMIT);
    u.sparse_release(p).is_some()
}

/// After `queue.ExecuteCommandLists`: copy GPU-written sparse ranges (e.g. `Write32/64`,
/// `FillMemoryWith32BitValueX`, `CopyMemoryToMemoryX`) back to their CPU view so CPU reads
/// and spin-waits see the GPU writes without clobbering `sent` hashes.
/// The queue tile mappings go out on, once there is one. Answered without `STATE` (`QUEUE`):
/// a caller may be inside a D3D12 call that a holder of `STATE` waits for.
pub fn queue() -> Option<ID3D12CommandQueue> {
    QUEUE.lock().unwrap().clone()
}

/// Pop a pooled `RbJob` of at least `need` bytes for `ty` under the leaf `readbacks` lock, or
/// allocate a new one on `dev` outside both `STATE` and `readbacks`.
unsafe fn take_or_alloc_rb_job(dev: &ID3D12Device, readbacks: &SharedReadbacks, need: usize, ty: D3D12_COMMAND_LIST_TYPE) -> Option<RbJob> {
    {
        let mut rq = readbacks.lock().unwrap();
        if let Some(i) = rq.rb_pool.iter().enumerate().filter(|(_, j)| j.cap >= need && j.ty == ty).min_by_key(|(_, j)| j.cap).map(|(i, _)| i) {
            return Some(rq.rb_pool.swap_remove(i));
        }
    }
    let cap = need.max(64 << 10).next_power_of_two().min(READBACK_STAGING);
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).ok()?;
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).ok()?;
    list.Close().ok()?;
    let _ = list.SetName(windows::core::w!("xcl sparse readback list"));
    let mut d = buffer_desc(cap as u64);
    d.Flags = D3D12_RESOURCE_FLAG_NONE;
    let mut r: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r).ok()?;
    let buf = r?;
    let mut p: *mut c_void = std::ptr::null_mut();
    buf.Map(0, None, Some(&mut p)).ok()?;
    Some(RbJob { alloc, list, buf, ptr: p as *mut u8, cap, rb_v: 0, done_v: 0, copies: Vec::new(), ty, fences: None, seq: 0 })
}

/// Whether `rq` holds at least one finished unpopped `ImmJob` or `RbJob`.
unsafe fn rq_has_unpopped_finished(rq: &ReadbackQueue) -> bool {
    if rq.imm_jobs.iter().any(|j| super::real::fence_completed(&j.fence) >= j.value) {
        return true;
    }
    rq.rb_jobs.iter().any(|j| {
        let done_f = match (&j.fences, &rq.rbq_done) {
            (Some((_, b)), _) => b,
            (None, Some(d)) => d,
            (None, None) => return false,
        };
        super::real::fence_completed(done_f) >= j.done_v
    })
}

/// Whether any finished unpopped readback job exists (used by the background thread so it skips
/// `STATE` whenever nothing is ready to land or another thread is already landing).
unsafe fn has_unpopped_finished_jobs(rb: &SharedReadbacks) -> bool {
    let rq = rb.lock().unwrap();
    rq_has_unpopped_finished(&rq)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadyCheck {
    NothingReady,
    LandNow,
    TimedOut,
}

/// Wait outside `STATE` and outside `READBACKS` for any already-prepared `prepare_after_execute`
/// plan (`rq.submitting > 0`) to publish or cancel within `deadline`, and then for any
/// `Drain::TitleDone` readback job whose title lists are already done on the GPU (up to `spin_us`
/// total across all such jobs).
///
/// Returns `ReadyCheck::LandNow` iff at least one job is finished or currently being landed under
/// `STATE` (meaning the caller must acquire `STATE` to land or synchronize with an in-flight land),
/// `ReadyCheck::TimedOut` if `deadline` expired while a submission was still publishing, or
/// `ReadyCheck::NothingReady` if no work is ready to land.
unsafe fn wait_and_check_ready(rb: &SharedReadbacks, how: Drain, spin_us: u128, deadline: Option<std::time::Instant>) -> ReadyCheck {
    {
        let mut rq = rb.lock().unwrap();
        while rq.submitting > 0 {
            if let Some(dl) = deadline {
                let Some(rem) = dl.checked_duration_since(std::time::Instant::now()) else {
                    return ReadyCheck::TimedOut;
                };
                if rem.is_zero() {
                    return ReadyCheck::TimedOut;
                }
                let (g, res) = rb.published.wait_timeout(rq, rem).unwrap();
                rq = g;
                if rq.submitting > 0 && res.timed_out() {
                    return ReadyCheck::TimedOut;
                }
            } else {
                rq = rb.published.wait(rq).unwrap();
            }
        }
    }
    if how == Drain::TitleDone && spin_us > 0 {
        let to_wait: Vec<(ID3D12Fence, u64)> = {
            let rq = rb.lock().unwrap();
            let mut out = Vec::new();
            for j in &rq.rb_jobs {
                let (title_f, done_f, rb_v, done_v) = match (&j.fences, &rq.rbq_done) {
                    (Some((a, b)), _) => (a, b, j.rb_v, j.done_v),
                    (None, Some(d)) => (&rq.rb_fence, d, j.rb_v, j.done_v),
                    (None, None) => continue,
                };
                if super::real::fence_completed(done_f) < done_v && super::real::fence_completed(title_f) >= rb_v {
                    out.push((done_f.clone(), done_v));
                }
            }
            out
        };
        if !to_wait.is_empty() {
            // Single total spin budget across all title-done copies, completely outside all locks.
            let t0 = std::time::Instant::now();
            let mut timed_out = 0u64;
            for (done_f, done_v) in &to_wait {
                while super::real::fence_completed(done_f) < *done_v && t0.elapsed().as_micros() < spin_us {
                    std::thread::yield_now();
                }
                if super::real::fence_completed(done_f) < *done_v {
                    timed_out += 1;
                }
            }
            if timed_out > 0 {
                static LATE: AtomicU64 = AtomicU64::new(0);
                let prev = LATE.fetch_add(timed_out, Ordering::Relaxed);
                for n in 1..=timed_out {
                    let k = prev + n;
                    if k.is_power_of_two() {
                        crate::xlog_warn!("sparse: {k} copies back so far not finished after the title's work they follow – left for later");
                    }
                }
            }
        }
    }
    let rq = rb.lock().unwrap();
    if rq.landing > 0 || rq_has_unpopped_finished(&rq) {
        ReadyCheck::LandNow
    } else {
        ReadyCheck::NothingReady
    }
}

/// Record and submit async readback copies + fences on `queue` under `plan.rb_submit` (outside
/// `STATE`), then push the resulting `ImmJob` and `RbJob`s to `plan.readbacks` once both fences
/// have been signalled on `queue`.
unsafe fn submit_async_readback(queue: &ID3D12CommandQueue, plan: AsyncReadbackPlan) {
    submit_async_readback_inner(
        queue,
        plan,
        #[cfg(test)]
        None,
    );
}

#[cfg(test)]
type PublishHook<'a> = &'a dyn Fn(&ID3D12Fence, u64);

unsafe fn submit_async_readback_inner(
    queue: &ID3D12CommandQueue,
    plan: AsyncReadbackPlan,
    #[cfg(test)] before_publish: Option<PublishHook<'_>>,
) {
    let AsyncReadbackPlan { rb_submit, readbacks, windows, batches, imm, submitting } = plan;
    // Lock order: the queue's submit lease (`queue_submit::with`) first, then `rb_submit`. The
    // signals and executes below take the lease again (`real::native_queue_signal`, `pc_execute`),
    // which is re-entrant only for a thread that already holds it: every caller of
    // `readback_after_execute` must run inside `queue`'s lease (list_x's submit effects and
    // kickoff do), or one thread holds `rb_submit` waiting for the lease while the lease holder
    // waits for `rb_submit`.
    let mut sub = rb_submit.lock().unwrap();
    // on this queue: [the title's lists] Signal(first, v) [the copies] Signal(second, v)
    let key = queue.as_raw() as usize;
    if !sub.rb_queue_fences.contains_key(&key) {
        let (Ok(a), Ok(b)) = (sub.dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE), sub.dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE)) else {
            return;
        };
        sub.rb_queue_fences.insert(key, (a, b, 0));
    }
    let (fences, v) = {
        let e = sub.rb_queue_fences.get_mut(&key).expect("fences");
        e.2 += 1;
        ((e.0.clone(), e.1.clone()), e.2)
    };
    let _ = super::real::native_queue_signal(queue, &fences.0, v);

    let mut submitted: Vec<RbJob> = Vec::with_capacity(batches.len());
    if !batches.is_empty() {
        let ty = queue.GetDesc().Type;
        for batch in batches {
            let need = batch.iter().map(|c| c.len).sum::<usize>();
            let Some(mut job) = take_or_alloc_rb_job(&sub.dev, &readbacks, need, ty) else {
                crate::xlog_warn!("sparse: no readback buffer for a copy back of {need} bytes – left out");
                // (the Signal after the loop still goes out: what was recorded for it stands)
                break;
            };
            let _ = job.alloc.Reset();
            let _ = job.list.Reset(&job.alloc, None);
            let mut off = 0usize;
            for c in batch {
                let (base, ref res) = windows[c.win];
                job.list.CopyBufferRegion(&job.buf, off as u64, res, (c.va - base) as u64, c.len as u64);
                job.copies.push((c.va, off, c.len, c.gen));
                off += c.len;
            }
            // (the PC slot in the queue's ordered turn, `real::native_queue_execute`)
            if job.list.Close().is_ok() {
                if let Ok(list) = job.list.cast::<ID3D12CommandList>() {
                    super::real::native_queue_execute(queue, &list);
                }
            }
            job.rb_v = v;
            job.done_v = v;
            job.fences = Some(fences.clone());
            submitted.push(job);
        }
        let _ = super::real::native_queue_signal(queue, &fences.1, v);
    }

    #[cfg(test)]
    if let Some(hook) = before_publish {
        hook(&fences.0, v);
    }

    let mut rq = readbacks.lock().unwrap();
    let mut added = 0usize;
    if !imm.is_empty() {
        rq.job_seq += 1;
        let seq = rq.job_seq;
        rq.imm_jobs.push_back(ImmJob { seq, fence: fences.0.clone(), value: v, writes: imm });
        added += 1;
    }
    for mut job in submitted {
        rq.job_seq += 1;
        job.seq = rq.job_seq;
        rq.rb_jobs.push_back(job);
        added += 1;
    }
    submitting.finish(&mut rq, added);
    if added > 0 {
        RB_ASYNC.store(true, Ordering::Relaxed);
    }
    drop(rq);
    readbacks.published.notify_all();
}

unsafe fn readback_after_execute_with(state_lock: &Mutex<Option<State>>, queue: &ID3D12CommandQueue, ranges: &[(usize, usize)], imms: &[(usize, u64, u8)]) {
    if ranges.is_empty() && imms.is_empty() {
        return;
    }
    let is_global = std::ptr::eq(state_lock, &STATE);
    // (the `STATE` stage takes the upload gate first, as every uploader does: its rare synchronous
    // copy back uses the upload ring; the recording of an async copy back goes on outside both)
    let measure = is_global && super::memory_report::enabled();
    let (mut uploads, timing) = if is_global {
        let (guard, timing) = UploadGateTiming::acquire(measure);
        (guard, Some(timing))
    } else {
        (None, None)
    };
    let (plan, excess) = {
        let t0 = (is_global && super::look_stats::deep()).then(std::time::Instant::now);
        let mut g = if is_global { lock_for_submit() } else { state_lock.lock().unwrap() };
        let _held = is_global.then(|| holding(Holder::After));
        if let Some(t0) = t0 {
            HELD.add(HELD_AFTER_WAIT, t0.elapsed().as_micros() as u64);
        }
        let Some(s) = g.as_mut() else { return };
        s.prepare_after_execute(queue, ranges, imms)
    };
    let times = timing.map(UploadGateTiming::finish);
    drop(uploads.take());
    if let Some(times) = times.filter(|_| measure) {
        note_readback_gate(times);
    }
    drop(excess);
    if let Some(plan) = plan {
        submit_async_readback(queue, plan);
        if is_global {
            start_readback_thread();
        }
    }
}

pub unsafe fn readback_after_execute(queue: &ID3D12CommandQueue, ranges: &[(usize, usize)], imms: &[(usize, u64, u8)]) {
    readback_after_execute_with(&STATE, queue, ranges, imms);
}

/// The PC `ExecuteCommandLists` of `queue` for `list` (an adopted title queue's own method,
/// not XCL's hook: this is not a submit of the title's).
pub(super) unsafe fn pc_execute(queue: &ID3D12CommandQueue, list: &ID3D12GraphicsCommandList) {
    type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
    const SLOT: usize = 10;
    let q = queue.as_raw();
    let f: ExecFn = if super::real::is_adopted(q) { std::mem::transmute(super::real::orig_fn(q, SLOT)) } else { std::mem::transmute(*(*(q as *const *const usize)).add(SLOT)) };
    let one = [list.as_raw()];
    // (main's ordered queue turn: re-entrant, so a caller already in it keeps it)
    super::queue_submit::with(q as usize, || f(q, 1, one.as_ptr()));
}

/// copies back submitted and not landed yet, and whether any ran on the readback queue
static RB_PENDING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static RB_ASYNC: AtomicBool = AtomicBool::new(false);

/// Whether copies back land asynchronously (any ran on the readback queue so far)
pub fn readbacks_async() -> bool {
    RB_ASYNC.load(Ordering::Relaxed)
}

unsafe fn settle_readbacks_with(state_lock: &Mutex<Option<State>>, rb: &Arc<SharedReadbacks>, how: Drain, spin_us: u128, lock_ms: Option<u64>) -> bool {
    let deadline = lock_ms.map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));
    match wait_and_check_ready(rb, how, spin_us, deadline) {
        ReadyCheck::NothingReady => return true,
        ReadyCheck::TimedOut => return false,
        ReadyCheck::LandNow => {}
    }
    let is_global = std::ptr::eq(state_lock, &STATE);
    let excess = match deadline {
        None => {
            let mut g = state_lock.lock().unwrap();
            let _held = is_global.then(|| holding(Holder::Settle));
            if let Some(s) = g.as_mut().filter(|s| Arc::ptr_eq(&s.readbacks, rb)) {
                s.land_finished_readbacks()
            } else {
                Vec::new()
            }
        }
        Some(dl) => {
            let rem = dl.saturating_duration_since(std::time::Instant::now());
            let Some(mut g) = try_lock_giving_way(state_lock, &SUBMITS_WAITING, rem, is_global && submit_lock_first()) else {
                return false;
            };
            let _held = is_global.then(|| holding(Holder::Settle));
            if let Some(s) = g.as_mut().filter(|s| Arc::ptr_eq(&s.readbacks, rb)) {
                s.land_finished_readbacks()
            } else {
                Vec::new()
            }
        }
    };
    drop(excess);
    true
}

/// Copies back that are done on the GPU, into the title's memory, and those whose title work
/// is done (waiting for the copy): before a CPU read of memory the title's lists wrote.
pub unsafe fn settle_readbacks() {
    if RB_PENDING.load(Ordering::Acquire) == 0 {
        return;
    }
    let Some(rb) = readbacks_arc() else { return };
    let _ = settle_readbacks_with(&STATE, &rb, Drain::TitleDone, 2000, None);
}

/// `settle_readbacks` from a fence observation: the state's lock is only tried (for ~5 ms), as
/// XCL may be waiting on a fence of its own with the lock held.
pub unsafe fn settle_readbacks_try() {
    settle_readbacks_within(5);
}

/// `settle_readbacks` with the state's lock waited for `ms` ms at most (false: it was not had:
/// what is pending stays pending). A copy back whose title work is done is waited for 0.3 ms
/// outside all locks; if no job finished yet, `STATE` is not locked at all.
pub unsafe fn settle_readbacks_within(ms: u64) -> bool {
    if RB_PENDING.load(Ordering::Acquire) == 0 {
        return true;
    }
    // (`XCL_LOOK_STATS=2`: the time a title's thread spends here when it looks at a fence)
    let _cost = super::look_stats::deep().then(|| crate::hle::api_cost::time(crate::hle::api_cost::Call::FenceSettle));
    let Some(rb) = readbacks_arc() else { return true };
    settle_readbacks_with(&STATE, &rb, Drain::TitleDone, 300, Some(ms))
}

/// The thread that moves finished copies back into the title's memory, so it sees what its
/// lists wrote without submitting again (started with the first copy back).
fn start_readback_thread() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-sparse-readback".into()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_micros(500));
            if RB_PENDING.load(Ordering::Acquire) == 0 {
                continue;
            }
            // (`XCL_SUBMIT_LOCK_FIRST=1`: a submit waits for the lock: not now)
            if submit_lock_first() && SUBMITS_WAITING.load(Ordering::Acquire) > 0 {
                continue;
            }
            let Some(rb) = readbacks_arc() else { continue };
            if unsafe { !has_unpopped_finished_jobs(&rb) } {
                continue;
            }
            if submit_lock_first() && SUBMITS_WAITING.load(Ordering::Acquire) > 0 {
                continue;
            }
            let excess = {
                if let Ok(mut g) = STATE.try_lock() {
                    let _held = holding(Holder::ReadbackThread);
                    if let Some(s) = g.as_mut().filter(|s| Arc::ptr_eq(&s.readbacks, &rb)) {
                        unsafe { s.land_finished_readbacks() }
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            };
            drop(excess);
        });
    });
}

impl State {
    /// Snapshot mapped tiles and immediates under `&mut self` (`STATE`) for async readback, or
    /// execute the rare synchronous `rbq.is_none()` fallback inline.
    unsafe fn prepare_after_execute(&mut self, queue: &ID3D12CommandQueue, ranges: &[(usize, usize)], imms: &[(usize, u64, u8)]) -> (Option<AsyncReadbackPlan>, Vec<RbJob>) {
        let s = self;
        let mut filtered: Vec<(usize, usize)> = ranges.iter().copied().filter(|&(a, n)| n > 0 && s.window_index(a).is_some()).collect();
        // immediates: stored by XCL when the lists are done. One that another write of these
        // lists overlaps, and all of them without a queue of the title's to order them on,
        // come back with a copy like the rest.
        let mut imm: Vec<(usize, u64, u8, u32)> = Vec::new();
        for &(a, v, n) in imms {
            if s.window_index(a).is_none() {
                continue;
            }
            let len = (n & 0x7f) as usize;
            if s.rbq.is_none() || ranges.iter().any(|&(ra, rn)| a < ra + rn && ra < a + len) {
                filtered.push((a, len));
            } else {
                // (with the memory it is for, `TileEntry::gen`: no tile there yet counts as the
                // window's own, as before)
                imm.push((a, v, n, s.mapped.get(&(a & !(TILE - 1))).map_or(0, |e| e.gen)));
            }
        }
        if filtered.is_empty() && imm.is_empty() {
            return (None, Vec::new());
        }
        filtered.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(filtered.len());
        for (a, n) in filtered {
            match merged.last_mut() {
                Some(last) if a <= last.0.saturating_add(last.1) => {
                    let end = (last.0 + last.1).max(a + n);
                    last.1 = end - last.0;
                }
                _ => merged.push((a, n)),
            }
        }
        if s.rbq.is_some() {
            let mut windows: Vec<(usize, ID3D12Resource)> = Vec::new();
            let mut win_slot: Vec<usize> = vec![usize::MAX; s.windows.len()];
            let mut batches: Vec<Vec<RbChunk>> = Vec::new();
            let mut used = 0usize;
            for &(a, n) in &merged {
                let mut at = 0usize;
                while at < n {
                    let cur = a + at;
                    let tile = cur & !(TILE - 1);
                    let Some(wi) = s.window_index(cur) else { break };
                    let to_tile_end = (tile + TILE) - cur;
                    // (a tile with no video memory has nothing to copy back, like one not mapped)
                    let Some(gen) = s.mapped.get(&tile).filter(|e| e.pool != LAZY).map(|e| e.gen) else {
                        at += (n - at).min(to_tile_end);
                        continue;
                    };
                    if batches.is_empty() || used == READBACK_STAGING {
                        batches.push(Vec::new());
                        used = 0;
                    }
                    let lwi = if win_slot[wi] != usize::MAX {
                        win_slot[wi]
                    } else {
                        let idx = windows.len();
                        windows.push((s.windows[wi].0, s.windows[wi].1.clone()));
                        win_slot[wi] = idx;
                        idx
                    };
                    let to_win_end = (s.windows[wi].0 + crate::gpu::uma::WINDOW) - cur;
                    let chunk = (n - at).min(READBACK_STAGING - used).min(to_tile_end).min(to_win_end);
                    batches.last_mut().expect("batch").push(RbChunk { win: lwi, va: cur, len: chunk, gen });
                    used += chunk;
                    at += chunk;
                }
            }
            if imm.is_empty() && batches.is_empty() {
                return (None, Vec::new());
            }
            let submitting = SubmittingGuard::begin(&s.readbacks);
            return (
                Some(AsyncReadbackPlan {
                    rb_submit: Arc::clone(&s.rb_submit),
                    readbacks: Arc::clone(&s.readbacks),
                    windows,
                    batches,
                    imm,
                    submitting,
                }),
                Vec::new(),
            );
        }
        s.rb_value += 1;
        let v = s.rb_value;
        if super::real::native_queue_signal(queue, &s.rb_fence, v).is_ok() {
            let _ = super::real::native_queue_wait(&s.queue, &s.rb_fence, v);
            super::list_x::note_internal_wait(s.queue.as_raw() as usize, "sparse readback behind a title queue", &s.rb_fence, v);
        }
        // (the ranges are the lists' own GPU writes)
        (None, s.readback_ranges(&merged, true))
    }

    /// `readback_after_execute` on a standalone `State` (tests only).
    #[cfg(test)]
    unsafe fn after_execute(&mut self, queue: &ID3D12CommandQueue, ranges: &[(usize, usize)], imms: &[(usize, u64, u8)]) {
        let (plan, excess) = self.prepare_after_execute(queue, ranges, imms);
        drop(excess);
        if let Some(plan) = plan {
            submit_async_readback(queue, plan);
        }
    }
}

/// Synchronize `[va, va + len)` in a sparse window for a CPU read (`ID3D12Resource::Map` with
/// a non-empty read range, or `Map(0, NULL)` on a sparse readback resource): flushes pending
/// tile clears and unsubmitted CPU-changed chunks first so CPU writes are preserved, then
/// copies the range back from VRAM into CPU memory.
pub unsafe fn readback_now(va: usize, len: usize) {
    if len == 0 {
        return;
    }
    // (copies back already submitted land first, waited for outside all locks)
    if RB_PENDING.load(Ordering::Acquire) != 0 {
        if let Some(rb) = readbacks_arc() {
            wait_and_check_ready(&rb, Drain::TitleDone, 2000, None);
        }
    }
    let measure = super::memory_report::enabled();
    let _upload_lifetime = UPLOAD_STATE_LIFETIME.read().unwrap();
    let (mut uploads, mut timing) = UploadGateTiming::acquire(measure);
    let excess = readback_now_locked(va, len, &mut uploads, &mut timing);
    let times = timing.finish();
    drop(uploads.take());
    drop(_upload_lifetime);
    // (recycled copy-back jobs past the pool's cap drop outside all locks)
    drop(excess);
    if measure {
        note_readback_gate(times);
    }
}

/// `readback_now` with the upload gate held. Returns any excess recycled `RbJob`s for the caller
/// to drop outside `STATE`.
unsafe fn readback_now_locked(
    va: usize, len: usize, uploads: &mut Option<UploadGuard>, timing: &mut UploadGateTiming,
) -> Vec<RbJob> {
    'prepare: loop {
    let mut g = lock_state();
    let held = holding(Holder::CpuRead);
    let Some(s) = g.as_mut() else { return Vec::new() };
    if s.window_index(va).is_none() {
        return Vec::new();
    }
    let zeroed = zero_runs(std::mem::take(&mut s.to_zero));
    if !zeroed.is_empty() {
        for &(a, n) in &zeroed {
            s.clear_sent_range(a, n);
        }
        match s.upload_zeros(&zeroed) {
            UploadStep::Done(PreparationOutcome::Ready) => {}
            UploadStep::Wait(ticket) => {
                // Publish the clear retry journal before another uploader can take the gate. It
                // may consume it while this thread waits; after reacquiring, restart from state.
                rejournal_zeros(&mut s.to_zero, zeroed);
                drop(held);
                drop(g);
                if !timing.with_released(uploads, || wait_upload_released(ticket)) { return Vec::new() }
                continue 'prepare;
            }
            UploadStep::Done(_) => {
                rejournal_zeros(&mut s.to_zero, zeroed);
                return Vec::new();
            }
        }
    }
    // Preserve any unsubmitted CPU writes on pages overlapping `[va, va + len)` (read at the
    // tile's CPU address: a mirrored tile shows another CPU view)
    let mut dirty_runs: Vec<(usize, usize)> = Vec::new();
    let mut pg = va & !(PAGE - 1);
    let end = va + len;
    while pg < end {
        let t = pg & !(TILE - 1);
        let tile_end = (t + TILE).min(end);
        if let Some(entry) = s.mapped.get_mut(&t) {
            let cpu = entry.cpu(t);
            let mut p = pg;
            while p < tile_end {
                let idx = (p - t) / PAGE;
                if entry.holes & (1u16 << idx) == 0 {
                    let bytes = std::slice::from_raw_parts((cpu + (p - t)) as *const u8, PAGE);
                    scan_page(bytes, p, &entry.sent[idx], &mut dirty_runs);
                }
                p += PAGE;
            }
        }
        pg = tile_end;
    }
    if !dirty_runs.is_empty() {
        match s.upload_dirty(&dirty_runs) {
            UploadStep::Done(PreparationOutcome::Ready) => {}
            UploadStep::Wait(ticket) => {
                // Dirty bytes were only observed, not removed from a journal. Another uploader
                // may send them while this thread waits; restart and rescan current hashes.
                drop(held);
                drop(g);
                if !timing.with_released(uploads, || wait_upload_released(ticket)) { return Vec::new() }
                continue 'prepare;
            }
            UploadStep::Done(_) => return Vec::new(),
        }
    }
    // (the range is what the CPU reads, not what the GPU wrote: a changed chunk keeps its bytes)
    return s.readback_ranges(&[(va, len)], false);
    }
}

/// Clears taken from `State::to_zero` that did not go out, back into it, a piece per tile (a
/// decommit removes the pieces of its tiles by their start: a run over several tiles would
/// outlive one of them).
fn rejournal_zeros(to_zero: &mut BTreeSet<(usize, usize)>, runs: Vec<(usize, usize)>) {
    for (a, n) in runs {
        let end = a + n;
        let mut p = a;
        while p < end {
            let next = ((p & !(TILE - 1)) + TILE).min(end);
            to_zero.insert((p, next - p));
            p = next;
        }
    }
}

/// Submits waiting for the state's lock. A thread that takes it again and again – a title mapping
/// its memory a page at a time – can keep them out for seconds: the lock goes to the first to ask
/// once it is free, and the thread that just gave it up is back before a waiting one has woken (a
/// retail title's loading screen stood still for 8 s while another of its threads mapped 1.2 GB).
static SUBMITS_WAITING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// what a commit waits for them at most
const GIVE_WAY_US: u128 = 2000;

/// `m` for a submit: the others let it in first (`lock_in_turn`).
fn lock_first<'a, T>(m: &'a Mutex<T>, waiting: &std::sync::atomic::AtomicUsize) -> std::sync::MutexGuard<'a, T> {
    waiting.fetch_add(1, Ordering::AcqRel);
    let g = m.lock();
    waiting.fetch_sub(1, Ordering::AcqRel);
    g.unwrap()
}

/// Submits in `before_execute` waiting for the state's lock (they send the put-off mappings).
static FLUSHERS_WAITING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `XCL_SUBMIT_LOCK_FIRST=1` (off unless set): a submit's other uses of the state's lock (its
/// after stage's copies back, its ranges' mapped tiles) are let in first like its `before_execute`,
/// and the threads that only try the lock in a loop (copies back settled at a fence look, the
/// readback thread, put-off mappings sent) do not try it while a submit waits. A retail title's
/// world: each submit waited ~2 ms for the lock while title threads looked at fences ~170 times a
/// second, each trying the lock for up to 5 ms; a thread that tries the lock as it is let go takes
/// it before the waiting submit has woken.
fn submit_lock_first() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SUBMIT_LOCK_FIRST").is_ok_and(|v| v.trim() == "1"))
}

/// The state's lock for a submit's after stage and ranges: `lock_first` with
/// `XCL_SUBMIT_LOCK_FIRST=1`, else `lock_state` as before.
fn lock_for_submit() -> std::sync::MutexGuard<'static, Option<State>> {
    if !submit_lock_first() {
        return lock_state();
    }
    let mut g = lock_first(&STATE, &SUBMITS_WAITING);
    forget_released(&mut g);
    g
}

/// `m` tried until it is had or `budget` has gone by (None). With `give_way`, it is not tried
/// while a submit waits for it (`waiting`); that time does not count against `budget`, up to
/// `budget` more (a caller that may not wait for the lock still gives up in the end).
fn try_lock_giving_way<'a, T>(m: &'a Mutex<T>, waiting: &std::sync::atomic::AtomicUsize, budget: std::time::Duration, give_way: bool) -> Option<std::sync::MutexGuard<'a, T>> {
    let t0 = std::time::Instant::now();
    let mut gave = std::time::Duration::ZERO;
    loop {
        if give_way && gave < budget && waiting.load(Ordering::Acquire) > 0 {
            let t = std::time::Instant::now();
            std::thread::yield_now();
            gave += t.elapsed();
            continue;
        }
        if let Ok(g) = m.try_lock() {
            return Some(g);
        }
        if t0.elapsed().saturating_sub(gave) >= budget {
            return None;
        }
        std::thread::yield_now();
    }
}

/// `m` once no submit waits for it (or `GIVE_WAY_US` have gone by).
fn lock_in_turn<'a, T>(m: &'a Mutex<T>, waiting: &std::sync::atomic::AtomicUsize) -> std::sync::MutexGuard<'a, T> {
    if waiting.load(Ordering::Acquire) > 0 {
        let t0 = std::time::Instant::now();
        while waiting.load(Ordering::Acquire) > 0 && t0.elapsed().as_micros() < GIVE_WAY_US {
            std::thread::yield_now();
        }
    }
    m.lock().unwrap()
}

/// The title committed `[a, a + n)` in a sparse window. The pages of tiles mapped for it that
/// lie outside the range are not committed (holes); the range's pages are.
pub unsafe fn committed(a: usize, n: usize) {
    committed_with_protection(a, n, true);
}

/// `committed`, recording the protection established by the same successful commit without an
/// interval in which an affirmative writability query can observe the new page incorrectly.
pub unsafe fn committed_with_protection(a: usize, n: usize, writable: bool) {
    if n == 0 {
        return;
    }
    super::title_textures::mark_hot(a, n);
    let mut g = lock_in_turn(&STATE, &SUBMITS_WAITING);
    let _held = holding(Holder::Commits);
    forget_released(&mut g);
    let Some(s) = g.as_mut() else {
        let (start, end) = page_span(a, n);
        early_add(&mut EARLY.lock().unwrap(), start, end);
        if writable {
            early_remove(&mut EARLY_NOT_RW.lock().unwrap(), start, end);
        } else {
            early_add(&mut EARLY_NOT_RW.lock().unwrap(), start, end);
        }
        return;
    };
    s.commit_range_with_protection(a, n, writable);
}

/// The title decommitted `[a, a + n)`: tiles whose pages are all uncommitted (whether in one
/// full-tile decommit or incrementally via sub-tile decommits) give their video memory back
/// (unless a placed buffer still pins the pool tile).
pub unsafe fn decommitted(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    let mut g = lock_in_turn(&STATE, &SUBMITS_WAITING);
    let _held = holding(Holder::Commits);
    forget_released(&mut g);
    let Some(s) = g.as_mut() else {
        let (start, end) = page_span(a, n);
        early_remove(&mut EARLY.lock().unwrap(), start, end);
        return;
    };
    s.decommit_range(a, n);
}

/// `decommitted` around the caller's decommit `os` of `[a, a + n)` in a sparse window (its
/// VirtualFree, called with no lock of XCL's held; its result is returned): a CPU view there (a
/// title's reservation of 64 KiB pages that a GPU view shows) has its pages taken as holes by
/// submits until Windows is done, as `source_decommit` does outside the windows. For a release,
/// drop the pair and `forget_mirrors` inside `os`, after the VirtualFree.
pub unsafe fn decommit<R>(a: usize, n: usize, os: impl FnOnce() -> R) -> R {
    if n == 0 {
        return os();
    }
    let (span, _) = freeing_begins(&SOURCES, a, n);
    decommitted(a, n);
    let r = os();
    freeing_ends(&SOURCES, span);
    r
}

/// Tiles wait for their mapping call (`State::pending`; read without the lock).
static PENDING: AtomicBool = AtomicBool::new(false);
static PENDING_WAKE: (Mutex<()>, std::sync::Condvar) = (Mutex::new(()), std::sync::Condvar::new());

/// A tile's mapping was put off: sent by the thread started here in 2 ms, unless something sends
/// it before. Work already on the GPU can reach the memory without another submit (it waits for a
/// value the CPU writes).
fn pending_soon() {
    if PENDING.swap(true, Ordering::AcqRel) {
        return;
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-tile-maps".into()).spawn(|| loop {
            {
                let mut g = PENDING_WAKE.0.lock().unwrap_or_else(|e| e.into_inner());
                while !PENDING.load(Ordering::Acquire) {
                    g = PENDING_WAKE.1.wait(g).unwrap_or_else(|e| e.into_inner());
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
            if PENDING.load(Ordering::Acquire) {
                send_pending();
            }
        });
    });
    drop(PENDING_WAKE.0.lock().unwrap_or_else(|e| e.into_inner()));
    PENDING_WAKE.1.notify_one();
}

/// The mappings put off go to the GPU now: the CPU is about to signal a fence, and work submitted
/// before the title committed the memory may run because of it.
pub fn send_pending() {
    // (a signal under the state's lock must not wait for it; the thread sends them then)
    let t0 = std::time::Instant::now();
    loop {
        // (asked each time: another thread may be sending them, and has sent them when the flag goes)
        if !PENDING.load(Ordering::Acquire) {
            return;
        }
        // (`XCL_SUBMIT_LOCK_FIRST=1`: a submit that waits for the lock in `before_execute` sends
        // them itself, `flush_pending`)
        if submit_lock_first() && FLUSHERS_WAITING.load(Ordering::Acquire) > 0 {
            if t0.elapsed().as_millis() >= 2 {
                return;
            }
            std::thread::yield_now();
            continue;
        }
        match STATE.try_lock() {
            Ok(mut g) => {
                let _held = holding(Holder::TileMaps);
                if let Some(s) = g.as_mut() {
                    unsafe { s.flush_pending() };
                }
                PENDING.store(false, Ordering::Release);
                return;
            }
            Err(std::sync::TryLockError::Poisoned(_)) => return PENDING.store(false, Ordering::Release),
            Err(std::sync::TryLockError::WouldBlock) if t0.elapsed().as_millis() >= 2 => return,
            Err(std::sync::TryLockError::WouldBlock) => std::thread::yield_now(),
        }
    }
}

impl State {
    /// `committed` with the state locked
    unsafe fn commit_range(&mut self, a: usize, n: usize) {
        self.commit_range_with_protection(a, n, true);
    }

    /// `committed_with_protection` with the state locked: `writable` false marks the committed
    /// pages read-only (`TileEntry::not_rw`) for `known_writable`
    unsafe fn commit_range_with_protection(&mut self, a: usize, n: usize, writable: bool) {
        // a tile over a CPU view committed at its own address: its own memory from here on
        if !self.mirror_src.is_empty() {
            let mut t = a & !(TILE - 1);
            while t < a.saturating_add(n) {
                if self.unlink_mirror(t) {
                    if let Some(e) = self.mapped.get_mut(&t).filter(|e| e.src != 0) {
                        // (what it held was the CPU view's: cleared, and the pages committed here
                        // are no holes below)
                        *e = TileEntry::with_holes(e.pool, 0xFFFF);
                        self.to_zero.insert((t, TILE));
                        self.mirrored -= 1;
                        early_add(&mut self.runs, t, t + TILE);
                        MAPPED_GEN.fetch_add(1, Ordering::Release);
                    }
                }
                match t.checked_add(TILE) {
                    Some(next) => t = next,
                    None => break,
                }
            }
        }
        if lazy_commits() {
            self.ensure_listed(a, n);
        } else {
            self.ensure_mapped(a, n, |_| (0xFFFF, 0));
        }
        let mut pg = a & !(PAGE - 1);
        let end = a.saturating_add(n);
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = t.saturating_add(TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    let bit = 1u16 << ((p - t) / PAGE);
                    entry.holes &= !bit;
                    if writable {
                        entry.not_rw &= !bit;
                    } else {
                        entry.not_rw |= bit;
                    }
                    p += PAGE;
                }
            }
            if tile_end <= pg {
                break;
            }
            pg = tile_end;
        }
        self.mapping_epoch = self.mapping_epoch.wrapping_add(1);
    }

    /// `decommitted` with the state locked: the tiles that go are parked (`flush_parked` unmaps
    /// them and frees their pool tiles)
    unsafe fn decommit_range(&mut self, a: usize, n: usize) {
        let mut pools: Vec<(usize, (u32, u32))> = Vec::new();
        let end = a.saturating_add(n);
        // tiles over a CPU view go as a whole (their holes are the CPU view's pages, not these)
        if !self.mirror_src.is_empty() {
            let mut t = a & !(TILE - 1);
            while t < end {
                if self.mirror_src.contains_key(&t) {
                    self.drop_mirror(t);
                }
                match t.checked_add(TILE) {
                    Some(next) => t = next,
                    None => break,
                }
            }
            if self.parked.len() >= PARKED_MAX && self.flush_parked() {
                self.retire_free_pools();
            }
        }
        // a CPU view in a window (a title's reservation of 64 KiB pages a GPU view shows): the
        // tiles over it have holes there too (below, only the entries of these addresses get them)
        if !self.src_runs.is_empty() {
            self.source_decommitted_in(a, n);
        }
        let mut t = a & !(TILE - 1);
        while t < end {
            if let Some(e) = self.mapped.get(&t).filter(|e| e.pool != LAZY) {
                pools.push((t, e.pool));
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        let tiles = decommit_tiles(&mut self.mapped, &self.pinned, &mut Vec::new(), &mut self.to_zero, a, n);
        self.mapping_epoch = self.mapping_epoch.wrapping_add(1);
        if tiles.is_empty() {
            return;
        }
        for &t in &tiles {
            early_remove(&mut self.runs, t, t + TILE);
            if let Ok(i) = pools.binary_search_by_key(&t, |p| p.0) {
                self.parked.insert(t, pools[i].1);
            }
        }
        MAPPED_GEN.fetch_add(1, Ordering::Release);
        if self.parked.len() >= PARKED_MAX && self.flush_parked() {
            self.drain_quarantined();
            self.retire_free_pools();
        }
    }

    /// A pool tile a buffer used at `t` is freed (`cooling`), unless its window maps it, it
    /// is parked there, it waits in `quarantined`, or another buffer uses it. Kept mapped for
    /// the buffer: parked from here on.
    fn release_if_unused(&mut self, t: usize, tile: (u32, u32)) {
        if self.pinned.contains_key(&tile) {
            return;
        }
        if self.kept.get(&t) == Some(&tile) {
            self.kept.remove(&t);
            self.parked.insert(t, tile);
            return;
        }
        if self.mapped.get(&t).is_none_or(|e| e.pool != tile)
            && self.parked.get(&t) != Some(&tile)
            && !self.quarantined.iter().any(|b| b.tiles.contains(&tile))
        {
            self.cooling.push_back((tile, std::time::Instant::now()));
        }
    }

    /// The fence `quarantine_tiles_for` signals on title queue `q` after its work, made now (once per
    /// queue; a queue of another device, e.g. an isolated unit test's, is left out). Which queues
    /// get a mark is `present`'s registry (`queue_completion_snapshot`), so a queue is never kept
    /// alive here; `forget_queue` drops its entry when it goes. False: no fence could be made.
    #[cfg(test)]
    unsafe fn note_queue(&mut self, q: &ID3D12CommandQueue) -> bool {
        let key = q.as_raw() as usize;
        if key == self.queue.as_raw() as usize || self.queue_fences.contains_key(&key) {
            return true;
        }
        let mut qdev: Option<ID3D12Device> = None;
        if q.GetDevice(&mut qdev).is_err() || qdev.as_ref().is_none_or(|d| d.as_raw() != self.dev.as_raw()) {
            return true;
        }
        match self.dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) {
            Ok(fence) => {
                self.queue_fences.insert(key, (fence, 0));
                true
            }
            Err(_) => false,
        }
    }

    unsafe fn quarantine_tiles_for(
        &mut self,
        tiles: Vec<(u32, u32)>,
        queues: &[ID3D12CommandQueue],
        retiring: &[(ID3D12Fence, u64)],
        blocked: bool,
    ) {
        if tiles.is_empty() {
            return;
        }
        let mut marks = Vec::new();
        let mut ok = !blocked;
        if ok {
            // Establish a checked mark after the unmaps. An earlier unchecked signal cannot
            // authorize reuse, and a removed fence's UINT64_MAX is never normal completion.
            match next_retirement_value(self.value) {
                Some(next) if super::real::native_queue_signal(&self.queue, &self.fence, next).is_ok() => {
                    self.value = next;
                    marks = initial_quarantine_marks((self.fence.clone(), next), retiring, blocked).unwrap_or_default();
                }
                _ => ok = false,
            }
        }
        let mut seen = Vec::with_capacity(queues.len());
        marks.reserve(queues.len());
        for q in queues.iter().filter(|_| ok) {
            let key = q.as_raw() as usize;
            if key == self.queue.as_raw() as usize || seen.contains(&key) {
                continue;
            }
            let mut qdev: Option<ID3D12Device> = None;
            if q.GetDevice(&mut qdev).is_err() || qdev.as_ref().is_none_or(|d| d.as_raw() != self.dev.as_raw()) {
                continue;
            }
            seen.push(key);
            let entry = match self.queue_fences.entry(key) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => match self.dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) {
                    Ok(fence) => entry.insert((fence, 0)),
                    Err(_) => {
                        ok = false;
                        break;
                    }
                },
            };
            let Some(next) = next_retirement_value(entry.1) else {
                ok = false;
                break;
            };
            if super::real::native_queue_signal(&q, &entry.0, next).is_ok() {
                entry.1 = next;
                marks.push((entry.0.clone(), next));
            } else {
                ok = false;
                break;
            }
        }
        self.quarantined.push_back(RetiredTiles { tiles, marks: ok.then_some(marks), at: std::time::Instant::now() });
    }

    /// The `quarantined` tiles the GPU is past (`completed` answers for a fence) go to `cooling`
    /// (from when they were unmapped).
    unsafe fn drain_quarantined_with(&mut self, mut completed: impl FnMut(&ID3D12Fence) -> u64) -> bool {
        let mut freed = false;
        while let Some(front) = self.quarantined.front() {
            let ready = front.marks.as_ref().is_some_and(|marks| retirement_marks_reached(marks, &mut completed));
            if !ready {
                break;
            }
            let batch = self.quarantined.pop_front().unwrap();
            let at = batch.at;
            for tile in batch.tiles {
                if !self.pinned.contains_key(&tile) {
                    self.cooling.push_back((tile, at));
                    freed = true;
                }
            }
        }
        freed
    }

    /// `drain_quarantined_with` XCL's own query of the fences: the title's `GetCompletedValue`
    /// hook takes the state's lock, which the caller holds.
    unsafe fn drain_quarantined(&mut self) -> bool {
        self.drain_quarantined_with(|fence| super::real::fence_completed(fence))
    }

    /// The parked tiles unmapped from their windows (one call per window) and their pool tiles
    /// (but those of placed buffers) quarantined behind `self.queue` and every known title
    /// queue's completion fence, then freed to `cooling` (`drain_quarantined`). Pools left all
    /// free are not retired here: callers do that (`retire_free_pools`).
    unsafe fn flush_parked(&mut self) -> bool {
        self.flush_parked_with(None)
    }

    unsafe fn flush_parked_with(&mut self, extra: Option<&ID3D12CommandQueue>) -> bool {
        if self.parked.is_empty() {
            return false;
        }
        let mut completion = super::present::queue_completion_snapshot();
        if let Some(queue) = extra.filter(|queue| {
            super::present::queue_accepts_internal_work(queue.as_raw() as usize)
                && !completion.active.iter().any(|active| active.as_raw() == queue.as_raw())
        }) {
            completion.active.push(queue.clone());
        }
        self.flush_parked_for(&completion.active, &completion.retiring, completion.blocked)
    }

    unsafe fn flush_parked_for(
        &mut self,
        queues: &[ID3D12CommandQueue],
        retiring: &[(ID3D12Fence, u64)],
        blocked: bool,
    ) -> bool {
        if self.parked.is_empty() {
            return false;
        }
        let mut by_window: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        let mut tiles: Vec<(u32, u32)> = Vec::new();
        for (t, tile) in std::mem::take(&mut self.parked) {
            // (a placed buffer uses it)
            if self.pinned.contains_key(&tile) {
                self.kept.insert(t, tile);
                continue;
            }
            if let Some(wi) = self.window_index(t) {
                by_window.entry(wi).or_default().push(t);
            }
            tiles.push(tile);
        }
        if by_window.is_empty() {
            // (nothing to unmap)
            self.quarantine_tiles_for(tiles, queues, retiring, blocked);
            return false;
        }
        for (wi, ts) in by_window {
            let (b, r) = (self.windows[wi].0, self.windows[wi].1.clone());
            TILE_WORK[2].fetch_add(ts.len() as u64, Ordering::Relaxed);
            self.unmap_tiles(&r, b, &ts);
        }
        self.mapping_epoch = self.mapping_epoch.wrapping_add(1);
        self.signal();
        self.quarantine_tiles_for(tiles, queues, retiring, blocked);
        true
    }
}

/// Whether `[p, p + n)` lies in a sparse window and every page is recorded as committed and
/// writable. False means the caller must ask Windows; this cache only supplies affirmative
/// answers, avoiding `VirtualQuery` walking a multi-gigabyte sparse reservation.
pub fn known_writable(p: usize, n: usize) -> bool {
    let Some(end) = p.checked_add(n) else { return false };
    if n == 0 || !crate::gpu::uma::get().is_some_and(|u| u.is_sparse(p) && u.is_sparse(end - 1)) {
        return false;
    }
    {
        let state = lock_in_turn(&STATE, &SUBMITS_WAITING);
        let _held = holding(Holder::Other);
        let Some(state) = state.as_ref() else { return false };
        if !recorded_committed(&state.mapped, p, end) {
            return false;
        }
    }
    resident_pages_writable(p, end)
}

/// Every page touched by `[p, end)` is recorded (a tile of the window's own, with video memory or
/// `LAZY`), is committed, and has not been given a non-writable protection. A tile over a CPU view
/// (`TileEntry::src`) is never: its own address is no CPU memory (the title reserved it with no
/// access) and its `holes` are the CPU view's pages. Check one 64 KiB tile at a time while
/// holding `STATE`.
fn recorded_committed(mapped: &FxHashMap<usize, TileEntry>, p: usize, end: usize) -> bool {
    let mut tile = p & !(TILE - 1);
    while tile < end {
        let first = (p.max(tile) - tile) / PAGE;
        let tile_end = tile.saturating_add(TILE).min(end);
        let last = (tile_end - 1 - tile) / PAGE;
        let pages = last - first + 1;
        let mask = ((((1u32 << pages) - 1) << first) & u16::MAX as u32) as u16;
        if !mapped.get(&tile).is_some_and(|entry| entry.src == 0 && (entry.holes | entry.not_rw) & mask == 0) {
            return false;
        }
        if tile_end <= tile {
            return false;
        }
        tile = tile_end;
    }
    true
}

/// Record a successful protection change in a sparse window. `writable` means a writable
/// protection without PAGE_GUARD. Before device creation, retain page ranges for `init` replay.
pub fn protection_changed(a: usize, n: usize, writable: bool) {
    let Some(end) = a.checked_add(n) else { return };
    let Some(uma) = crate::gpu::uma::get() else { return };
    let mut state = lock_in_turn(&STATE, &SUBMITS_WAITING);
    let _held = holding(Holder::Commits);
    for k in 0..uma.shader_windows() {
        if !uma.is_sparse_window(k) {
            continue;
        }
        let (base, size) = uma.window(k);
        let (start, stop) = (a.max(base), end.min(base.saturating_add(size)));
        if start >= stop {
            continue;
        }
        match state.as_mut() {
            Some(state) => mark_protection(&mut state.mapped, start, stop - start, writable),
            None => early_protection(&mut EARLY_NOT_RW.lock().unwrap(), start, stop - start, writable),
        }
    }
}

fn mark_protection(mapped: &mut FxHashMap<usize, TileEntry>, a: usize, n: usize, writable: bool) {
    let end = a.saturating_add(n);
    let mut tile = a & !(TILE - 1);
    while tile < end {
        let first = (a.max(tile) - tile) / PAGE;
        let tile_end = tile.saturating_add(TILE).min(end);
        let last = (tile_end - 1 - tile) / PAGE;
        let pages = last - first + 1;
        let mask = ((((1u32 << pages) - 1) << first) & u16::MAX as u32) as u16;
        if let Some(entry) = mapped.get_mut(&tile) {
            if writable {
                entry.not_rw &= !mask;
            } else {
                entry.not_rw |= mask;
            }
        }
        if tile_end <= tile {
            break;
        }
        tile = tile_end;
    }
}

/// No resident page in `[p, end)` has a non-writable or guard protection. Nonresident pages are
/// judged by the protection record above; this query inspects metadata without touching pages.
fn resident_pages_writable(p: usize, end: usize) -> bool {
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    const BATCH: usize = 64;
    const WRITABLE: usize = 0x04 | 0x08 | 0x40 | 0x80;
    let (first, last) = (p >> 12, (end - 1) >> 12);
    let mut entries: [PSAPI_WORKING_SET_EX_INFORMATION; BATCH] = unsafe { std::mem::zeroed() };
    let mut page = first;
    while page <= last {
        let count = (last - page + 1).min(BATCH);
        for (i, entry) in entries[..count].iter_mut().enumerate() {
            *entry = unsafe { std::mem::zeroed() };
            entry.VirtualAddress = ((page + i) << 12) as *mut c_void;
        }
        let bytes = (count * std::mem::size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32;
        if unsafe {
            K32QueryWorkingSetEx(
                windows_sys::Win32::System::Threading::GetCurrentProcess(),
                entries.as_mut_ptr() as *mut c_void,
                bytes,
            )
        } == 0
        {
            return false;
        }
        if entries[..count].iter().any(|entry| {
            let flags = unsafe { entry.VirtualAttributes.Flags };
            let protection = (flags >> 4) & 0x7FF;
            flags & 1 != 0 && (protection & WRITABLE == 0 || protection & PAGE_GUARD as usize != 0)
        }) {
            return false;
        }
        page += count;
    }
    true
}

/// Window tiles asked for by `map_mirror`: each is queued once, until it is forgotten
/// (`forget_mirrors`) or left out (`State::map_mirrors`).
static MIRROR_ASKED: super::sharded::Sharded<()> = super::sharded::Sharded::new();
/// How often a tile left `MIRROR_ASKED` (a view that went, a tile given back with its source):
/// whoever remembers having asked (`view_want`'s threads) asks again.
static MIRROR_FORGOTTEN: AtomicU64 = AtomicU64::new(0);

/// Counts up whenever a tile that was asked for may be asked for again (`map_mirror` would
/// queue it): a caller that keeps what it asked for forgets it when this changes.
pub fn mirror_asks_forgotten() -> u64 {
    MIRROR_FORGOTTEN.load(Ordering::Acquire)
}

/// Tile `t` is asked for no more.
fn forget_ask(t: usize) {
    if MIRROR_ASKED.remove(t).is_some() {
        MIRROR_FORGOTTEN.fetch_add(1, Ordering::Release);
        #[cfg(test)]
        tests::ASKS_FORGOTTEN.with(|n| n.set(n.get() + 1));
    }
}
/// The tiles asked for that no submit has taken yet. A lock of its own: `map_mirror` is called
/// while recording, on any thread, and never waits for `STATE`.
static MIRROR_ASKS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
static MIRROR_ASKING: AtomicBool = AtomicBool::new(false);

/// What a thread that decommits the title's CPU memory needs to know of the tiles over CPU views,
/// and what a submit needs to know of those decommits. A lock of its own, which takes no other:
/// a decommit asks it with no lock of the state's (most of them are of memory no tile shows, and
/// a title's loading threads decommit all the time), a submit with `STATE` held.
#[derive(Default)]
struct Sources {
    /// `State::src_runs`, changed with it (`link_mirror`, `unlink_mirror`)
    runs: BTreeMap<usize, usize>,
    /// the pages `[start, end)` of decommits under way (`source_decommit`): told, Windows not
    /// done yet. A submit takes them as holes, whatever Windows says of them for now
    freeing: Vec<(usize, usize)>,
}

impl Sources {
    /// Whether tiles show any of `[a, end)`
    fn shows(&self, a: usize, end: usize) -> bool {
        self.runs.range(..end).next_back().is_some_and(|(_, &e)| e > a)
    }
}

static SOURCES: Mutex<Sources> = Mutex::new(Sources { runs: BTreeMap::new(), freeing: Vec::new() });

fn lock_sources(m: &Mutex<Sources>) -> std::sync::MutexGuard<'_, Sources> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A decommit of the pages of `[a, a + n)` begins: submits take them as holes until it ends
/// (`freeing_ends` with the span returned). True: tiles show some of them, and the state has to
/// be told before Windows frees them. Either a submit linked a tile there before this (and the
/// state is told), or it links it after and finds the pages here (in the one lock, in turn).
fn freeing_begins(m: &Mutex<Sources>, a: usize, n: usize) -> ((usize, usize), bool) {
    let span = page_span(a, n);
    let mut l = lock_sources(m);
    l.freeing.push(span);
    (span, l.shows(span.0, span.1))
}

/// The decommit of `span` (`freeing_begins`) is done.
fn freeing_ends(m: &Mutex<Sources>, span: (usize, usize)) {
    let mut l = lock_sources(m);
    if let Some(i) = l.freeing.iter().position(|&f| f == span) {
        l.freeing.swap_remove(i);
    }
}

/// What window tile `g` shows (`tile_shows`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Under {
    /// not in a GPU view
    NoView,
    /// nothing (zeros): unmapped by the queue's page mappings, no page behind it, or a page kept
    /// where kernelx does not see the memory go (no tile may show it there)
    Nothing,
    /// the 64 KiB-aligned CPU tile: the CPU view's at the same offset, or where the page mapped
    /// there is kept
    Cpu(usize),
}

/// What window tile `g` of a GPU view shows (`kernelx::mirrors::shows`; locks of its own, which
/// take no other: it may be asked with `STATE` held). While nothing is mapped, the CPU view's tile
/// as before.
fn tile_shows(g: usize) -> Under {
    use crate::hle::kernelx::mirrors::{shows, Shows};
    match shows(g) {
        Shows::NoView => Under::NoView,
        Shows::Same(c) => Under::Cpu(c & !(TILE - 1)),
        Shows::Page { at, tile: true } => Under::Cpu(at & !(TILE - 1)),
        Shows::Page { .. } | Shows::Nothing => Under::Nothing,
    }
}

/// The 64 KiB-aligned CPU tile behind window tile `g` of a GPU view (`tile_shows`).
fn mirror_source_tile(g: usize) -> Option<usize> {
    match tile_shows(g) {
        Under::Cpu(c) => Some(c),
        _ => None,
    }
}

/// 64 KiB of XCL's own, reserved and never committed: the source of a tile that shows nothing but
/// keeps its video memory, as a placed buffer maps its tiles once (`buffer_tiles`,
/// `remap_mirrors`). Its pages are always holes: nothing scans it, and no copy back lands there.
fn no_source() -> usize {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *AT.get_or_init(|| {
        use windows_sys::Win32::System::Memory::MEM_RESERVE;
        unsafe { VirtualAlloc(std::ptr::null(), TILE, MEM_RESERVE, PAGE_NOACCESS) as usize }
    })
}

/// Window tiles whose page the queue's page mappings changed (`remapped`), for the next submit to
/// link again (`State::remap_mirrors`). A lock of its own, as `MIRROR_ASKS`.
static REMAPS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
static REMAPPING: AtomicBool = AtomicBool::new(false);

/// What the GPU-view tiles `tiles` show changed (`kernelx::mirrors`: a mapping, an unmap, a freed
/// page, a page that got or lost its place): those asked for (`map_mirror`) are linked again at
/// the next submit. Any thread; no state's lock.
pub fn remapped(tiles: &[usize]) {
    let asked: Vec<usize> = tiles.iter().copied().filter(|&t| MIRROR_ASKED.contains(t)).collect();
    if asked.is_empty() {
        return;
    }
    REMAPS.lock().unwrap().extend(asked);
    REMAPPING.store(true, Ordering::Release);
}

/// The tiles of `q` (`REMAPS`) join those waiting to be linked again (`State::remaps`). Past
/// `MIRROR_ASKS_MAX` the queue is sorted and each tile kept once: a page changed at every frame,
/// or by several calls, waits once, and the queue holds no more than the tiles asked for.
fn take_remaps(remaps: &mut Vec<usize>, q: &mut Vec<usize>) {
    remaps.append(q);
    if remaps.len() > MIRROR_ASKS_MAX {
        remaps.sort_unstable();
        remaps.dedup();
    }
}

/// The physical page the GPU-view tiles `tiles` show moved from `from` to `to` (both 64 KiB
/// aligned; `whole`: its bytes were copied there): the tiles linked over `from` follow it now,
/// with their video memory (`State::retarget`). Called on the moving thread (kernelx, AWAY held)
/// before the memory at `from` can go. The state's lock only when tiles show `from` (it gives way
/// to submits).
pub unsafe fn moved(tiles: &[usize], from: usize, to: usize, whole: bool) {
    if tiles.is_empty() || from == 0 || to == 0 || !lock_sources(&SOURCES).shows(from, from + TILE) {
        return;
    }
    let mut g = lock_in_turn(&STATE, &SUBMITS_WAITING);
    let _held = holding(Holder::Sources);
    forget_released(&mut g);
    let Some(s) = g.as_mut() else { return };
    for &w in tiles {
        s.retarget(w, from, to, whole);
    }
}

/// The GPU-view tiles linked again so far: [to show what the page mappings put there (or nothing),
/// that followed their page].
static VIEW_TILES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// `VIEW_TILES` so far.
pub fn page_tiles() -> [u64; 2] {
    std::array::from_fn(|k| VIEW_TILES[k].load(Ordering::Relaxed))
}

/// The CPU tile a placed buffer's tile is linked over for what it shows: a tile that shows nothing
/// keeps its video memory over `no_source` (the buffer maps its tiles once). None: no view.
fn buffer_source(under: Under) -> Option<usize> {
    match under {
        Under::Cpu(c) => Some(c),
        Under::Nothing => Some(no_source()).filter(|&n| n != 0),
        Under::NoView => None,
    }
}

/// The window tiles of `[va, va + len)` are to show the title's CPU view behind them (`va` in a
/// GPU view of its memory, `kernelx::mirrors`): video memory, filled from the CPU view and kept in
/// step with it, from the next submit on. Any thread; it waits for no submit (no `STATE`) and
/// asks Windows nothing. Tiles that are not in a sparse window or in a GPU view are left out
/// there. Returns how many of its tiles were queued now (the others were asked for before).
pub fn map_mirror(va: usize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let end = va.saturating_add(len);
    let mut new: Vec<usize> = Vec::new();
    let mut t = va & !(TILE - 1);
    while t < end {
        if MIRROR_ASKED.insert(t, ()).is_none() {
            new.push(t);
        }
        match t.checked_add(TILE) {
            Some(next) => t = next,
            None => break,
        }
    }
    if new.is_empty() {
        return 0;
    }
    let queued = new.len();
    if view_switches().stats {
        VIEW_ASKED.fetch_add(queued as u64, Ordering::Relaxed);
    }
    MIRROR_ASKS.lock().unwrap().extend(new);
    MIRROR_ASKING.store(true, Ordering::Release);
    queued
}

/// The title's CPU memory `[a, a + n)` is decommitted by `os` (the caller's VirtualFree, called
/// with no lock of XCL's held; its result is returned): the tiles shown over it have holes
/// there first (`source_decommitted`), and while `os` runs a submit takes those pages as holes,
/// whatever Windows says of them. Told only before the decommit, a submit in between asked Windows
/// about a page written before, found it still committed, read it as Windows freed it, and left it
/// committed in its tile for good (a later copy back or immediate wrote to it). The state's lock
/// is taken only when tiles show some of the range.
pub unsafe fn source_decommit<R>(a: usize, n: usize, os: impl FnOnce() -> R) -> R {
    if n == 0 {
        return os();
    }
    let (span, shown) = freeing_begins(&SOURCES, a, n);
    if shown {
        source_decommitted(a, n);
    }
    let r = os();
    freeing_ends(&SOURCES, span);
    r
}

/// `source_decommit` for a release of the title's CPU memory `[a, a + n)` by `os`: after it the
/// tiles over it are forgotten as well (`forget_mirrors`: the address may be given out anew),
/// before the pages stop being holes for submits. Drop the pair (`kernelx::mirrors::released`)
/// before `os` returns, so that no tile is linked over the address after the forget.
pub unsafe fn source_release<R>(a: usize, n: usize, os: impl FnOnce() -> R) -> R {
    if n == 0 {
        return os();
    }
    let (span, shown) = freeing_begins(&SOURCES, a, n);
    if shown {
        source_decommitted(a, n);
    }
    let r = os();
    forget_mirrors(a, n);
    freeing_ends(&SOURCES, span);
    r
}

/// The part of `source_decommit` that tells the state: the tiles shown over the title's CPU
/// memory `[a, a + n)` have holes there, cleared on the GPU (`sent = None` says the GPU holds
/// zeros), and a tile with no committed page left gives its video memory back. A submit's scan
/// reads those pages with the state's lock held: after this it does not – unless it asks Windows
/// before the decommit lands, which only `source_decommit` rules out (call that).
pub unsafe fn source_decommitted(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    let (start, end) = page_span(a, n);
    if !lock_sources(&SOURCES).shows(start, end) {
        return;
    }
    let mut g = lock_in_turn(&STATE, &SUBMITS_WAITING);
    let _held = holding(Holder::Sources);
    forget_released(&mut g);
    let Some(s) = g.as_mut() else { return };
    s.source_decommitted_in(a, n);
}

/// `[a, a + n)` is no GPU view or CPU view any more (released, or the pair is gone): tiles in it,
/// or over CPU memory in it, show nothing from here on (their video memory goes back), and may be
/// asked for again (the addresses may be given out anew). Call it once the pair is dropped
/// (`kernelx::mirrors::released`): with no tile linked anywhere it takes no state's lock, and a
/// tile linked after that looks its pair up again (`State::take_asks`), so one of the two sees
/// the other.
pub fn forget_mirrors(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    let end = a.saturating_add(n);
    // (out of the queue before out of the set: an ask in between finds the tile still asked
    // for, or stays queued, and the count after it has the thread ask again. The other way
    // round, an ask in between went with the queue and left the tile asked for, never mapped)
    MIRROR_ASKS.lock().unwrap().retain(|&t| !(t < end && t + TILE > a));
    #[cfg(test)]
    if let Some(f) = tests::BETWEEN_FORGETS.with(|f| f.take()) {
        f();
    }
    let mut asked: Vec<usize> = Vec::new();
    MIRROR_ASKED.for_each(|t, _| {
        if t < end && t + TILE > a {
            asked.push(t);
        }
    });
    for t in asked {
        forget_ask(t);
    }
    if lock_sources(&SOURCES).runs.is_empty() {
        return;
    }
    let mut g = lock_in_turn(&STATE, &SUBMITS_WAITING);
    let _held = holding(Holder::Sources);
    forget_released(&mut g);
    let Some(s) = g.as_mut() else { return };
    unsafe { s.forget_mirrors_in(a, end) };
}

/// The CPU address behind window address `va`, for what reads or writes the title's memory
/// there: `va` itself, unless it lies in a GPU view of the title's CPU memory, whose tiles show
/// their CPU view (`State::map_mirrors` takes them by the same rule), or where the page the
/// queue's page mappings put there is kept (wherever that is: the CPU reads and writes any). No
/// `STATE`.
pub fn cpu_of(va: usize) -> usize {
    use crate::hle::kernelx::mirrors::{shows, Shows};
    match shows(va) {
        Shows::Same(c) | Shows::Page { at: c, .. } => c,
        Shows::NoView | Shows::Nothing => va,
    }
}

impl State {
    /// Window tile `w` shows CPU tile `c` (with video memory or not yet).
    fn link_mirror(&mut self, w: usize, c: usize) {
        self.mirror_src.insert(w, c);
        let over = self.mirrors.entry(c).or_default();
        over.push(w);
        if over.len() == 1 {
            early_add(&mut self.src_runs, c, c + TILE);
            // (before the CPU tile's pages are asked about: a decommit from here on tells)
            early_add(&mut lock_sources(self.sources).runs, c, c + TILE);
            MAPPED_GEN.fetch_add(1, Ordering::Release);
        }
    }

    /// Window tile `w` shows no CPU tile any more (its entry in `mapped`, if any, is the caller's),
    /// and may be asked for again. False: it showed none.
    fn unlink_mirror(&mut self, w: usize) -> bool {
        let linked = self.unlink_kept(w);
        if linked {
            forget_ask(w);
        }
        linked
    }

    /// `unlink_mirror`, the ask kept: `w` is linked again (`remap_mirrors`), or waits asked for
    /// until its GPU-view page shows something. (Forgotten, every recording thread's memory of its
    /// asks would empty at each such change, `view_want`.)
    fn unlink_kept(&mut self, w: usize) -> bool {
        let Some(c) = self.mirror_src.remove(&w) else { return false };
        if let Some(over) = self.mirrors.get_mut(&c) {
            over.retain(|&x| x != w);
            if over.is_empty() {
                self.mirrors.remove(&c);
                early_remove(&mut self.src_runs, c, c + TILE);
                early_remove(&mut lock_sources(self.sources).runs, c, c + TILE);
                MAPPED_GEN.fetch_add(1, Ordering::Release);
            }
        }
        self.refused.remove(&w);
        true
    }

    /// Window tile `w` shows its CPU tile no more: its video memory is parked (unmapped before the
    /// next submit), and it may be asked for again.
    fn drop_mirror(&mut self, w: usize) {
        self.unlink_mirror(w);
        self.park(w);
    }

    /// The video memory of window tile `w` over a CPU view is parked (unmapped before the next
    /// submit, unless a placed buffer uses it, or it is taken back first: `new_mirror`).
    fn park(&mut self, w: usize) {
        if self.mapped.get(&w).is_some_and(|e| e.src != 0) {
            let e = self.mapped.remove(&w).expect("a tile over a CPU view");
            while let Some(&piece) = self.to_zero.range((w, 0)..(w + TILE, 0)).next() {
                self.to_zero.remove(&piece);
            }
            self.mirrored -= 1;
            self.give_back(w, e.pool);
        }
    }

    /// The video memory `pool` of window tile `w` over a CPU view goes back (no longer counted in
    /// `mirrored`): parked – or kept mapped there (`XCL_VIEW_KEEP`, `keep_view_tile`), to be taken
    /// back as a parked tile is (`new_mirror`).
    fn give_back(&mut self, w: usize, pool: (u32, u32)) {
        if keep_view_tile(self.diag.keep, self.mirrored + 1 + self.kept.len(), self.mirror_cap) {
            if self.diag.kept == 0 {
                crate::xlog_info!("XCL_VIEW_KEEP: tiles over CPU views keep their video memory once filled ({} tiles at most)", self.mirror_cap);
            }
            self.kept.insert(w, pool);
            self.diag.kept += 1;
        } else {
            self.parked.insert(w, pool);
            self.counts.given_back += 1;
        }
        if self.diag.stats {
            // (a tile that never comes back is not kept for ever)
            if self.diag.left_at.len() >= 1 << 20 {
                self.diag.left_at.clear();
            }
            self.diag.left_at.insert(w, self.diag.submits);
        }
    }

    /// Video memory for window tile `w` over CPU tile `src` (`holes`: its pages not committed
    /// there): taken back from `parked` / `kept` (cleared; it is held already, so past
    /// `mirror_cap` too: a kept tile refused stayed mapped there with old bytes), or a tile of a
    /// pool (cleared when it was used before), mapped with the next flush; all of it goes up at
    /// the next submit (`fill`). `force`: past `mirror_cap` too (it counts all the same), for a
    /// placed buffer's tile: the buffer maps its tiles once, and video memory the window tile got
    /// later would not be the buffer's. False: past `mirror_cap` (logged at powers of two) or no
    /// video memory (now or a moment ago, `no_memory`): `w` is in `refused` then (counted once).
    unsafe fn new_mirror(&mut self, w: usize, src: usize, holes: u16, force: bool) -> bool {
        if let Some(tile) = self.parked.remove(&w).or_else(|| self.kept.remove(&w)) {
            self.mapped.insert(w, TileEntry { src, gen: next_gen(), ..TileEntry::with_holes(tile, holes) });
            self.to_zero.insert((w, TILE));
        } else {
            // (`XCL_VIEW_KEEP`: the tiles kept hold video memory too)
            let held = self.mirrored + if self.diag.keep { self.kept.len() } else { 0 };
            let at_cap = !force && held >= self.mirror_cap;
            if at_cap && self.diag.keep && !self.kept.is_empty() {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| crate::xlog_warn!("XCL_VIEW_KEEP: {} tiles kept and {} with video memory reach the cap of {} (XCL_MIRROR_MB) – tiles are refused from here on, and read zeros", self.kept.len(), self.mirrored, self.mirror_cap));
            }
            // (a moment after a heap could not be made, only tiles at hand: parked ones and those
            // still quarantined or cooling are not)
            self.drain_quarantined();
            let at_hand = !self.free.is_empty() || cooled(&self.cooling, std::time::Instant::now()) || !self.retired.is_empty();
            let wait = at_cap || (!force && self.short_of_memory() && !at_hand);
            let got = if wait { None } else { self.take_tile() };
            let Some((tile, reused)) = got else {
                if !wait {
                    self.no_memory = Some(std::time::Instant::now());
                }
                if self.refused.insert(w) {
                    self.counts.refused += 1;
                    let n = self.counts.refused;
                    if at_cap && n.is_power_of_two() {
                        crate::xlog_warn!("sparse: {n} GPU-view tiles left without video memory (XCL_MIRROR_MB) – the GPU reads zeros there");
                    }
                }
                return false;
            };
            self.mapped.insert(w, TileEntry { src, gen: next_gen(), ..TileEntry::with_holes(tile, holes) });
            if reused {
                self.to_zero.insert((w, TILE));
            }
            self.pending.push(w);
        }
        self.refused.remove(&w);
        self.fill.push(w);
        self.mirrored += 1;
        self.counts.peak = self.counts.peak.max(self.mirrored);
        if self.diag.stats {
            if let Some(at) = self.diag.left_at.remove(&w) {
                self.diag.back[back_bucket(self.diag.submits - at)] += 1;
                self.diag.back_bytes += ((PAGES_PER_TILE - holes.count_ones() as usize) * PAGE) as u64;
            }
        }
        true
    }

    /// A tile over a CPU view found no video memory a moment ago (`no_memory`, 1 s)
    fn short_of_memory(&self) -> bool {
        self.no_memory.is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(1))
    }

    /// Tiles refused (`refused`) wait for video memory again (`later`) once there is room, as
    /// many as there is room for and `MIRROR_ASKS_MAX` at most: what their CPU view held goes up
    /// then (written once before, it may never be written again). One no longer linked or with
    /// video memory leaves `refused`.
    unsafe fn retry_refused(&mut self) {
        if self.refused.is_empty() || self.mirrored >= self.mirror_cap || self.short_of_memory() {
            return;
        }
        let room = (self.mirror_cap - self.mirrored).min(MIRROR_ASKS_MAX);
        let mut take: Vec<(usize, usize)> = Vec::new();
        let mut gone: Vec<usize> = Vec::new();
        for &w in &self.refused {
            if take.len() == room {
                break;
            }
            match self.mirror_src.get(&w) {
                Some(&c) if !self.mapped.contains_key(&w) => take.push((w, c)),
                _ => gone.push(w),
            }
        }
        for w in gone {
            self.refused.remove(&w);
        }
        // (in the order of their CPU tiles: a query for each region of them)
        take.sort_unstable_by_key(|t| t.1);
        self.later.extend(take);
    }

    /// At the top of a submit: tiles shown over their CPU views get video memory and are filled
    /// from them (`fill_later`), the committed pages of `fill_pages` at most (a burst of asks
    /// hashed up to 256 MiB and pushed it through the staging ring with the state's lock held,
    /// on the title's submitting thread). The tiles waiting from submits before go first, in
    /// their order; then the tiles refused before, when there is room again (`retry_refused`);
    /// then the tiles asked for since (`take_asks`).
    unsafe fn map_mirrors(&mut self) {
        let mut budget = self.fill_pages;
        if self.later.is_empty() {
            self.retry_refused();
        }
        loop {
            self.fill_later(&mut budget);
            if budget == 0 || !self.later.is_empty() || self.asks.is_empty() {
                return;
            }
            self.take_asks();
        }
    }

    /// The tiles waiting in `later` (linked, oldest first) get video memory and are filled while
    /// `budget` (committed pages) lasts; one whose CPU tile has no committed page leaves it (and
    /// `refused`: a write there gives it video memory, `mirror_writes`). One refused for want of
    /// room or video memory goes to `refused` (`new_mirror`).
    unsafe fn fill_later(&mut self, budget: &mut usize) {
        /// tiles asked about at a time
        const BATCH: usize = 256;
        while *budget > 0 && !self.later.is_empty() {
            let batch: Vec<(usize, usize)> = self.later.iter().take(BATCH).copied().collect();
            let holes = self.holes_now(&batch.iter().map(|b| b.1).collect::<Vec<_>>());
            let mut done = 0;
            for (&(w, c), h) in batch.iter().zip(holes) {
                if *budget == 0 {
                    break;
                }
                done += 1;
                if self.mirror_src.get(&w) != Some(&c) || self.mapped.contains_key(&w) {
                    continue;
                }
                if h == 0xFFFF {
                    self.refused.remove(&w);
                } else if self.new_mirror(w, c, h, false) {
                    *budget = budget.saturating_sub(PAGES_PER_TILE - h.count_ones() as usize);
                }
            }
            self.later.drain(..done);
            if done < batch.len() {
                return;
            }
        }
    }

    /// The tiles asked for (`asks`, `MIRROR_ASKS_MAX` at most at a time) linked over their CPU
    /// views, to wait for their video memory in `later`: a tile of a sparse window in a GPU view
    /// gets it unless its CPU tile has no committed page (it waits in `mirrors` for the CPU to
    /// write there). Others are left out, and may be asked for again.
    unsafe fn take_asks(&mut self) {
        if self.asks.is_empty() {
            return;
        }
        let n = self.asks.len().min(MIRROR_ASKS_MAX);
        let mut asks: Vec<usize> = self.asks.drain(..n).collect();
        asks.sort_unstable();
        asks.dedup();
        let mut want: Vec<(usize, usize)> = Vec::new();
        for g in asks {
            if self.mirror_src.contains_key(&g) {
                continue;
            }
            let under = if !self.mapped.contains_key(&g) && self.window_index(g).is_some() { tile_shows(g) } else { Under::NoView };
            match under {
                Under::Cpu(c) => want.push((g, c)),
                // (it stays asked for: a page mapped there later links it, `remapped`)
                Under::Nothing => {}
                // (not counted: an address left out at every submit would empty every thread's
                // memory of its asks at every submit; a pair made later changes the views)
                Under::NoView => {
                    MIRROR_ASKED.remove(g);
                }
            }
        }
        if want.is_empty() {
            return;
        }
        #[cfg(test)]
        if let Some(f) = tests::BEFORE_LINK.with(|f| f.take()) {
            f();
        }
        // linked before their CPU tiles' pages are asked about (`holes_now`)
        for &(g, c) in &want {
            self.link_mirror(g, c);
        }
        // (a pair dropped since it was looked up: its `forget_mirrors` may have found nothing
        // linked, and the CPU tile may be given out anew; a page mapped or unmapped there since:
        // linked again, the ask kept)
        want.retain_mut(|(g, c)| match tile_shows(*g) {
            Under::Cpu(now) if now == *c => true,
            Under::Cpu(now) => {
                self.unlink_kept(*g);
                self.link_mirror(*g, now);
                *c = now;
                true
            }
            Under::Nothing => {
                self.unlink_kept(*g);
                false
            }
            Under::NoView => !self.unlink_mirror(*g),
        });
        self.later.extend(want);
    }

    /// Window tile `w`, linked over `from`, follows its page to `to` (`moved`): the link moves. With
    /// video memory and `whole` (the page's 64 KiB were copied to `to`) its entry keeps its pool
    /// tile, gen and sent hashes – nothing is cleared or filled: the known write kernelx notes at
    /// `to` has the next scan send what differs, and a copy back recorded before lands at `to`.
    /// Not `whole` (a page never used, not committed at `to`): its video memory is parked and it
    /// waits to be filled from `to` (`later`), as one with no video memory does.
    fn retarget(&mut self, w: usize, from: usize, to: usize, whole: bool) {
        if self.mirror_src.get(&w) != Some(&from) || from == to {
            return;
        }
        self.unlink_kept(w);
        self.link_mirror(w, to);
        VIEW_TILES[1].fetch_add(1, Ordering::Relaxed);
        match self.mapped.get_mut(&w).filter(|e| e.src == from) {
            Some(e) if whole => {
                e.src = to;
                e.holes = 0;
            }
            Some(_) => {
                self.park(w);
                self.later.push_back((w, to));
            }
            None => self.later.push_back((w, to)),
        }
    }

    /// The tiles whose GPU-view page the queue's page mappings changed (`remaps`, from `remapped`;
    /// `MIRROR_ASKS_MAX` at most a submit) are linked to what they show now: video memory over a
    /// place is cleared and filled from its committed pages (`fill_later`, before the asks), one
    /// that shows nothing is given back – but a placed buffer's, which keeps its video memory
    /// cleared over `no_source` – and stays asked for. Called after the finished copies back land
    /// (a copy back recorded for the old link finds a new gen) and before `map_mirrors`.
    unsafe fn remap_mirrors(&mut self) {
        if self.remaps.is_empty() {
            return;
        }
        let n = self.remaps.len().min(MIRROR_ASKS_MAX);
        let mut tiles: Vec<usize> = self.remaps.drain(..n).collect();
        tiles.sort_unstable();
        tiles.dedup();
        for g in tiles {
            // (a tile of the window's own is not a view's)
            if self.mapped.get(&g).is_some_and(|e| e.src == 0) {
                continue;
            }
            let now = tile_shows(g);
            let linked = self.mirror_src.get(&g).copied();
            match now {
                Under::Cpu(c) if linked == Some(c) => continue,
                Under::Nothing if linked.is_none_or(|l| l == no_source()) => continue,
                _ => {}
            }
            let pinned = self.mapped.get(&g).is_some_and(|e| self.pinned.contains_key(&e.pool));
            self.unlink_kept(g);
            self.park(g);
            if pinned {
                // (a kept tile must not keep the bytes it showed)
                self.to_zero.insert((g, TILE));
            }
            match now {
                Under::Cpu(c) if self.window_index(g).is_some() => {
                    self.link_mirror(g, c);
                    // (it was shown: before the older asks)
                    self.later.push_front((g, c));
                    VIEW_TILES[0].fetch_add(1, Ordering::Relaxed);
                }
                Under::Nothing => {
                    if pinned && no_source() != 0 {
                        self.link_mirror(g, no_source());
                        self.new_mirror(g, no_source(), 0xFFFF, true);
                    }
                    VIEW_TILES[0].fetch_add(1, Ordering::Relaxed);
                }
                _ => forget_ask(g),
            }
        }
    }

    /// The hole masks of CPU tiles `tiles` (`source_holes`), with the pages of decommits under way
    /// as holes too (`Sources::freeing`): Windows has them still, and frees them while the submit
    /// reads them. Asked once the tiles are linked (`link_mirror`): a decommit that begins after
    /// this finds them and tells the state before Windows frees anything; one that began before
    /// is in the list read here.
    unsafe fn holes_now(&self, tiles: &[usize]) -> Vec<u16> {
        let freeing = lock_sources(self.sources).freeing.clone();
        let mut holes = source_holes(tiles);
        for &(a, end) in &freeing {
            for (h, &t) in holes.iter_mut().zip(tiles) {
                let (lo, hi) = (a.max(t), end.min(t + TILE));
                if lo < hi {
                    *h |= ((1u32 << ((hi - t) / PAGE)) - (1u32 << ((lo - t) / PAGE))) as u16;
                }
            }
        }
        holes
    }

    /// Before a submit, the CPU pages written (`dirty`, `page` bytes each) that tiles show: a tile
    /// over one with no video memory gets it now (its CPU tile had no committed page, gave its
    /// memory back, or was refused), and a written page that was a hole there is asked about
    /// again: committed since (a page of the CPU view committed lazily, on its first write), it is
    /// one no more. Every `mapped` insert of a submit is made here, before pointers into the
    /// entries are taken.
    unsafe fn mirror_writes(&mut self, dirty: &[usize], page: usize) {
        if self.mirrors.is_empty() {
            return;
        }
        // CPU tiles written that tiles show, and the pages written there
        let mut written: Vec<(usize, u16)> = Vec::new();
        for &a in dirty {
            let mut x = a & !(PAGE - 1);
            while x < a + page {
                let t = x & !(TILE - 1);
                if self.mirrors.contains_key(&t) {
                    let bit = 1u16 << ((x - t) / PAGE);
                    match written.last_mut() {
                        Some(l) if l.0 == t => l.1 |= bit,
                        _ => written.push((t, bit)),
                    }
                }
                x += PAGE;
            }
        }
        // (tile over it, CPU tile, holes written – None: the tile has no video memory)
        let mut ask: Vec<(usize, usize, Option<u16>)> = Vec::new();
        // (a tile refused is not asked about while there is no room or memory for it: it would be
        // refused again, and its CPU tile may be written at every submit)
        let waiting = self.mirrored >= self.mirror_cap || self.short_of_memory();
        for &(t, bits) in &written {
            for &w in &self.mirrors[&t] {
                match self.mapped.get(&w) {
                    Some(e) if e.src == t => {
                        if e.holes & bits != 0 {
                            ask.push((w, t, Some(e.holes & bits)));
                        }
                    }
                    Some(_) => {}
                    None if waiting && self.refused.contains(&w) => {}
                    None => ask.push((w, t, None)),
                }
            }
        }
        if ask.is_empty() {
            return;
        }
        let now = self.holes_now(&ask.iter().map(|a| a.1).collect::<Vec<_>>());
        for (&(w, t, holes), h) in ask.iter().zip(now) {
            match holes {
                Some(bits) => {
                    self.counts.rechecks += bits.count_ones() as u64;
                    let committed = bits & !h;
                    if let Some(e) = self.mapped.get_mut(&w) {
                        e.holes &= !committed;
                        for i in 0..PAGES_PER_TILE {
                            if committed & (1u16 << i) != 0 {
                                e.sent[i] = None;
                            }
                        }
                    }
                }
                None if h != 0xFFFF && !self.mapped.contains_key(&w) => {
                    self.new_mirror(w, t, h, false);
                }
                None => {}
            }
        }
    }

    /// The CPU tiles of `mirrors` that `[a, end)` touches (by `src_runs`).
    fn src_tiles_in(&self, a: usize, end: usize) -> Vec<usize> {
        let a0 = a & !(TILE - 1);
        let mut out = Vec::new();
        let before = self.src_runs.range(..a0).next_back().map(|(&s, &e)| (s, e));
        for (s, e) in before.into_iter().chain(self.src_runs.range(a0..end).map(|(&s, &e)| (s, e))) {
            let mut t = s.max(a0);
            while t < e.min(end) {
                out.push(t);
                t += TILE;
            }
        }
        out
    }

    /// `source_decommitted` with the state locked: the pieces of the tiles over `[a, a + n)` get
    /// holes (`decommit_tiles`); a tile all holes is parked, and stays in `mirrors` (a commit and
    /// write there gives it video memory again).
    unsafe fn source_decommitted_in(&mut self, a: usize, n: usize) {
        let end = a.saturating_add(n);
        for t in self.src_tiles_in(a, end) {
            let (lo, hi) = (a.max(t), end.min(t + TILE));
            let over = self.mirrors.get(&t).cloned().unwrap_or_default();
            for w in over {
                let Some(pool) = self.mapped.get(&w).filter(|e| e.src == t).map(|e| e.pool) else { continue };
                if !decommit_tiles(&mut self.mapped, &self.pinned, &mut Vec::new(), &mut self.to_zero, w + (lo - t), hi - lo).is_empty() {
                    self.mirrored -= 1;
                    self.give_back(w, pool);
                }
            }
        }
        if self.parked.len() >= PARKED_MAX && self.flush_parked() {
            self.retire_free_pools();
        }
    }

    /// `forget_mirrors` with the state locked (`[a, end)`)
    unsafe fn forget_mirrors_in(&mut self, a: usize, end: usize) {
        let touches = |x: usize| x < end && x + TILE > a;
        let gone: Vec<usize> = self.mirror_src.iter().filter(|&(&w, &c)| touches(w) || touches(c)).map(|(&w, _)| w).collect();
        for w in gone {
            self.drop_mirror(w);
        }
        self.asks.retain(|&w| {
            let keep = !touches(w) && !mirror_source_tile(w).is_some_and(touches);
            if !keep {
                forget_ask(w);
            }
            keep
        });
        if self.parked.len() >= PARKED_MAX && self.flush_parked() {
            self.retire_free_pools();
        }
    }

    /// The tiles of a buffer placed at `[start, start + len)` (`place_buffer`, `start` a tile's),
    /// with video memory where they can have it: a tile of a GPU view shows its CPU view
    /// (`new_mirror`, with no committed page and past `mirror_cap` too: the buffer needs the tile,
    /// and maps it once), the others are the
    /// window's own, with the pages committed there.
    unsafe fn buffer_tiles(&mut self, start: usize, len: usize) -> Vec<(usize, (u32, u32))> {
        let end = start + len;
        let mut over: Vec<(usize, usize)> = Vec::new();
        for t in (start..end).step_by(TILE) {
            if self.mapped.contains_key(&t) || self.window_index(t).is_none() {
                continue;
            }
            if let Some(c) = self.mirror_src.get(&t).copied().or_else(|| buffer_source(tile_shows(t))) {
                over.push((t, c));
            }
        }
        if !over.is_empty() {
            // linked before their CPU tiles' pages are asked about (`holes_now`), and given up
            // when their pair went meanwhile, linked again when their page changed (as in
            // `take_asks`)
            over.retain_mut(|(t, c)| {
                if self.mirror_src.contains_key(t) {
                    return true;
                }
                self.link_mirror(*t, *c);
                match buffer_source(tile_shows(*t)) {
                    Some(now) if now == *c => true,
                    Some(now) => {
                        self.unlink_kept(*t);
                        self.link_mirror(*t, now);
                        *c = now;
                        true
                    }
                    None => !self.unlink_mirror(*t),
                }
            });
            let holes = self.holes_now(&over.iter().map(|o| o.1).collect::<Vec<_>>());
            for (&(t, c), h) in over.iter().zip(holes) {
                self.new_mirror(t, c, h, true);
            }
        }
        // tiles listed with no video memory get it for the buffer (before `ensure_mapped`, which
        // sends what is pending or asks for it soon); one that finds none is left out
        let mut reused = Vec::new();
        self.materialize(start, len, true, &mut reused);
        self.to_zero.extend(reused);
        self.ensure_mapped(start, len, |t| {
            let holes = query_tile_holes(t);
            (holes, !holes)
        });
        (start..end).step_by(TILE).filter_map(|t| self.mapped.get(&t).filter(|e| e.pool != LAZY).map(|e| (t, e.pool))).collect()
    }

    /// `mapped_ranges` of this state
    fn watched_ranges(&self) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = self.runs.iter().chain(self.src_runs.iter()).map(|(&a, &b)| (a, b)).collect();
        if !self.src_runs.is_empty() {
            out.sort_unstable();
        }
        out
    }
}

/// The sparse windows' tiles listed (with video memory, or to get it at their first use,
/// `LAZY`) as sorted (start, end) runs, and the CPU memory
/// tiles of GPU views show (never the views' own addresses): where a CPU write has to reach the
/// GPU.
pub fn mapped_ranges() -> Vec<(usize, usize)> {
    let t0 = super::look_stats::deep().then(std::time::Instant::now);
    let g = lock_for_submit();
    let _held = holding(Holder::Ranges);
    if let Some(t0) = t0 {
        HELD.add(HELD_RANGES_WAIT, t0.elapsed().as_micros() as u64);
    }
    let Some(s) = g.as_ref() else { return Vec::new() };
    s.watched_ranges()
}

const LOOK_FIELDS: usize = 37;
const ACTIVE_US: usize = 0;
const REFRESH_WAIT_US: usize = 1;
const WATCH_US: usize = 2;
const PAGES: usize = 3;
const LOOKS: usize = 4;
const WATCHED: usize = 5;
const RANGES: usize = 6;
const QUEUE_WAIT_US: usize = 7;
const MAINTENANCE_US: usize = 8;
const SCAN_US: usize = 9;
const UPLOAD_US: usize = 10;
const RUNS: usize = 11;
const TRANSFER_BYTES: usize = 12;
const UPLOAD_WAIT_US: usize = 13;
const SUBMIT_GATE_WAIT_US: usize = 14;
const SUBMIT_GATE_HOLD_US: usize = 15;
const READBACK_GATE_WAIT_US: usize = 16;
const READBACK_GATE_HOLD_US: usize = 17;
const READBACK_GATE_RELEASED_US: usize = 18;
const TEXTURE_SELECT_WAIT_US: usize = 19;
const TEXTURE_SELECT_HOLD_US: usize = 20;
const TEXTURES_VISITED: usize = 21;
const TEXTURE_DIRTY_TESTS: usize = 22;
const TEXTURE_MATCHES: usize = 23;
const TEXTURES_RETAINED: usize = 24;
const VRAM_GATE_WAIT_US: usize = 25;
const VRAM_DEVICE_US: usize = 26;
const VRAM_UNMAP_POLL_US: usize = 27;
const VRAM_STATE_US: usize = 28;
const VRAM_QUEUE_WAIT_US: usize = 29;
const VRAM_RETIREMENT_US: usize = 30;
const VRAM_DEVICE_CACHE_HITS: usize = 31;
const VRAM_DEVICE_QUERIES: usize = 32;
const VRAM_CALLS: usize = 33;
const VRAM_MAPPING_WAITS: usize = 34;
const VRAM_RECLAIM_PASSES: usize = 35;
const VRAM_RETIREMENT_PASSES: usize = 36;

/// time spent and pages found by the write-watch scans (`refresh_once`), for the timing line
static WATCH: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// [lock wait, chunk scan, upload] in µs, runs uploaded
static PHASES: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Microseconds so far in [looks for CPU writes, waits for the state lock, chunk scans, uploads].
pub fn timing_totals() -> [u64; 4] {
    [WATCH[0].load(Ordering::Relaxed), PHASES[0].load(Ordering::Relaxed), PHASES[1].load(Ordering::Relaxed), PHASES[2].load(Ordering::Relaxed)]
}

pub fn note_watch(t: std::time::Duration, pages: usize) {
    WATCH[0].fetch_add(t.as_micros() as u64, Ordering::Relaxed);
    WATCH[1].fetch_add(pages as u64, Ordering::Relaxed);
}

/// One complete `XCL_ACTIVITY` look. It is built without shared updates, then published once so
/// the reporting thread cannot split one look across adjacent intervals.
pub(super) struct LookSample([u64; LOOK_FIELDS]);

impl Default for LookSample {
    fn default() -> Self {
        Self([0; LOOK_FIELDS])
    }
}

impl LookSample {
    pub(super) fn add(&mut self, field: usize, value: u64) {
        self.0[field] = self.0[field].wrapping_add(value);
    }

    pub(super) fn add_time(&mut self, field: usize, value: std::time::Duration) {
        self.add(field, value.as_micros() as u64);
    }
}

/// Cumulative complete looks. Disabled title submission never locks or updates this mutex.
static LOOK: Mutex<[u64; LOOK_FIELDS]> = Mutex::new([0; LOOK_FIELDS]);

fn merge_look(total: &mut [u64; LOOK_FIELDS], sample: [u64; LOOK_FIELDS]) {
    for (total, value) in total.iter_mut().zip(sample) {
        *total = total.wrapping_add(value);
    }
}

fn add_gate_times(sample: &mut [u64; LOOK_FIELDS], wait: usize, hold: usize, released: Option<usize>, times: UploadGateTimes) {
    sample[wait] = sample[wait].wrapping_add(times.wait_us);
    sample[hold] = sample[hold].wrapping_add(times.hold_us);
    if let Some(released) = released {
        sample[released] = sample[released].wrapping_add(times.released_us);
    }
}

fn note_readback_gate(times: UploadGateTimes) {
    let mut total = LOOK.lock().unwrap();
    add_gate_times(
        &mut total, READBACK_GATE_WAIT_US, READBACK_GATE_HOLD_US, Some(READBACK_GATE_RELEASED_US), times,
    );
}

/// `XCL_ACTIVITY`: a look's write-watch time and sizes, into its sample (nv's `note_watch` feeds
/// `timing_totals`).
pub(super) fn note_watch_sample(sample: &mut LookSample, t: std::time::Duration, pages: usize, watched: usize, ranges: usize) {
    sample.add_time(WATCH_US, t);
    sample.add(PAGES, pages as u64);
    sample.add(WATCHED, watched as u64);
    sample.add(RANGES, ranges as u64);
}

pub(super) fn note_texture_selection(
    sample: &mut LookSample,
    wait: std::time::Duration,
    hold: std::time::Duration,
    visited: usize,
    dirty_tests: usize,
    matches: usize,
    retained: usize,
) {
    sample.add_time(TEXTURE_SELECT_WAIT_US, wait);
    sample.add_time(TEXTURE_SELECT_HOLD_US, hold);
    sample.add(TEXTURES_VISITED, visited as u64);
    sample.add(TEXTURE_DIRTY_TESTS, dirty_tests as u64);
    sample.add(TEXTURE_MATCHES, matches as u64);
    sample.add(TEXTURES_RETAINED, retained as u64);
}

fn note_vram_before_execute(sample: &mut LookSample, timing: super::vram::BeforeExecuteTiming) {
    sample.add_time(VRAM_GATE_WAIT_US, timing.gate_wait);
    sample.add_time(VRAM_DEVICE_US, timing.device);
    sample.add_time(VRAM_UNMAP_POLL_US, timing.unmap_poll);
    sample.add_time(VRAM_STATE_US, timing.state);
    sample.add_time(VRAM_QUEUE_WAIT_US, timing.queue_wait);
    sample.add_time(VRAM_RETIREMENT_US, timing.retirement);
    sample.add(VRAM_DEVICE_CACHE_HITS, timing.device_cache_hits as u64);
    sample.add(VRAM_DEVICE_QUERIES, timing.device_queries as u64);
    sample.add(VRAM_CALLS, timing.calls as u64);
    sample.add(VRAM_MAPPING_WAITS, timing.mapping_waits as u64);
    sample.add(VRAM_RECLAIM_PASSES, timing.reclaim_passes as u64);
    sample.add(VRAM_RETIREMENT_PASSES, timing.retirement_passes as u64);
}

/// `XCL_ACTIVITY`: one complete look, published (nv's `note_look` marks the look for
/// `XCL_LOOK_STATS=2`).
pub(super) fn note_look_sample(mut sample: LookSample, active: std::time::Duration, refresh_wait: std::time::Duration) {
    if let Some(timing) = super::vram::take_before_execute_timing() {
        note_vram_before_execute(&mut sample, timing);
    }
    sample.add_time(ACTIVE_US, active);
    sample.add_time(REFRESH_WAIT_US, refresh_wait);
    sample.add(LOOKS, 1);
    let mut total = LOOK.lock().unwrap();
    merge_look(&mut total, sample.0);
}

pub fn look_counters() -> [u64; LOOK_FIELDS] {
    *LOOK.lock().unwrap()
}

pub fn look_delta(now: [u64; LOOK_FIELDS], before: [u64; LOOK_FIELDS]) -> [u64; LOOK_FIELDS] {
    std::array::from_fn(|i| now[i].wrapping_sub(before[i]))
}

/// Active foreground elapsed time and serialization wait before title submits in one interval.
pub fn look_line(d: [u64; LOOK_FIELDS], secs: u64) -> String {
    let [
        active, refresh_wait, watch, pages, looks, watched, ranges, queue_wait, maintenance, scan, upload,
        runs, transfer_bytes, upload_wait, submit_gate_wait, submit_gate_hold, readback_gate_wait,
        readback_gate_hold, readback_gate_released, texture_select_wait, texture_select_hold,
        textures_visited, texture_dirty_tests, texture_matches, textures_retained,
        vram_gate_wait, vram_device, vram_unmap_poll, vram_state, vram_queue_wait, vram_retirement,
        vram_device_cache_hits, vram_device_queries, vram_calls, vram_mapping_waits,
        vram_reclaim_passes, vram_retirement_passes,
    ] = d;
    let ms = |us: u64| us / 1000;
    let sparse = queue_wait.saturating_add(maintenance).saturating_add(scan).saturating_add(upload).saturating_add(upload_wait);
    let texture_select = texture_select_wait.saturating_add(texture_select_hold);
    let vram = vram_gate_wait
        .saturating_add(vram_device)
        .saturating_add(vram_unmap_poll)
        .saturating_add(vram_state)
        .saturating_add(vram_queue_wait)
        .saturating_add(vram_retirement);
    format!(
        "looks before submits ({secs} s): {looks} looks did {} ms active work ({:.1} % of the time) after {} ms waiting for the refresh turn: write-watch {} ms ({:.1} MB in {:.1} ranges/look, {pages} pages written); texture select wait/hold {}/{} ms ({textures_visited} visited, {texture_dirty_tests} dirty-tested, {texture_matches} matched, {textures_retained} retained); sparse queue wait/fence+state/scan/upload/ownership wait {}/{}/{}/{}/{} ms ({runs} runs, {:.1} MB zero/dirty bytes); VRAM gate/device/unmap poll/state/queue wait/retire {}/{}/{}/{}/{}/{} ms ({vram_calls} calls, {vram_device_cache_hits} device-cache hits, {vram_device_queries} GetDevice queries, {vram_mapping_waits} mapping waits, {vram_reclaim_passes} reclaim passes, {vram_retirement_passes} retirement passes); upload gate submit wait/hold {}/{} ms, readback wait/hold/released wait+reacquire {}/{}/{} ms; other active work {} ms",
        ms(active),
        active as f64 / (secs.max(1) * 10_000) as f64,
        ms(refresh_wait),
        ms(watch),
        watched as f64 / looks.max(1) as f64 / (1 << 20) as f64,
        ranges as f64 / looks.max(1) as f64,
        ms(texture_select_wait),
        ms(texture_select_hold),
        ms(queue_wait),
        ms(maintenance),
        ms(scan),
        ms(upload),
        ms(upload_wait),
        transfer_bytes as f64 / (1 << 20) as f64,
        ms(vram_gate_wait),
        ms(vram_device),
        ms(vram_unmap_poll),
        ms(vram_state),
        ms(vram_queue_wait),
        ms(vram_retirement),
        ms(submit_gate_wait),
        ms(submit_gate_hold),
        ms(readback_gate_wait),
        ms(readback_gate_hold),
        ms(readback_gate_released),
        ms(active.saturating_sub(watch.saturating_add(sparse).saturating_add(texture_select).saturating_add(vram)))
    )
}

fn poll_unique<F>(mut fences: Vec<F>, key: impl Fn(&F) -> usize, mut completed: impl FnMut(&F) -> u64) -> Vec<(F, u64)> {
    fences.sort_unstable_by_key(&key);
    fences.dedup_by(|a, b| key(a) == key(b));
    fences.into_iter().map(|fence| {
        let value = completed(&fence);
        (fence, value)
    }).collect()
}

fn observed_completion<F>(observed: &[(F, u64)], fence: &F, same: impl Fn(&F, &F) -> bool) -> u64 {
    observed.iter().find_map(|(known, value)| same(known, fence).then_some(*value)).unwrap_or(0)
}

fn reserve_queue_wait(waited: &mut FxHashMap<usize, u64>, queue: usize, value: u64) -> bool {
    if value == 0 || waited.get(&queue).copied().unwrap_or(0) >= value {
        return false;
    }
    waited.insert(queue, value);
    true
}

fn retry_journal(outcome: PreparationOutcome, mut deferred: Vec<usize>, runs: &[(usize, usize)]) -> Vec<usize> {
    if outcome != PreparationOutcome::Ready {
        for &(a, n) in runs {
            let mut pg = a & !(PAGE - 1);
            let end = a.saturating_add(n);
            while pg < end {
                deferred.push(pg);
                pg += PAGE;
            }
        }
    }
    deferred.sort_unstable();
    deferred.dedup();
    deferred
}

fn remember_metric_runs(seen: &mut BTreeMap<usize, usize>, runs: &[(usize, usize)]) {
    for &(a, n) in runs {
        early_add(seen, a, a.saturating_add(n));
    }
}

fn note_transfer_metrics(
    sample: Option<&mut LookSample>, seen: Option<&(BTreeMap<usize, usize>, BTreeMap<usize, usize>)>,
) {
    let (Some(sample), Some((zero, dirty))) = (sample, seen) else { return };
    sample.add(RUNS, (zero.len() + dirty.len()) as u64);
    let bytes = zero.iter().chain(dirty).fold(0usize, |sum, (&a, &b)| sum.saturating_add(b.saturating_sub(a)));
    sample.add(TRANSFER_BYTES, bytes as u64);
}

fn rollback_queue_wait(waited: &mut FxHashMap<usize, u64>, queue: usize, value: u64, previous: Option<u64>) {
    if waited.get(&queue) != Some(&value) {
        return;
    }
    match previous {
        Some(previous) => { waited.insert(queue, previous); }
        None => { waited.remove(&queue); }
    }
}

/// Wait with both `STATE` and `UPLOAD_GATE` released. The caller's shared generation lease keeps
/// every old-state allocator, list, staging byte and destination alive. Only an exact ticket may
/// retire ring storage; a crossing uploader otherwise makes the caller rebuild from current state.
unsafe fn wait_upload_released(ticket: WaitTicket) -> bool {
    loop {
        match crate::gpu::fence::wait(&ticket.fence, ticket.target, &ticket.dev, None, 5000) {
            Ok(()) => break,
            Err(crate::gpu::fence::FenceError::Timeout(_)) => {
                crate::xlog_warn!("sparse: upload retirement fence {} is still pending; STATE remains released", ticket.target);
            }
            Err(e) => {
                crate::xlog_warn!("sparse: upload retirement fence {} failed: {e}", ticket.target);
                return false;
            }
        }
    }
    let proceed = {
        let mut g = lock_state();
        match g.as_mut() {
            None => false,
            Some(s) if s.generation != ticket.state_generation || s.fence.as_raw() != ticket.fence.as_raw() => {
                // The caller's shared lifetime makes replacement impossible. Fail closed if a
                // future teardown path violates that protocol.
                false
            }
            Some(s) if !s.uploads_ordered => false,
            Some(s) => {
                // Another uploader can retire/reuse this slot or ring range after the waited
                // target completes. That invalidates exact ownership but not the completed fence;
                // loop on current state. Reclaim only if neither it nor a remap crossed this ticket.
                if s.wait_ticket_valid(&ticket) {
                    s.reclaim_ring();
                }
                true
            }
        }
    };
    drop(ticket);
    proceed
}

/// Snapshot strong fence owners under `STATE`, then query them without it. A batch or a higher
/// mark added while polling is checked against the same observation; a new fence has no
/// observation and therefore remains quarantined.
unsafe fn poll_retirement_fences() -> Vec<(ID3D12Fence, u64)> {
    let fences = {
        let g = lock_state();
        let Some(s) = g.as_ref() else { return Vec::new() };
        std::iter::once(s.fence.clone())
            .chain(s.quarantined.iter().flat_map(|batch| batch.marks.iter().flatten().map(|(fence, _)| fence.clone())))
            .collect()
    };
    // (XCL's own query: not through the title's GetCompletedValue hook and its settle)
    poll_unique(fences, |fence| fence.as_raw() as usize, |fence| super::real::fence_completed(fence))
}

/// One page to scan: the CPU address of its bytes, its address in its window, and the slot of its
/// hashes (disjoint for every page of one submit: the window pages are distinct and their tiles'
/// entries are only touched through these).
struct PageSlot {
    cpu: usize,
    win: usize,
    slot: *mut Option<Box<[u64; CHUNKS]>>,
}

/// Of the pages `found` (CPU addresses, sorted), how many slots of `pages` the scan went over and
/// how many of those it found unchanged: no run of `runs` (window addresses, sorted, as the scan
/// made them) in the slot's window page.
fn own_scanned(found: &[usize], pages: &[PageSlot], runs: &[(usize, usize)]) -> (usize, usize) {
    if found.is_empty() {
        return (0, 0);
    }
    let (mut scanned, mut unchanged) = (0, 0);
    for p in pages.iter().filter(|p| found.binary_search(&p.cpu).is_ok()) {
        scanned += 1;
        let k = runs.partition_point(|r| r.0 + r.1 <= p.win);
        if !runs.get(k).is_some_and(|r| r.0 < p.win + PAGE) {
            unchanged += 1;
        }
    }
    (scanned, unchanged)
}

/// The pages of one submit, shared with the worker threads (`worker_pool`).
struct SharedPages<'a>(&'a [PageSlot]);
// SAFETY: the slots are disjoint, and the scan finishes before the state's lock is given up
unsafe impl Sync for SharedPages<'_> {}

/// Scan `pages` (`scan_page` on each, the bytes at their CPU address); the changed runs go to
/// `out` as window addresses, in their order. How the work was shared with the workers, if it was.
unsafe fn scan_pages(pages: &[PageSlot], out: &mut Vec<(usize, usize)>) -> Option<super::worker_pool::Shares> {
    /// pages a worker takes at a time, and the number from which the work is shared
    const PIECE: usize = 64;
    const SHARE_FROM: usize = 256;
    let scan = |p: &PageSlot, out: &mut Vec<(usize, usize)>| {
        let bytes = std::slice::from_raw_parts(p.cpu as *const u8, PAGE);
        scan_page(bytes, p.win, &mut *p.slot, out);
    };
    if pages.len() < SHARE_FROM {
        pages.iter().for_each(|p| scan(p, out));
        return None;
    }
    let pieces = pages.len().div_ceil(PIECE);
    let outs: Vec<Mutex<Vec<(usize, usize)>>> = (0..pieces).map(|_| Mutex::new(Vec::new())).collect();
    let shared = SharedPages(pages);
    let work = |k: usize| {
        // (the whole struct, not its field: that is what is Sync)
        let shared = &shared;
        let mut mine = Vec::new();
        for p in &shared.0[k * PIECE..((k + 1) * PIECE).min(shared.0.len())] {
            scan(p, &mut mine);
        }
        *outs[k].lock().unwrap_or_else(|e| e.into_inner()) = mine;
    };
    let shares = super::worker_pool::run(pieces, &work);
    for o in outs {
        for (addr, len) in o.into_inner().unwrap_or_else(|e| e.into_inner()) {
            push_run(out, addr, len);
        }
    }
    Some(shares)
}

/// Copies from the title's memory into the staging ring: shared with the worker threads when
/// there are many bytes.
unsafe fn run_copies(jobs: &mut Vec<(usize, usize, usize)>, staging: *mut u8) {
    const SHARE_FROM: usize = 512 << 10;
    let total: usize = jobs.iter().map(|j| j.2).sum();
    if total < SHARE_FROM || jobs.len() < 2 {
        for &(src, off, n) in jobs.iter() {
            std::ptr::copy_nonoverlapping(src as *const u8, staging.add(off), n);
        }
    } else {
        // pieces of about the same bytes, in the order of the jobs
        let target = (total / 16).max(64 << 10);
        let mut pieces: Vec<(usize, usize)> = Vec::new();
        let (mut from, mut bytes) = (0usize, 0usize);
        for (k, j) in jobs.iter().enumerate() {
            bytes += j.2;
            if bytes >= target {
                pieces.push((from, k + 1));
                from = k + 1;
                bytes = 0;
            }
        }
        if from < jobs.len() {
            pieces.push((from, jobs.len()));
        }
        struct Staging(*mut u8);
        // SAFETY: the copies go to disjoint pieces of the ring, all from the caller's job list
        unsafe impl Sync for Staging {}
        let st = Staging(staging);
        let list: &[(usize, usize, usize)] = jobs;
        super::worker_pool::run(pieces.len(), &|k| {
            let st = &st;
            for &(src, off, n) in &list[pieces[k].0..pieces[k].1] {
                unsafe { std::ptr::copy_nonoverlapping(src as *const u8, st.0.add(off), n) };
            }
        });
    }
    jobs.clear();
}

/// Before the title's ExecuteCommandLists on `queue`: the pages the CPU wrote (`dirty`, 4 KiB
/// pages from the write-watch, sorted) and the reused tiles are copied asynchronously into video
/// memory, and `queue` GPU-waits for that and for new tile mappings.
/// What the title's lists being recorded will write on the GPU in the sparse windows (address,
/// bytes; `will_write`), taken by the next submit: with `lazy_commits` a tile there may have no
/// video memory yet, and the write would go nowhere.
static WRITE_ASKS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
static WRITE_ASKING: AtomicBool = AtomicBool::new(false);

/// A list being recorded writes `[va, va + len)` on the GPU with a memory operation (an
/// immediate, a fill, a copy, a timestamp): with `lazy_commits`, its tiles get their video memory
/// at the next submit, which comes before the list's (`State::give_memory`). Never waits for the
/// state.
pub(super) fn will_write(va: usize, len: usize) {
    if len == 0 || !lazy_commits() {
        return;
    }
    WRITE_ASKS.lock().unwrap_or_else(|e| e.into_inner()).push((va, len));
    WRITE_ASKING.store(true, Ordering::Release);
}

/// microseconds in `before_execute`, the wait for the state's lock with it
static SUBMIT_US: AtomicU64 = AtomicU64::new(0);

/// `XCL_LOOK_STATS=2`: the parts of what `submit_cost` calls "sparse lock" (`PHASES[0]`: the wait
/// for the state's lock and the work before the scan), in µs but where said: a long wait asks for
/// less work under the lock elsewhere, the parts for less of that work here. (The pages of the
/// tiles that came to show a CPU view are gathered after it: the scan split's gather.)
static LOCK_SPLIT: super::look_stats::Sums<11> = super::look_stats::Sums::new();
const LS_SUBMITS: usize = 0;
/// `PHASES[0]` of these submits
const LS_ALL: usize = 1;
const LS_WAIT: usize = 2;
/// landing finished copies back, and their bytes
const LS_BACK: usize = 3;
const LS_BACK_BYTES: usize = 4;
/// the tiles over CPU views (`map_mirrors`, `mirror_writes`)
const LS_VIEWS: usize = 5;
/// `flush_pending`, its tiles and calls
const LS_MAP: usize = 6;
const LS_MAP_TILES: usize = 7;
const LS_MAP_CALLS: usize = 8;
/// `flush_parked` (with the pools it left free), its tiles
const LS_UNMAP: usize = 9;
const LS_UNMAP_TILES: usize = 10;

/// so far: [tiles `flush_pending` mapped, its calls, tiles `flush_parked` unmapped, bytes of
/// copies back landed] (read under the state's lock, which all of them are done with)
static TILE_WORK: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// so far, always counted: the sparse queue's UpdateTileMappings calls and their tiles, [map
/// calls, tiles mapped, unmap calls, tiles unmapped] in the sparse windows, then the same in
/// the view windows (GPU views). A placed buffer's mapping counts with its window's.
static TILE_CALLS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// `calls` UpdateTileMappings of `tiles` tiles of the resource at `base` (`TILE_CALLS`)
fn count_tile_calls(base: usize, unmap: bool, calls: usize, tiles: usize) {
    let k = if super::view_want::is_view(base as u64) { 4 } else { 0 } + if unmap { 2 } else { 0 };
    TILE_CALLS[k].fetch_add(calls as u64, Ordering::Relaxed);
    TILE_CALLS[k + 1].fetch_add(tiles as u64, Ordering::Relaxed);
}

/// `TILE_CALLS` so far, for the periodic log line (`tile_calls_line`)
pub fn tile_calls() -> [u64; 8] {
    std::array::from_fn(|k| TILE_CALLS[k].load(Ordering::Relaxed))
}

/// The tile mapping calls between two `tile_calls`, None when there were none.
pub fn tile_calls_line(now: &[u64; 8], before: &[u64; 8]) -> Option<String> {
    let d: [u64; 8] = std::array::from_fn(|k| now[k].saturating_sub(before[k]));
    if d.iter().all(|&n| n == 0) {
        return None;
    }
    Some(format!(
        "sparse windows {} calls mapping {} tiles, {} calls unmapping {} tiles; GPU views {} calls mapping {} tiles, {} calls unmapping {} tiles",
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]
    ))
}

/// The GPU-view diagnostics' counts (`VS`), all zero but the asks before there is a state.
fn view_stats() -> [u64; VS] {
    let mut x = [0u64; VS];
    x[0] = VIEW_ASKED.load(Ordering::Relaxed);
    let g = lock_state();
    let _held = holding(Holder::Other);
    let Some(s) = g.as_ref() else { return x };
    let d = &s.diag;
    let rest = [s.counts.fills, s.counts.given_back, d.kept, d.back[0], d.back[1], d.back[2], d.back[3], d.back_bytes, d.submits, d.deferred];
    x[1..VS_NOW].copy_from_slice(&rest);
    let now = [s.mirrored, s.kept.len(), s.later.len() + s.asks.len(), s.refused.len()];
    for (k, n) in now.into_iter().enumerate() {
        x[VS_NOW + k] = n as u64;
    }
    x
}

/// The GPU-view diagnostics' line between two `view_stats`, None when nothing was counted.
fn view_stats_line(now: &[u64; VS], before: &[u64; VS]) -> Option<String> {
    let d: [u64; VS_NOW] = std::array::from_fn(|k| now[k].saturating_sub(before[k]));
    if d.iter().all(|&n| n == 0) {
        return None;
    }
    Some(format!(
        "GPU-view tiles (2 s): {} asked for, {} filled, {} given back, {} kept; back after 0-1 / 2-8 / 9-64 / more submits: {} / {} / {} / {} ({:.1} MB filled again); {} of {} submits left asked tiles waiting; now {} with video memory, {} kept, {} waiting, {} refused",
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8] as f64 / (1 << 20) as f64, d[10], d[9],
        now[VS_NOW], now[VS_NOW + 1], now[VS_NOW + 2], now[VS_NOW + 3]
    ))
}

/// `view_stats_line` every 2 s on a thread of its own (`ViewSwitches::stats`; once).
fn start_view_stats() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-view-stats".into()).spawn(|| {
            let mut before = view_stats();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                let now = view_stats();
                if let Some(line) = view_stats_line(&now, &before) {
                    crate::xlog_info!("D3D12.X: {line}");
                }
                before = now;
            }
        });
    });
}

/// The sparse lock split line of `x` (`LOCK_SPLIT`'s sums), as averages a submit.
fn lock_split_line(x: &[u64; 11]) -> String {
    let n = x[LS_SUBMITS].max(1) as f64;
    let ms = |k: usize| x[k] as f64 / n / 1e3;
    let parts = [LS_WAIT, LS_BACK, LS_VIEWS, LS_MAP, LS_UNMAP].iter().map(|&k| x[k]).sum::<u64>();
    format!(
        "sparse lock split ({} submits): wait {:.3} ms, copies back {:.3} ms ({:.0} KB), GPU-view tiles {:.3} ms, mappings {:.3} ms ({:.1} tiles in {:.1} calls), unmappings {:.3} ms ({:.1} tiles), rest {:.3} ms a submit (of {:.3} ms)",
        x[LS_SUBMITS],
        ms(LS_WAIT),
        ms(LS_BACK),
        x[LS_BACK_BYTES] as f64 / n / 1024.0,
        ms(LS_VIEWS),
        ms(LS_MAP),
        x[LS_MAP_TILES] as f64 / n,
        x[LS_MAP_CALLS] as f64 / n,
        ms(LS_UNMAP),
        x[LS_UNMAP_TILES] as f64 / n,
        x[LS_ALL].saturating_sub(parts) as f64 / n / 1e3,
        ms(LS_ALL)
    )
}

impl State {
    /// `before_execute` with the state locked: everything up to the queue's wait. Returns the step
    /// and excess recycled `RbJob`s to drop outside `STATE`. `observed`: the retirement fences'
    /// values read before the lock (`poll_retirement_fences`).
    #[allow(clippy::too_many_arguments)]
    unsafe fn before_execute_on_with(
        &mut self,
        queue: &ID3D12CommandQueue,
        dirty: &[usize],
        page: usize,
        observed: &[(ID3D12Fence, u64)],
        mut look: Option<&mut LookSample>,
        metric_runs: &mut Option<(BTreeMap<usize, usize>, BTreeMap<usize, usize>)>,
        t_started: Option<std::time::Instant>,
    ) -> (SubmitStep, Vec<RbJob>) {
        let t0 = std::time::Instant::now();
        let s = self;
        let q = queue.as_raw() as usize;
        // (`XCL_LOOK_STATS=2`: the time after each step, and the work counted so far)
        let split = super::look_stats::deep();
        let stamp = || split.then(std::time::Instant::now);
        let work0: [u64; 4] = std::array::from_fn(|k| if split { TILE_WORK[k].load(Ordering::Relaxed) } else { 0 });
        // XCL's own writes since the look before, and those of them this look found (before the
        // copies back below land: those the next look finds)
        let own = (split && LOOKED.with(|l| l.replace(false))).then(|| {
            let mut own = std::mem::take(&mut s.own_pages);
            own.sort_unstable();
            own.dedup();
            let found: Vec<usize> = own.iter().copied().filter(|p| dirty.binary_search(p).is_ok()).collect();
            (own.len(), found)
        });
        // Snapshot the exact-generation coordination owners while this State is authoritative. The
        // Arc-backed ledgers may later be used after STATE is released without borrowing replacement
        // state; this parity stage still leaves the legacy upload path authoritative.
        let coordination = s.coordination.clone();
        // (copies back done before the tiles change below: a page that stops being a hole there
        // would take what the GPU had in it)
        let excess = s.land_finished_readbacks();
        let t_back = stamp();
        // tiles whose page the queue's page mappings changed (after the copies back: one for the
        // old link lands nowhere), tiles asked to show CPU views, and the CPU writes that change
        // them: before the flushes below, so their mappings reach the title's work (a tile taken
        // back from `parked` needs none)
        s.remap_mirrors();
        s.map_mirrors();
        s.diag.submits += 1;
        if s.diag.stats && !(s.later.is_empty() && s.asks.is_empty()) {
            s.diag.deferred += 1;
        }
        s.mirror_writes(dirty, page);
        let t_views = stamp();
        // the title's work sees what was committed mapped …
        s.flush_pending();
        PENDING.store(false, Ordering::Release);
        let t_map = stamp();
        // the title's work sees what was decommitted unmapped, as it was (quarantined behind this
        // queue and the title queues `present` knows; pools whose last tiles cooled since go with
        // no tile freed now too: those a flush frees wait for the GPU in `quarantined`, then still
        // cool there: `retire_due` sees them)
        let flushed = s.flush_parked_with(Some(queue));
        s.drain_quarantined_with(|fence| observed_completion(observed, fence, |known, current| known.as_raw() == current.as_raw()));
        if flushed || retire_due(&s.cooling, !s.retired.is_empty() || s.free.has_empty_heap(), s.retire_tried, std::time::Instant::now()) {
            let sparse_done = observed_completion(observed, &s.fence, |known, current| known.as_raw() == current.as_raw());
            s.retire_free_pools_with(sparse_done);
        }
        let t_locked = std::time::Instant::now();
        PHASES[0].fetch_add((t_locked - t0).as_micros() as u64, Ordering::Relaxed);
        if let (Some(t_back), Some(t_views), Some(t_map)) = (t_back, t_views, t_map) {
            let us = |a: std::time::Instant, b: std::time::Instant| (b - a).as_micros() as u64;
            let work = |k: usize| TILE_WORK[k].load(Ordering::Relaxed) - work0[k];
            LOCK_SPLIT.add(LS_ALL, us(t0, t_locked));
            LOCK_SPLIT.add(LS_BACK, us(t0, t_back));
            LOCK_SPLIT.add(LS_BACK_BYTES, work(3));
            LOCK_SPLIT.add(LS_VIEWS, us(t_back, t_views));
            LOCK_SPLIT.add(LS_MAP, us(t_views, t_map));
            LOCK_SPLIT.add(LS_MAP_TILES, work(0));
            LOCK_SPLIT.add(LS_MAP_CALLS, work(1));
            LOCK_SPLIT.add(LS_UNMAP, us(t_map, t_locked));
            LOCK_SPLIT.add(LS_UNMAP_TILES, work(2));
            if LOCK_SPLIT.add(LS_SUBMITS, 1) >= 1024 {
                let x = LOCK_SPLIT.take();
                crate::xlog_info!("D3D12.X: {}; {}", lock_split_line(&x), holders_line(&HELD.take(), x[LS_SUBMITS]));
            }
        }
        let t_maintained = t_started.map(|_| t_locked);
        let zero_runs = zero_runs(std::mem::take(&mut s.to_zero));
        for &(a, n) in &zero_runs {
            s.clear_sent_range(a, n);
        }
        let mut dirty_runs = std::mem::take(&mut s.scratch_runs);
        dirty_runs.clear();
        // pages an upload before could not send (`retry_pages`, window addresses): scanned again
        // with this submit's; one that lost its video memory meanwhile stays journaled
        let mut deferred = Vec::new();
        let retry_pages = std::mem::take(&mut s.retry_pages);
        let mut pages: Vec<PageSlot> = Vec::with_capacity(dirty.len() + retry_pages.len());
        // tiles that came to show a CPU view: all of it (what is committed there)
        let mut fill = std::mem::take(&mut s.fill);
        fill.sort_unstable();
        fill.dedup();
        for &w in &fill {
            let Some(entry) = s.mapped.get_mut(&w).filter(|e| e.src != 0) else { continue };
            s.counts.fills += 1;
            for idx in 0..PAGES_PER_TILE {
                if entry.holes & (1u16 << idx) == 0 {
                    pages.push(PageSlot { cpu: entry.src + idx * PAGE, win: w + idx * PAGE, slot: &mut entry.sent[idx] });
                    s.counts.fill_bytes += PAGE as u64;
                }
            }
        }
        // tiles whose bytes found no video memory at an upload: all that is committed there, once
        // a moment went by since (the upload gives them their tile, or leaves them here again)
        let mut again = false;
        if !s.starved.is_empty() && !s.short_of_memory() {
            let mut starved = std::mem::take(&mut s.starved);
            starved.sort_unstable();
            starved.dedup();
            for &t in &starved {
                let Some(entry) = s.mapped.get_mut(&t).filter(|e| e.src == 0) else { continue };
                again = true;
                for idx in 0..PAGES_PER_TILE {
                    if entry.holes & (1u16 << idx) == 0 {
                        pages.push(PageSlot { cpu: t + idx * PAGE, win: t + idx * PAGE, slot: &mut entry.sent[idx] });
                    }
                }
            }
        }
        let mut i = 0;
        while i < dirty.len() {
            let a = dirty[i];
            let mut n = page;
            i += 1;
            while i < dirty.len() && dirty[i] == a + n {
                n += page;
                i += 1;
            }
            // only what has video memory (committed tiles of a sparse window, and the tiles that
            // show CPU memory written)
            let mut x = a;
            while x < a + n {
                let t = x & !(TILE - 1);
                let end = (t + TILE).min(a + n);
                if let Some(entry) = s.mapped.get_mut(&t).filter(|e| e.src == 0) {
                    let mut pg = x & !(PAGE - 1);
                    while pg < end {
                        let idx = (pg - t) / PAGE;
                        if entry.holes & (1u16 << idx) == 0 {
                            pages.push(PageSlot { cpu: pg, win: pg, slot: &mut entry.sent[idx] });
                        }
                        pg += PAGE;
                    }
                }
                if let Some(over) = s.mirrors.get(&t) {
                    for &w in over {
                        let Some(entry) = s.mapped.get_mut(&w).filter(|e| e.src == t) else { continue };
                        let mut pg = x & !(PAGE - 1);
                        while pg < end {
                            let idx = (pg - t) / PAGE;
                            if entry.holes & (1u16 << idx) == 0 {
                                pages.push(PageSlot { cpu: pg, win: w + (pg - t), slot: &mut entry.sent[idx] });
                            }
                            pg += PAGE;
                        }
                    }
                }
                x = end;
            }
        }
        for &pg in &retry_pages {
            let t = pg & !(TILE - 1);
            let idx = (pg - t) / PAGE;
            let Some(entry) = s.mapped.get_mut(&t).filter(|e| e.holes & (1u16 << idx) == 0) else {
                deferred.push(pg);
                continue;
            };
            // (read at the tile's CPU address: a tile over a CPU view shows that view's bytes)
            pages.push(PageSlot { cpu: entry.cpu(t) + (pg - t), win: pg, slot: &mut entry.sent[idx] });
        }
        if !s.mirrors.is_empty() || again || !retry_pages.is_empty() {
            // (in window order, for the runs; a page filled, written or retried is one slot,
            // scanned once)
            pages.sort_unstable_by_key(|p| p.win);
            pages.dedup_by_key(|p| p.win);
        }
        fill.clear();
        s.fill = fill;
        // (`XCL_LOOK_STATS=2`: the gather of the pages apart from their scan)
        let t_gathered = super::look_stats::deep().then(std::time::Instant::now);
        let shares = scan_pages(&pages, &mut dirty_runs);
        let t_scan_end = std::time::Instant::now();
        super::look_stats::scan(pages.len(), &dirty_runs);
        if let Some(t_gathered) = t_gathered {
            super::look_stats::scan_split(t_gathered - t_locked, t_scan_end - t_gathered, shares);
        }
        if let Some((written, found)) = own {
            let (scanned, unchanged) = own_scanned(&found, &pages, &dirty_runs);
            super::look_stats::own_writes(written, found.len(), scanned, unchanged);
        }
        drop(pages);
        PHASES[1].fetch_add((t_scan_end - t_locked).as_micros() as u64, Ordering::Relaxed);
        PHASES[3].fetch_add((zero_runs.len() + dirty_runs.len()) as u64, Ordering::Relaxed);
        let t_scanned = t_started.map(|_| t_scan_end);
        // Dual-journal exactly what the legacy attempt is about to consume. This sealed batch is a
        // parity record only: the established upload path remains authoritative and retires it as
        // obsolete after publishing its own retry state.
        let legacy_batch = coordination
            .seal_legacy(q as u64, &zero_runs, &retry_pages, &dirty_runs)
            .ok()
            .flatten();
        // Owned planning remains preparatory only (main #2962): capturing and revalidating source
        // hashes both require title-memory reads under STATE. Do not activate it until capture can
        // be made genuinely lock-free and measured on a title workload.
        let state_generation = s.generation;
        let mapping_epoch = s.mapping_epoch;
        let mut legacy_parity = legacy_batch.map(|lease| LegacyParityUpload::Lease {
            lease, state_generation, mapping_epoch,
        });
        if let Some(parity) = legacy_parity.as_ref() {
            match parity.revalidate(s.generation, s.mapping_epoch, |page| s.upload_page_observation(page)) {
                // "Ready" accepts only the legacy attempt. The parity lease stays pending until that
                // authoritative path finishes; coordinated accept/publish remains a later activation.
                LegacyParityRevalidation::Ready => {}
                LegacyParityRevalidation::RestartForGeneration => {
                    let _ = legacy_parity.take().unwrap().retry();
                    return (SubmitStep::Restart, excess);
                }
                LegacyParityRevalidation::RetryCurrentGeneration => {
                    rejournal_zeros(&mut s.to_zero, zero_runs);
                    s.retry_pages = retry_journal(PreparationOutcome::RetryRequired, deferred, &dirty_runs);
                    let _ = legacy_parity.take().unwrap().retry();
                    return (SubmitStep::Restart, excess);
                }
            }
        }
        // Submit tile clears before CPU writes so overlapping CopyBufferRegion destinations have no WAW hazard.
        let mut outcome = PreparationOutcome::Ready;
        let mut upload_wait = None;
        if !zero_runs.is_empty() {
            match s.upload_zeros(&zero_runs) {
                UploadStep::Done(done) => outcome = done,
                UploadStep::Wait(ticket) => {
                    outcome = PreparationOutcome::RetryRequired;
                    upload_wait = Some(ticket);
                }
            }
            if outcome != PreparationOutcome::Ready {
                rejournal_zeros(&mut s.to_zero, zero_runs.clone());
            }
        }
        if outcome == PreparationOutcome::Ready && !dirty_runs.is_empty() {
            match s.upload_dirty(&dirty_runs) {
                UploadStep::Done(done) => outcome = done,
                UploadStep::Wait(ticket) => {
                    outcome = PreparationOutcome::RetryRequired;
                    upload_wait = Some(ticket);
                }
            }
        }
        if s.poison.as_ref().is_some_and(|p| p.at.elapsed() >= std::time::Duration::from_millis(16)) {
            s.refresh_poison();
        }
        PHASES[2].fetch_add(t_scan_end.elapsed().as_micros() as u64, Ordering::Relaxed);
        {
            static CALLS: AtomicU64 = AtomicU64::new(0);
            static BYTES: AtomicU64 = AtomicU64::new(0);
            let bytes: usize = zero_runs.iter().chain(dirty_runs.iter()).map(|r| r.1).sum();
            let n = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
            let b = BYTES.fetch_add(bytes as u64, Ordering::Relaxed) + bytes as u64;
            let us = SUBMIT_US.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed) + t0.elapsed().as_micros() as u64;
            if n.is_power_of_two() && n >= 16 {
                let listed = s.mapped.values().filter(|e| e.pool == LAZY).count();
                crate::xlog_info!("sparse windows: {n} submits, {} MB of CPU writes uploaded, {} ms spent, {} tiles mapped", b >> 20, us / 1000, s.mapped.len() - listed);
                let (own, unsent) = never_sent(&s.mapped);
                crate::xlog_info!("sparse windows: {unsent} of the windows' own {own} mapped tiles ({} of {} MB) never got a byte from the CPU (zeros, or what the GPU wrote); {listed} listed with no video memory (XCL_SPARSE_LAZY)", unsent * TILE >> 20, own * TILE >> 20);
                let ms = |x: &AtomicU64| x.load(Ordering::Relaxed) / 1000;
                let c = &s.counts;
                crate::xlog_info!(
                    "sparse windows timing: write-watch {} ms ({} dirty pages), lock wait {} ms, chunk scan {} ms, upload {} ms ({} runs, {} of them in {} dispatches); GPU views: {} tiles with video memory ({} at most, cap {}), {} filled ({} MB), {} refused, {} given back, {} holes asked again",
                    ms(&WATCH[0]),
                    WATCH[1].load(Ordering::Relaxed),
                    ms(&PHASES[0]),
                    ms(&PHASES[1]),
                    ms(&PHASES[2]),
                    PHASES[3].load(Ordering::Relaxed),
                    SCATTERED[1].load(Ordering::Relaxed),
                    SCATTERED[0].load(Ordering::Relaxed),
                    s.mirrored,
                    c.peak,
                    s.mirror_cap,
                    c.fills,
                    c.fill_bytes >> 20,
                    c.refused,
                    c.given_back,
                    c.rechecks
                );
            }
        }
        s.retry_pages = retry_journal(outcome, deferred, &dirty_runs);
        if let Some(parity) = legacy_parity {
            let _ = parity.obsolete();
        }
        if let Some((zero, dirty)) = metric_runs.as_mut() {
            remember_metric_runs(zero, &zero_runs);
            remember_metric_runs(dirty, &dirty_runs);
        }
        if let (Some(sample), Some(t_started), Some(t_maintained), Some(t_scanned)) =
            (look.as_deref_mut(), t_started, t_maintained, t_scanned)
        {
            let t_uploaded = std::time::Instant::now();
            sample.add_time(MAINTENANCE_US, t_maintained.duration_since(t_started));
            sample.add_time(SCAN_US, t_scanned.duration_since(t_maintained));
            sample.add_time(UPLOAD_US, t_uploaded.duration_since(t_scanned));
        }
        dirty_runs.clear();
        s.scratch_runs = dirty_runs;
        let step = match upload_wait {
            Some(ticket) => SubmitStep::Wait(ticket),
            None => SubmitStep::Done(outcome),
        };
        (step, excess)
    }

    /// The retirement fences' values now (`poll_retirement_fences` on a standalone `State`).
    #[cfg(test)]
    unsafe fn observe_retirement_fences(&self) -> Vec<(ID3D12Fence, u64)> {
        let fences = std::iter::once(self.fence.clone())
            .chain(self.quarantined.iter().flat_map(|batch| batch.marks.iter().flatten().map(|(fence, _)| fence.clone())))
            .collect();
        poll_unique(fences, |fence| fence.as_raw() as usize, |fence| super::real::fence_completed(fence))
    }

    /// `before_execute` on a standalone `State` (tests): its steps under the caller's hold of the
    /// state, a wait for the GPU where an upload needs one, then the queue's wait for the state's
    /// fence. Returns excess recycled `RbJob`s.
    #[cfg(test)]
    unsafe fn before_execute_on(&mut self, queue: &ID3D12CommandQueue, dirty: &[usize], page: usize) -> Vec<RbJob> {
        let mut metric_runs = None;
        let mut excess = Vec::new();
        loop {
            let observed = self.observe_retirement_fences();
            let (step, more) = self.before_execute_on_with(queue, dirty, page, &observed, None, &mut metric_runs, None);
            excess.extend(more);
            match step {
                SubmitStep::Restart => {}
                SubmitStep::Wait(ticket) => {
                    if crate::gpu::fence::wait(&ticket.fence, ticket.target, &ticket.dev, None, 5000).is_err() {
                        return excess;
                    }
                    if self.wait_ticket_valid(&ticket) {
                        self.reclaim_ring();
                    }
                }
                SubmitStep::Done(PreparationOutcome::Ready) => {
                    let q = queue.as_raw() as usize;
                    if reserve_queue_wait(&mut self.waited, q, self.value) {
                        let _ = super::real::native_queue_wait(queue, &self.fence, self.value);
                        super::list_x::note_internal_wait(q, "sparse uploads and tile mappings", &self.fence, self.value);
                    }
                    return excess;
                }
                SubmitStep::Done(_) => return excess,
            }
        }
    }
}

/// Before a title submit on `queue`: what the title committed and decommitted mapped and unmapped
/// on the GPU, the tiles over CPU views and what they show, and the CPU writes at `dirty` (pages
/// of `page` bytes, sorted) uploaded; the queue then waits for all of it. `look`
/// (`XCL_ACTIVITY`) gets the look's times. Not `Ready`: the submit must not go (main's
/// preparation outcome); the journals of what did not go out are kept for the next submit.
pub(super) unsafe fn before_execute(queue: &ID3D12CommandQueue, dirty: &[usize], page: usize, mut look: Option<&mut LookSample>) -> PreparationOutcome {
    if look.is_some() {
        super::vram::begin_before_execute_timing();
    }
    let t_entry = std::time::Instant::now();
    let q = queue.as_raw() as usize;
    let t0 = look.is_some().then_some(t_entry);
    let queue_gate = WAIT_GATES.lock().unwrap().entry(q).or_default().clone();
    let _queue_serial = queue_gate.lock().unwrap();
    let _upload_lifetime = UPLOAD_STATE_LIFETIME.read().unwrap();
    let (mut uploads, mut timing) = UploadGateTiming::acquire(look.is_some());
    let t_gated = t0.map(|_| std::time::Instant::now());
    if let (Some(sample), Some(t0), Some(t_gated)) = (look.as_deref_mut(), t0, t_gated) {
        sample.add_time(QUEUE_WAIT_US, t_gated.duration_since(t0));
    }
    let outcome = before_execute_locked(queue, dirty, page, q, t_entry, t0, &mut look, &mut uploads, &mut timing);
    let times = timing.finish();
    drop(uploads.take());
    if let Some(sample) = look.as_deref_mut() {
        add_gate_times(&mut sample.0, SUBMIT_GATE_WAIT_US, SUBMIT_GATE_HOLD_US, None, times);
    }
    outcome
}

/// What `State::before_execute_on_with` came to.
enum SubmitStep {
    /// prepared (`Ready`), or the outcome the submit gets (the journals were put back)
    Done(PreparationOutcome),
    /// an upload slot or ring room the GPU still holds: waited for with the locks let go, then again
    /// from the start (the journals were put back)
    Wait(WaitTicket),
    /// the state's generation changed under the parity record: again from the start
    Restart,
}

#[allow(clippy::too_many_arguments)]
unsafe fn before_execute_locked(
    queue: &ID3D12CommandQueue,
    dirty: &[usize],
    page: usize,
    q: usize,
    t_entry: std::time::Instant,
    t0: Option<std::time::Instant>,
    look: &mut Option<&mut LookSample>,
    uploads: &mut Option<UploadGuard>,
    timing: &mut UploadGateTiming,
) -> PreparationOutcome {
    let mut metric_runs = look.is_some().then(|| (BTreeMap::new(), BTreeMap::new()));
    'prepare: loop {
    let t_started = t0.map(|_| std::time::Instant::now());
    let observed = poll_retirement_fences();
    let t_lock = std::time::Instant::now();
    // (who had the lock when this began to wait for it: `XCL_LOOK_STATS=2`)
    let behind = (HOLDER.load(Ordering::Relaxed) as usize).min(HOLDERS - 1);
    FLUSHERS_WAITING.fetch_add(1, Ordering::AcqRel);
    let mut g = lock_first(&STATE, &SUBMITS_WAITING);
    FLUSHERS_WAITING.fetch_sub(1, Ordering::AcqRel);
    let held = holding(Holder::Submit);
    let t_got = super::look_stats::deep().then(std::time::Instant::now);
    forget_released(&mut g);
    let Some(s) = g.as_mut() else {
        if let (Some(sample), Some(t_started)) = (look.as_deref_mut(), t_started) {
            sample.add_time(MAINTENANCE_US, t_started.elapsed());
        }
        note_transfer_metrics(look.as_deref_mut(), metric_runs.as_ref());
        return PreparationOutcome::Ready;
    };
    if MIRROR_ASKING.swap(false, Ordering::AcqRel) {
        s.asks.append(&mut MIRROR_ASKS.lock().unwrap());
    }
    if REMAPPING.swap(false, Ordering::AcqRel) {
        take_remaps(&mut s.remaps, &mut REMAPS.lock().unwrap());
    }
    if WRITE_ASKING.swap(false, Ordering::AcqRel) {
        let asks = std::mem::take(&mut *WRITE_ASKS.lock().unwrap_or_else(|e| e.into_inner()));
        s.give_memory(&asks);
    }
    let waited = t_lock.elapsed().as_micros() as u64;
    PHASES[0].fetch_add(waited, Ordering::Relaxed);
    // (the gates before the lock count with it: all of `before_execute` is the submit's sparse time)
    SUBMIT_US.fetch_add(waited + t_lock.duration_since(t_entry).as_micros() as u64, Ordering::Relaxed);
    if let Some(t_got) = t_got {
        LOCK_SPLIT.add(LS_WAIT, (t_got - t_lock).as_micros() as u64);
        LOCK_SPLIT.add(LS_ALL, waited);
        HELD.add(HOLDERS + behind, (t_got - t_lock).as_micros() as u64);
    }
    let (step, excess) = s.before_execute_on_with(queue, dirty, page, &observed, look.as_deref_mut(), &mut metric_runs, t_started);
    let outcome = match step {
        SubmitStep::Restart => {
            drop(held);
            drop(g);
            drop(observed);
            drop(excess);
            continue 'prepare;
        }
        SubmitStep::Wait(ticket) => {
            // `to_zero` and `retry_pages` were restored before ownership is released. Another
            // queue may consume either journal; reacquisition is included in UPLOAD_WAIT_US and the
            // next iteration always rebuilds its plan from current state.
            drop(held);
            drop(g);
            drop(observed);
            drop(excess);
            let wait_started = t0.map(|_| std::time::Instant::now());
            if !timing.with_released(uploads, || wait_upload_released(ticket)) {
                if let (Some(sample), Some(wait_started)) = (look.as_deref_mut(), wait_started) {
                    sample.add_time(UPLOAD_WAIT_US, wait_started.elapsed());
                }
                note_transfer_metrics(look.as_deref_mut(), metric_runs.as_ref());
                return PreparationOutcome::RetryRequired;
            }
            if let (Some(sample), Some(wait_started)) = (look.as_deref_mut(), wait_started) {
                sample.add_time(UPLOAD_WAIT_US, wait_started.elapsed());
            }
            continue 'prepare;
        }
        SubmitStep::Done(outcome) => outcome,
    };
    if outcome != PreparationOutcome::Ready {
        drop(held);
        drop(g);
        drop(observed);
        drop(excess);
        note_transfer_metrics(look.as_deref_mut(), metric_runs.as_ref());
        return outcome;
    }
    // the queue waits for the uploads and tile mappings (outside the lock; given back if it fails)
    let value = s.value;
    let previous = s.waited.get(&q).copied();
    let wait = reserve_queue_wait(&mut s.waited, q, value).then(|| (s.fence.clone(), value, previous));
    drop(held);
    drop(g);
    drop(observed);
    // (recycled copy-back jobs past the pool's cap drop outside all locks)
    drop(excess);
    if let Some((fence, value, previous)) = wait {
        if super::real::native_queue_wait(queue, &fence, value).is_err() {
            let mut g = lock_state();
            if let Some(s) = g.as_mut().filter(|s| s.fence.as_raw() == fence.as_raw() && s.waited.get(&q) == Some(&value)) {
                rollback_queue_wait(&mut s.waited, q, value, previous);
            }
            note_transfer_metrics(look.as_deref_mut(), metric_runs.as_ref());
            return PreparationOutcome::RetryRequired;
        }
        super::list_x::note_internal_wait(q, "sparse uploads and tile mappings", &fence, value);
    }
    note_transfer_metrics(look.as_deref_mut(), metric_runs.as_ref());
    return PreparationOutcome::Ready;
    }
}

/// A buffer the title places at `va` in a sparse window (`bytes` long, `desc` the PC
/// description): a reserved resource over the window's tiles there, starting at the 64 KiB
/// below `va` (returned: its distance from there).
pub unsafe fn place_buffer(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<(crate::util::HRESULT, usize)> {
    let _placement = PLACEMENT.lock().unwrap();
    place_buffer_in(
        &STATE,
        &FORGOTTEN,
        va,
        desc,
        state,
        |d| {
            let hr = super::real::dev_create_reserved_resource(this, d as *const _ as usize, state as usize, 0, riid, ppv);
            (hr >= 0 && !ppv.is_null() && !(*ppv).is_null()).then_some((hr, *ppv))
        },
        |raw| {
            if !ppv.is_null() && *ppv == raw {
                *ppv = std::ptr::null_mut();
            }
            drop(ID3D12Resource::from_raw(raw));
        },
    )
}

unsafe fn place_buffer_in(
    state_lock: &Mutex<Option<State>>,
    forgotten: &Mutex<Vec<usize>>,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    create: impl FnOnce(&D3D12_RESOURCE_DESC) -> Option<(crate::util::HRESULT, *mut c_void)>,
    on_abort: impl FnOnce(*mut c_void),
) -> Option<(crate::util::HRESULT, usize)> {
    let start = va & !(TILE - 1);
    let delta = va - start;
    let bytes = desc.Width as usize;
    if bytes == 0 {
        return None;
    }
    let end = va.checked_add(bytes)?.checked_sub(1)?;
    let generation = {
        let g = lock_state_in(state_lock, forgotten);
        let _held = holding(Holder::Buffers);
        let s = g.as_ref()?;
        s.window_index(start)?;
        s.window_index(end)?;
        s.fence.clone()
    };
    let mut d = *desc;
    d.Width = (delta + bytes) as u64;
    d.Alignment = 0;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // Reserved buffers only accept NONE or ALLOW_UNORDERED_ACCESS (strip console flags and ALLOW_CROSS_ADAPTER).
    d.Flags &= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if state & (D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32 | D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32) != 0 {
        d.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    // Driver creation may call back into a release path which needs `STATE`. Revalidate the
    // strongly owned fence generation before publishing the resulting resource.
    let (hr, raw) = create(&d)?;
    let mut g = lock_state_in(state_lock, forgotten);
    let valid = g.as_ref().is_some_and(|s| {
        s.fence.as_raw() == generation.as_raw() && s.window_index(start).is_some() && s.window_index(end).is_some()
    });
    let Some(res) = valid.then(|| ID3D12Resource::from_raw_borrowed(&raw).cloned()).flatten() else {
        drop(g);
        drop(generation);
        on_abort(raw);
        return None;
    };
    let held = holding(Holder::Buffers);
    let s = g.as_mut().unwrap();
    // (a name for what a device removal reports about an address, `real::log_device_removed`)
    let _ = res.SetName(&windows::core::HSTRING::from(format!("title buffer over tiles at {va:#x} ({bytes:#x} bytes)")));
    // its tiles: the window's video memory at the same addresses; probe actual OS commitment
    // so uncommitted pages (e.g. [start, va) or reserved tails) stay marked as holes (of the CPU
    // view a tile of a GPU view shows)
    let tiles = s.buffer_tiles(start, delta + bytes);
    s.mapping_epoch = s.mapping_epoch.wrapping_add(1);
    // (`XCL_VIEW_POISON`: its tiles without video memory show the poison tile as the window's do)
    s.poison_all(&res, (delta + bytes).div_ceil(TILE) as u32);
    s.map_tiles(&res, start, &tiles);
    if s.buffers.contains_key(&(raw as usize)) {
        forget_in(s, raw as usize);
    }
    for &(_, tile) in &tiles {
        *s.pinned.entry(tile).or_default() += 1;
    }
    s.buffers.insert(raw as usize, tiles);
    s.signal();
    drop(held);
    drop(g);
    drop(res);
    drop(generation);
    Some((hr, delta))
}

/// Placed buffers released while `STATE` was held, let go by the next `lock_state`. Held only
/// to push or take, never across a call out.
static FORGOTTEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// `STATE`, with the buffers released while it was held let go first.
fn lock_state() -> std::sync::MutexGuard<'static, Option<State>> {
    lock_state_in(&STATE, &FORGOTTEN)
}

fn lock_state_in<'a>(state: &'a Mutex<Option<State>>, forgotten: &Mutex<Vec<usize>>) -> std::sync::MutexGuard<'a, Option<State>> {
    let mut g = state.lock().unwrap();
    forget_released_in(&mut g, forgotten);
    g
}

/// Who holds `STATE` (`HOLDER`): the time a submit waits for it is put down to the holder it found
/// there when it began to wait, and each holder's own time under it is summed (`XCL_LOOK_STATS=2`,
/// with the sparse lock split line).
#[derive(Clone, Copy)]
#[repr(u8)]
enum Holder {
    Nobody = 0,
    Submit,
    After,
    Ranges,
    Settle,
    ReadbackThread,
    TileMaps,
    Commits,
    Sources,
    Buffers,
    CpuRead,
    Other,
}
const HOLDERS: usize = 12;
const HOLDER_NAMES: [&str; HOLDERS] = [
    "nobody",
    "submits",
    "after stages",
    "ranges",
    "copies back settled",
    "readback thread",
    "put-off mappings",
    "commits",
    "GPU-view sources",
    "placed buffers",
    "CPU reads",
    "other",
];
static HOLDER: AtomicU8 = AtomicU8::new(0);
/// [µs each holder held the lock], [µs submits waited behind each], then the submit's other
/// waits for it: in its after stage, and for its ranges
static HELD: super::look_stats::Sums<{ 2 * HOLDERS + 2 }> = super::look_stats::Sums::new();
const HELD_AFTER_WAIT: usize = 2 * HOLDERS;
const HELD_RANGES_WAIT: usize = 2 * HOLDERS + 1;

/// `STATE` is held by its holder until this is dropped (declared after the guard: it goes first).
struct Holding(Holder, Option<std::time::Instant>);

fn holding(who: Holder) -> Holding {
    HOLDER.store(who as u8, Ordering::Relaxed);
    Holding(who, super::look_stats::deep().then(std::time::Instant::now))
}

impl Drop for Holding {
    fn drop(&mut self) {
        HOLDER.store(Holder::Nobody as u8, Ordering::Relaxed);
        if let Some(t) = self.1 {
            HELD.add(self.0 as usize, t.elapsed().as_micros() as u64);
        }
    }
}

/// The holders part of the sparse lock split line (`HELD`'s sums over `submits`), in ms a submit:
/// who held the lock how long, and whom the submits' waits for it were behind.
fn holders_line(x: &[u64; 2 * HOLDERS + 2], submits: u64) -> String {
    let n = submits.max(1) as f64;
    let part = |from: usize| -> String {
        let items: Vec<String> = (0..HOLDERS).filter(|&k| x[from + k] > 0).map(|k| format!("{} {:.3}", HOLDER_NAMES[k], x[from + k] as f64 / n / 1e3)).collect();
        if items.is_empty() { "none".into() } else { items.join(", ") }
    };
    format!(
        "held by: {} ms a submit; waits behind: {} ms a submit; the submit's other waits for it: after stage {:.3}, ranges {:.3} ms a submit",
        part(0),
        part(HOLDERS),
        x[HELD_AFTER_WAIT] as f64 / n / 1e3,
        x[HELD_RANGES_WAIT] as f64 / n / 1e3
    )
}

/// `forget_released_in` for the global state (`FORGOTTEN`).
fn forget_released(g: &mut Option<State>) {
    forget_released_in(g, &FORGOTTEN);
}

fn forget_released_in(g: &mut Option<State>, forgotten: &Mutex<Vec<usize>>) {
    let gone = std::mem::take(&mut *forgotten.lock().unwrap());
    let Some(s) = g.as_mut() else { return };
    for res in gone {
        forget_in(s, res);
    }
}

/// A placed buffer is gone: its tiles are not pinned by it any more. This runs from the
/// resource's last Release, which D3D12 can make while it holds a lock of its own (in a title's
/// command list Reset), and `before_execute` calls D3D12 with `STATE` held: waiting for `STATE`
/// here deadlocked the two threads. So it does not wait: with `STATE` held, the holder's next
/// `lock_state` lets the buffer go.
pub fn forget_buffer(res: usize) {
    forget_buffer_in(&STATE, &FORGOTTEN, res);
}

fn forget_buffer_in(state: &Mutex<Option<State>>, forgotten: &Mutex<Vec<usize>>, res: usize) {
    forgotten.lock().unwrap().push(res);
    match state.try_lock() {
        Ok(mut g) => {
            let _held = std::ptr::eq(state, &STATE).then(|| holding(Holder::Other));
            forget_released_in(&mut g, forgotten)
        }
        Err(std::sync::TryLockError::WouldBlock) => {}
        Err(e) => panic!("sparse state: {e}"),
    }
}

fn forget_in(s: &mut State, res: usize) {
    let Some(tiles) = s.buffers.remove(&res) else { return };
    for (t, tile) in tiles {
        if let Some(c) = s.pinned.get_mut(&tile) {
            *c -= 1;
            if *c == 0 {
                s.pinned.remove(&tile);
                s.release_if_unused(t, tile);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_retirement_requires_real_completion_on_every_queue() {
        let reached = |marks: &[(u64, u64)]| retirement_marks_reached(marks, |f| *f);
        assert!(!reached(&[]), "the sparse queue needs its own signalled mark");
        assert!(reached(&[(3, 3), (8, 7)]));
        assert!(!reached(&[(3, 3), (6, 7)]));
        assert!(!reached(&[(3, 3), (u64::MAX, 7)]));
        assert!(!retirement_reached(u64::MAX, 0), "removal never releases a retired heap");
        assert_eq!(next_retirement_value(u64::MAX - 2), Some(u64::MAX - 1));
        assert_eq!(next_retirement_value(u64::MAX - 1), None);
        assert_eq!(next_retirement_value(u64::MAX), None);
    }

    thread_local! {
        /// The VirtualQuery calls `source_holes` made on this thread.
        pub(super) static VIRTUAL_QUERIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// Run by `take_asks` on this thread between the look up of the asked tiles' pairs and
        /// their link (once)
        pub(super) static BEFORE_LINK: std::cell::Cell<Option<Box<dyn FnOnce()>>> = const { std::cell::Cell::new(None) };
        /// Run by `forget_mirrors` on this thread between dropping the queued asks and the asked
        /// set's tiles (once)
        pub(super) static BETWEEN_FORGETS: std::cell::Cell<Option<Box<dyn FnOnce()>>> = const { std::cell::Cell::new(None) };
        /// The asks `forget_ask` counted on this thread (`MIRROR_FORGOTTEN` is every test's)
        pub(super) static ASKS_FORGOTTEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        /// `lazy_commits` for the test on this thread (`LazyCommits`; None: as the environment
        /// says). A thread's own: the other tests' states run beside it in this process.
        pub(super) static LAZY_COMMITS_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
        /// `take_tile` finds no video memory on this thread
        pub(super) static NO_TILES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// The runs of the tiles mapped, made from all of them (but those over a CPU view: their
    /// own addresses are no CPU memory).
    fn runs_of(s: &State) -> BTreeMap<usize, usize> {
        let mut tiles: Vec<usize> = s.mapped.iter().filter(|(_, e)| e.src == 0).map(|(&t, _)| t).collect();
        tiles.sort_unstable();
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for t in tiles {
            match runs.last_mut() {
                Some(last) if last.1 == t => last.1 = t + TILE,
                _ => runs.push((t, t + TILE)),
            }
        }
        runs.into_iter().collect()
    }

    /// The tiles waiting to be linked again keep their order while few; past `MIRROR_ASKS_MAX` each
    /// waits once, however often its page changed.
    #[test]
    fn a_tile_remapped_again_and_again_waits_once() {
        let (mut remaps, mut q) = (Vec::new(), vec![3 * TILE, TILE, 3 * TILE]);
        take_remaps(&mut remaps, &mut q);
        assert!(q.is_empty());
        assert_eq!(remaps, vec![3 * TILE, TILE, 3 * TILE], "few: as they came");
        // the same 100 tiles changed at each of 100 submits, none taken meanwhile
        for _ in 0..100 {
            let mut q: Vec<usize> = (0..100).map(|k| k * TILE).collect();
            take_remaps(&mut remaps, &mut q);
            assert!(remaps.len() <= MIRROR_ASKS_MAX + 100);
        }
        remaps.sort_unstable();
        remaps.dedup();
        assert_eq!(remaps, (0..100).map(|k| k * TILE).collect::<Vec<_>>());
    }

    /// XCL's own writes in a look: a slot counts as scanned when its CPU page is one of them, and
    /// as unchanged when no changed run lies in its window page (a GPU view's slot has its own).
    #[test]
    fn own_writes_scanned_and_found_unchanged() {
        let slot = |cpu: usize, win: usize| PageSlot { cpu, win, slot: std::ptr::null_mut() };
        let (c, w) = (0x10_0000usize, 0x8000_0000usize);
        // CPU pages 0..4, and page 1 shown again by a GPU view's tile at `w`
        let pages = [slot(c, c), slot(c + PAGE, c + PAGE), slot(c + PAGE, w + PAGE), slot(c + 2 * PAGE, c + 2 * PAGE), slot(c + 3 * PAGE, c + 3 * PAGE)];
        // changed: a chunk of page 0, the end of page 1 (own address only), page 3 whole
        let runs = [(c + 0x40, 0x40), (c + 2 * PAGE - CHUNK, CHUNK), (c + 3 * PAGE, PAGE)];
        // XCL wrote pages 1, 2 and 3 (and 9, which was not scanned)
        let found = [c + PAGE, c + 2 * PAGE, c + 3 * PAGE, c + 9 * PAGE];
        assert_eq!(own_scanned(&found, &pages, &runs), (4, 2));
        assert_eq!(own_scanned(&[], &pages, &runs), (0, 0));
        assert_eq!(own_scanned(&found, &pages, &[]), (4, 4));
    }

    /// The sparse lock split: the parts and the rest are the whole, as averages a submit.
    #[test]
    fn the_sparse_lock_parts_and_rest_are_the_whole() {
        let mut x = [0u64; 11];
        x[LS_SUBMITS] = 4;
        x[LS_ALL] = 4_000;
        x[LS_WAIT] = 1_000;
        x[LS_BACK] = 400;
        x[LS_BACK_BYTES] = 8 << 10;
        x[LS_VIEWS] = 200;
        x[LS_MAP] = 1_200;
        x[LS_MAP_TILES] = 10;
        x[LS_MAP_CALLS] = 2;
        x[LS_UNMAP] = 800;
        x[LS_UNMAP_TILES] = 6;
        assert_eq!(
            lock_split_line(&x),
            "sparse lock split (4 submits): wait 0.250 ms, copies back 0.100 ms (2 KB), GPU-view tiles 0.050 ms, mappings 0.300 ms (2.5 tiles in 0.5 calls), unmappings 0.200 ms (1.5 tiles), rest 0.100 ms a submit (of 1.000 ms)"
        );
        // (parts counted over a submit that ended after the sums were taken: no negative rest)
        x[LS_ALL] = 0;
        assert!(lock_split_line(&x).contains("rest 0.000 ms"));
        assert!(lock_split_line(&[0; 11]).starts_with("sparse lock split (0 submits): wait 0.000 ms"));
    }

    /// The holders part of the line: only the holders with time, as averages a submit.
    #[test]
    fn the_holders_line_names_who_held_the_lock_and_whom_the_waits_were_behind() {
        let mut x = [0u64; 2 * HOLDERS + 2];
        x[Holder::Submit as usize] = 4_000;
        x[Holder::TileMaps as usize] = 800;
        x[HOLDERS + Holder::Settle as usize] = 2_000;
        x[HELD_AFTER_WAIT] = 400;
        assert_eq!(
            holders_line(&x, 4),
            "held by: submits 1.000, put-off mappings 0.200 ms a submit; waits behind: copies back settled 0.500 ms a submit; the submit's other waits for it: after stage 0.100, ranges 0.000 ms a submit"
        );
        assert!(holders_line(&[0; 2 * HOLDERS + 2], 0).starts_with("held by: none ms a submit; waits behind: none"));
    }

    /// A device of the test's own (a software one without a GPU).
    unsafe fn test_device() -> ID3D12Device {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
        let mut dev: Option<ID3D12Device> = None;
        if std::env::var_os("XCL_TEST_WARP").is_some() || D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
            D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
        }
        dev.expect("D3D12 device")
    }

    /// Whether `dev` is a software device (WARP, or the Microsoft Basic Render Driver CI runs on,
    /// which is the default adapter there and need not carry DXGI_ADAPTER_FLAG_SOFTWARE).
    unsafe fn software(dev: &ID3D12Device) -> bool {
        use windows::Win32::Graphics::Dxgi::*;
        let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else { return false };
        let adapter: windows::core::Result<IDXGIAdapter1> = factory.EnumAdapterByLuid(dev.GetAdapterLuid());
        adapter.and_then(|a| a.GetDesc1()).is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 || (d.VendorId, d.DeviceId) == (0x1414, 0x8c))
    }

    /// The small ones of an upload's runs go up in dispatches: each lands where it belongs (far
    /// into the window too, next to runs that go up with a copy), and what lies between them keeps
    /// what the GPU has there.
    #[test]
    fn small_runs_go_up_in_dispatches_and_leave_what_lies_between() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        const SIZE: usize = 3 << 30;
        unsafe {
            let dev = test_device();
            // (the software device makes a reserved buffer of 3 GiB and is removed by the first copies
            // into it, 0x887a0020 – and a process has one device an adapter: every test after it failed)
            if software(&dev) {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            }
            let mut reserved: Option<ID3D12Resource> = None;
            if dev.CreateReservedResource(&buffer_desc(SIZE as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved).is_err() {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            }
            let window = reserved.expect("reserved buffer");
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            let mut s = new_state(&dev, vec![(base, window.clone())]).expect("state");
            // (the 1 GiB a dispatch reaches from its base ends between the third and the fourth)
            let tiles = [base, base + TILE, base + (1 << 30) - TILE, base + (1 << 30), base + SIZE - TILE];
            let mut seed = 0x1234_5678_9ABC_DEF1u64;
            let mut random = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            for &t in &tiles {
                assert!(!VirtualAlloc(t as *const c_void, TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
                s.commit_range(t, TILE);
                for i in 0..TILE / 8 {
                    *(t as *mut u64).add(i) = random();
                }
            }
            // all of it goes up, then the CPU writes all of it again and only some runs go up
            let all: Vec<(usize, usize)> = tiles.iter().map(|&t| (t, TILE)).collect();
            s.upload_dirty(&all);
            let on_gpu: Vec<Vec<u8>> = tiles.iter().map(|&t| std::slice::from_raw_parts(t as *const u8, TILE).to_vec()).collect();
            let mut runs: Vec<(usize, usize)> = Vec::new();
            for &t in &tiles {
                for i in 0..TILE / 8 {
                    *(t as *mut u64).add(i) = random();
                }
                let mut o = 0;
                loop {
                    o += CHUNK * (1 + random() as usize % 9);
                    let n = if random() % 23 == 0 { SCATTER_RUN + CHUNK * (random() as usize % 40) } else { CHUNK * (1 + random() as usize % 12) };
                    if o + n > TILE {
                        break;
                    }
                    runs.push((t + o, n));
                    o += n;
                }
            }
            let small = runs.iter().filter(|r| r.1 < SCATTER_RUN).count() as u64;
            let counts = || (SCATTERED[0].load(Ordering::Relaxed), SCATTERED[1].load(Ordering::Relaxed));
            let before = counts();
            s.upload_dirty(&runs);
            let after = counts();
            assert!(small > 300 && runs.len() as u64 > small, "{small} small runs of {}", runs.len());
            assert!(after.0 - before.0 >= 3 && after.1 - before.1 >= small, "{} dispatches of {} runs", after.0 - before.0, after.1 - before.1);

            let mut d = buffer_desc((tiles.len() * TILE) as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            let mut back: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back).expect("readback buffer");
            let back = back.expect("readback buffer");
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            for (k, &t) in tiles.iter().enumerate() {
                list.CopyBufferRegion(&back, (k * TILE) as u64, &window, (t - base) as u64, TILE as u64);
            }
            list.Close().expect("Close");
            s.queue.ExecuteCommandLists(&[Some(list.cast().expect("list"))]);
            s.signal();
            s.wait_for(s.value);
            let mut p = std::ptr::null_mut();
            back.Map(0, None, Some(&mut p)).expect("Map");
            for (k, &t) in tiles.iter().enumerate() {
                let mut want = on_gpu[k].clone();
                for &(a, n) in runs.iter().filter(|r| r.0 >= t && r.0 < t + TILE) {
                    want[a - t..a - t + n].copy_from_slice(std::slice::from_raw_parts(a as *const u8, n));
                }
                let got = std::slice::from_raw_parts((p as *const u8).add(k * TILE), TILE);
                let wrong = (0..TILE / CHUNK).filter(|c| got[c * CHUNK..][..CHUNK] != want[c * CHUNK..][..CHUNK]).collect::<Vec<_>>();
                assert!(wrong.is_empty(), "tile {k}: chunks {:?}… of {} are not what they should be", &wrong[..wrong.len().min(8)], TILE / CHUNK);
            }
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// A copy back after a title submission does not wait for the title's queue: that queue can
    /// be waiting for work the title submits next, from the thread that is in here. The copy
    /// lands once the title's work is done (`drain_readbacks`). Before, the sparse queue
    /// GPU-waited for the title's queue and this thread waited for the copy (5 s, then on).
    #[test]
    fn a_copy_back_does_not_wait_for_the_title_queue() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let base = VirtualAlloc(std::ptr::null(), TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            assert!(!VirtualAlloc(base as *const c_void, TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc(TILE as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved).expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");
            let rbq_queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("readback queue");
            let rbq_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("readback fence");

            let rb_readbacks = Arc::new(SharedReadbacks::new(rb_fence.clone(), Some(rbq_fence.clone())));
            let rb_submit = Arc::new(Mutex::new(RbSubmit::new(dev.clone())));
            let mut s = State {
                generation: 1,
                mapping_epoch: 0,
                coordination: SparseCoordination::new(1, 1).unwrap(),
                dev: dev.clone(),
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                rbq: Some((rbq_queue, rbq_fence)),
                rbq_value: 0,
                readbacks: rb_readbacks,
                rb_submit,
                windows: vec![(base, reserved.clone())],
                pools: Vec::new(),
                free: FreePools::new(),
                cooling: VecDeque::new(),
                retired: Vec::new(),
                retire_tried: std::time::Instant::now(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                kept: FxHashMap::default(),
                pending: Vec::new(),
                quarantined: VecDeque::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: BTreeSet::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                uploads_ordered: true,
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: BTreeMap::new(),
                scratch_runs: Vec::new(),
                own_pages: Vec::new(),
                mirrors: FxHashMap::default(),
                mirror_src: FxHashMap::default(),
                src_runs: BTreeMap::new(),
                sources: &SOURCES,
                fill: Vec::new(),
                asks: Vec::new(),
                remaps: Vec::new(),
                later: VecDeque::new(),
                fill_pages: fill_pages(),
                mirrored: 0,
                mirror_cap: mirror_cap(),
                refused: FxHashSet::default(),
                no_memory: None,
                starved: Vec::new(),
                counts: MirrorCounts::default(),
                diag: ViewDiag::default(),
                poison: None,
                retry_pages: Vec::new(),
            };
            s.commit_range(base, TILE);
            assert!(s.mapped.contains_key(&base));
            // (as before the title's submit)
            s.flush_pending();
            std::ptr::write_bytes(base as *mut u8, 0, 256);

            // the title's queue: blocked on a fence nothing has signalled yet, then a copy of
            // 0..64 into the tile (a GPU write the title's next CPU read expects)
            let title: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate");
            let mut up: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &{ let mut d = buffer_desc(256); d.Flags = D3D12_RESOURCE_FLAG_NONE; d }, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut up)
                .expect("upload buffer");
            let up = up.expect("upload buffer");
            let mut p: *mut c_void = std::ptr::null_mut();
            up.Map(0, None, Some(&mut p)).expect("Map");
            for i in 0..64u32 {
                *(p as *mut u32).add(i as usize) = 0xC0DE_0000 | i;
            }
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            list.CopyBufferRegion(&reserved, 0, &up, 0, 256);
            list.Close().expect("close");
            title.Wait(&s.fence, s.value).expect("wait for the tile mapping");
            title.Wait(&gate, 1).expect("wait for the gate");
            title.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);

            let t0 = std::time::Instant::now();
            s.after_execute(&title, &[(base, 256)], &[]);
            let took = t0.elapsed();
            assert!(took < std::time::Duration::from_secs(1), "the copy back waited {took:?} for the title's queue");
            s.drain_readbacks(Drain::Finished);
            assert_eq!(*(base as *const u32), 0, "nothing lands before the title's work is done");

            // the title's queue goes on: the copy back lands once it is done
            gate.Signal(1).expect("gate signal");
            // as the title would: wait for its own work (here the fences its queue signals: the
            // first once its lists are done, the second once the copy behind them is)
            let t = std::time::Instant::now();
            loop {
                s.drain_readbacks(Drain::TitleDone);
                if *(base as *const u32) == 0xC0DE_0000 {
                    break;
                }
                assert!(t.elapsed() < std::time::Duration::from_secs(5), "the copy back did not land");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            for i in 0..64u32 {
                assert_eq!(*(base as *const u32).add(i as usize), 0xC0DE_0000 | i, "dword {i} after the copy back");
            }
            assert!(s.readbacks.lock().unwrap().rb_jobs.is_empty(), "the copy back was consumed");
            assert_eq!(s.readbacks.lock().unwrap().rb_pool.len(), 1, "its buffer is kept for the next one");
            drop(s);
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn activity_look_line_splits_foreground_costs() {
        let line = look_line([
            830_000,
            120_000,
            610_000,
            4_800,
            240,
            240 * 1_641 << 20,
            240 * 69,
            10_000,
            30_000,
            40_000,
            40_000,
            90,
            96 << 20,
            20_000,
            5_000,
            105_000,
            3_000,
            70_000,
            12_000,
            7_000,
            13_000,
            24_000,
            8_000,
            1_200,
            1_190,
            1_000,
            2_000,
            3_000,
            4_000,
            5_000,
            6_000,
            237,
            3,
            240,
            8,
            2,
            1,
        ], 2);
        assert!(line.contains("240 looks did 830 ms active work (41.5 % of the time) after 120 ms waiting"), "{line}");
        assert!(line.contains("write-watch 610 ms (1641.0 MB in 69.0 ranges/look, 4800 pages written)"), "{line}");
        assert!(line.contains("texture select wait/hold 7/13 ms (24000 visited, 8000 dirty-tested, 1200 matched, 1190 retained)"), "{line}");
        assert!(line.contains("sparse queue wait/fence+state/scan/upload/ownership wait 10/30/40/40/20 ms (90 runs, 96.0 MB zero/dirty bytes)"), "{line}");
        assert!(line.contains("VRAM gate/device/unmap poll/state/queue wait/retire 1/2/3/4/5/6 ms (240 calls, 237 device-cache hits, 3 GetDevice queries, 8 mapping waits, 2 reclaim passes, 1 retirement passes)"), "{line}");
        assert!(line.contains("upload gate submit wait/hold 5/105 ms, readback wait/hold/released wait+reacquire 3/70/12 ms; other active work 39 ms"), "{line}");
        let idle = look_line([0; LOOK_FIELDS], 2);
        assert!(idle.contains("0 looks did 0 ms active work (0.0 % of the time) after 0 ms waiting"), "{idle}");
        assert!(idle.contains("0.0 MB in 0.0 ranges/look"), "{idle}");
    }

    #[test]
    fn activity_look_counter_delta_survives_wrap() {
        assert_eq!(look_delta([0; LOOK_FIELDS], [u64::MAX; LOOK_FIELDS]), [1; LOOK_FIELDS]);
    }

    #[test]
    fn texture_selection_metrics_keep_wait_hold_and_counts_distinct() {
        let mut sample = LookSample::default();
        note_texture_selection(
            &mut sample,
            std::time::Duration::from_micros(7),
            std::time::Duration::from_micros(13),
            101,
            79,
            11,
            10,
        );
        assert_eq!(sample.0[TEXTURE_SELECT_WAIT_US..=TEXTURES_RETAINED], [7, 13, 101, 79, 11, 10]);
    }

    #[test]
    fn vram_metrics_keep_phases_and_counts_distinct() {
        let mut sample = LookSample::default();
        note_vram_before_execute(&mut sample, super::super::vram::BeforeExecuteTiming {
            gate_wait: std::time::Duration::from_micros(1),
            device: std::time::Duration::from_micros(2),
            unmap_poll: std::time::Duration::from_micros(3),
            state: std::time::Duration::from_micros(4),
            queue_wait: std::time::Duration::from_micros(5),
            retirement: std::time::Duration::from_micros(6),
            device_cache_hits: 7,
            device_queries: 8,
            calls: 9,
            mapping_waits: 10,
            reclaim_passes: 11,
            retirement_passes: 12,
        });
        assert_eq!(sample.0[VRAM_GATE_WAIT_US..=VRAM_RETIREMENT_PASSES], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let line = look_line(sample.0, 2);
        assert!(line.contains("9 calls, 7 device-cache hits, 8 GetDevice queries"), "{line}");
    }

    #[test]
    fn upload_gate_times_accumulate_into_distinct_submit_and_readback_fields() {
        let mut sample = [0; LOOK_FIELDS];
        add_gate_times(
            &mut sample,
            SUBMIT_GATE_WAIT_US,
            SUBMIT_GATE_HOLD_US,
            None,
            UploadGateTimes { wait_us: 11, hold_us: 22, released_us: 33 },
        );
        add_gate_times(
            &mut sample,
            READBACK_GATE_WAIT_US,
            READBACK_GATE_HOLD_US,
            Some(READBACK_GATE_RELEASED_US),
            UploadGateTimes { wait_us: 44, hold_us: 55, released_us: 66 },
        );
        assert_eq!(sample[SUBMIT_GATE_WAIT_US], 11);
        assert_eq!(sample[SUBMIT_GATE_HOLD_US], 22);
        assert_eq!(sample[READBACK_GATE_WAIT_US], 44);
        assert_eq!(sample[READBACK_GATE_HOLD_US], 55);
        assert_eq!(sample[READBACK_GATE_RELEASED_US], 66);
    }

    #[test]
    fn retried_upload_ranges_are_counted_once() {
        let mut zero = BTreeMap::new();
        let mut dirty = BTreeMap::new();
        remember_metric_runs(&mut zero, &[(0x1000, 0x1000)]);
        remember_metric_runs(&mut zero, &[(0x1000, 0x1000)]);
        remember_metric_runs(&mut dirty, &[(0x4000, 0x1000)]);
        remember_metric_runs(&mut dirty, &[(0x4000, 0x2000)]);
        let mut sample = LookSample::default();
        note_transfer_metrics(Some(&mut sample), Some(&(zero, dirty)));
        assert_eq!(sample.0[RUNS], 2);
        assert_eq!(sample.0[TRANSFER_BYTES], 0x3000);
    }

    #[test]
    fn activity_look_publishes_one_complete_sample() {
        let mut total = [10; LOOK_FIELDS];
        let sample = std::array::from_fn(|i| i as u64);
        merge_look(&mut total, sample);
        assert_eq!(total, std::array::from_fn(|i| 10 + i as u64));
        merge_look(&mut total, std::array::from_fn(|i| u64::MAX - 9 - i as u64));
        assert_eq!(total, [0; LOOK_FIELDS]);
    }

    #[test]
    fn writable_record_requires_every_touched_page_and_tracks_protection() {
        let mut mapped = FxHashMap::default();
        let (t0, t1) = (0x10_0000usize, 0x11_0000usize);
        mapped.insert(t0, TileEntry::with_holes((0, 1), 0));
        mapped.insert(t1, TileEntry::with_holes((0, 2), 1 << 2));

        assert!(recorded_committed(&mapped, t0, t0 + TILE));
        assert!(recorded_committed(&mapped, t0 + 0x800, t1 + 2 * PAGE));
        assert!(!recorded_committed(&mapped, t1 + 2 * PAGE, t1 + 2 * PAGE + 1));
        assert!(!recorded_committed(&mapped, t1 + TILE - 8, t1 + TILE + 8));

        mark_protection(&mut mapped, t0 + 5 * PAGE + 16, 8, false);
        assert!(!recorded_committed(&mapped, t0 + 5 * PAGE, t0 + 5 * PAGE + 1));
        assert!(recorded_committed(&mapped, t0 + 6 * PAGE, t0 + TILE));
        mark_protection(&mut mapped, t0 + 5 * PAGE, PAGE, true);
        assert!(recorded_committed(&mapped, t0, t0 + TILE));
    }

    #[test]
    fn early_protection_ranges_merge_split_and_commit_clears_them() {
        let base = 0x20_0000usize;
        let mut protected = BTreeMap::new();
        early_protection(&mut protected, base + PAGE + 16, 8, false);
        early_protection(&mut protected, base + 3 * PAGE, 3 * PAGE, false);
        early_protection(&mut protected, base + 5 * PAGE, PAGE, true);
        assert_eq!(protected.into_iter().collect::<Vec<_>>(), [(base + PAGE, base + 2 * PAGE), (base + 3 * PAGE, base + 5 * PAGE)]);

        let mut protected = BTreeMap::from([(base, base + 4 * PAGE)]);
        early_remove(&mut protected, base + PAGE, base + 3 * PAGE);
        assert_eq!(protected.into_iter().collect::<Vec<_>>(), [(base, base + PAGE), (base + 3 * PAGE, base + 4 * PAGE)]);
    }

    #[test]
    fn working_set_check_rejects_resident_read_only_pages_without_touching_cold_pages() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE, PAGE_READONLY};
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert_ne!(base, 0);
            assert!(!VirtualAlloc(base as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            assert!(!VirtualAlloc((base + PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_READONLY).is_null());
            assert!(!VirtualAlloc((base + 2 * PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::write_volatile(base as *mut u8, 1);
            assert_eq!(std::ptr::read_volatile((base + PAGE) as *const u8), 0);
            assert!(resident_pages_writable(base, base + PAGE));
            assert!(!resident_pages_writable(base + PAGE, base + 2 * PAGE));
            assert!(resident_pages_writable(base + 2 * PAGE, base + 3 * PAGE));

            let mut cold: windows_sys::Win32::System::ProcessStatus::PSAPI_WORKING_SET_EX_INFORMATION = std::mem::zeroed();
            cold.VirtualAddress = (base + 2 * PAGE) as *mut c_void;
            assert_ne!(
                windows_sys::Win32::System::ProcessStatus::K32QueryWorkingSetEx(
                    windows_sys::Win32::System::Threading::GetCurrentProcess(),
                    &mut cold as *mut _ as *mut c_void,
                    std::mem::size_of_val(&cold) as u32,
                ),
                0
            );
            assert_eq!(cold.VirtualAttributes.Flags & 1, 0, "the metadata query paged memory in");
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn memory_counts_committed_pages_and_separates_devices_and_retired_pools() {
        let holes = [0xFFFE, 0xFFF0].into_iter(); // one committed page, then four
        let (pools, occupied, retired, committed) = memory_byte_counts(7, 7, 1, 60, 1, holes);
        assert_eq!(pools, POOL_TILES as usize * TILE);
        assert_eq!(occupied, 4 * TILE, "occupied video tiles are independent of CPU page commitment");
        assert_eq!(retired, POOL_TILES as usize * TILE, "the retired pool is not also active");
        assert_eq!(committed, 5 * PAGE, "holes in a 64 KiB tile do not consume CPU commit");
        assert_eq!(memory_byte_counts(8, 7, 1, 60, 1, [0].into_iter()), (0, 0, 0, 0));
    }

    /// `queue()` answers while the state's lock is held: a resource's last `Release` asks for
    /// it (`late_release::drain_known`) from inside a D3D12 call that the lock's holder can be
    /// waiting for (a list's `Reset` against an upload's allocator `Reset`, with `XCL_DRED=1`).
    #[test]
    fn the_queue_is_answered_while_the_state_is_locked() {
        let held = STATE.lock().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let asked = std::thread::spawn(move || {
            let _ = queue();
            let _ = tx.send(());
        });
        let answered = rx.recv_timeout(std::time::Duration::from_secs(5)).is_ok();
        drop(held);
        asked.join().unwrap();
        assert!(answered, "queue() waited for the sparse state's lock");
    }

    /// A submit gets the lock soon although another thread takes it again and again for 0.2 ms each.
    #[test]
    fn a_submit_is_let_in_before_the_commits() {
        use std::sync::atomic::AtomicUsize;
        let (m, waiting, stop) = (Mutex::new(0u64), AtomicUsize::new(0), AtomicBool::new(false));
        std::thread::scope(|s| {
            s.spawn(|| {
                while !stop.load(Ordering::Acquire) {
                    let mut g = lock_in_turn(&m, &waiting);
                    *g += 1;
                    let t = std::time::Instant::now();
                    while t.elapsed().as_micros() < 200 {
                        std::hint::spin_loop();
                    }
                }
            });
            while *m.lock().unwrap() == 0 {
                std::thread::yield_now();
            }
            let mut turns: Vec<u64> = Vec::new();
            for _ in 0..20 {
                let before = *lock_first(&m, &waiting);
                std::thread::sleep(std::time::Duration::from_millis(2));
                let after = *lock_first(&m, &waiting);
                // (the turns the other thread had while this one slept and waited)
                turns.push(after - before);
            }
            stop.store(true, Ordering::Release);
            turns.sort_unstable();
            // 2 ms of sleep are 10 turns; without the rule a waiting thread stood behind hundreds
            assert!(turns[10] < 60, "the other thread's turns before a submit had the lock: {turns:?}");
        });
        assert_eq!(waiting.load(Ordering::Acquire), 0);
    }

    /// A thread that only tries the lock (a fence look settling copies back) does not take it while
    /// a submit waits for it, when it gives way; without that it takes it at once. Its time giving
    /// way does not count against its budget, but it does not give way for ever.
    #[test]
    fn a_thread_that_tries_the_lock_gives_way_to_a_waiting_submit() {
        use std::sync::atomic::AtomicUsize;
        use std::time::Duration;
        let (m, waiting) = (Mutex::new(()), AtomicUsize::new(0));
        // nobody waits: had at once either way
        assert!(try_lock_giving_way(&m, &waiting, Duration::from_millis(5), true).is_some());
        assert!(try_lock_giving_way(&m, &waiting, Duration::from_millis(5), false).is_some());
        // a submit waits for 8 ms: the lock, free all along, is not taken in the first 5 ms
        waiting.store(1, Ordering::Release);
        let t0 = std::time::Instant::now();
        let got = std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(Duration::from_millis(8));
                waiting.store(0, Ordering::Release);
            });
            try_lock_giving_way(&m, &waiting, Duration::from_millis(5), true).is_some()
        });
        let took = t0.elapsed();
        assert!(got, "the lock was free once the submit was done");
        assert!(took >= Duration::from_millis(5), "taken after {took:?}, while the submit waited");
        // a submit that never stops waiting: it is given way to for the budget, then tried
        waiting.store(1, Ordering::Release);
        let t0 = std::time::Instant::now();
        assert!(try_lock_giving_way(&m, &waiting, Duration::from_millis(3), true).is_some());
        assert!(t0.elapsed() >= Duration::from_millis(3));
        assert!(try_lock_giving_way(&m, &waiting, Duration::from_millis(3), false).is_some());
        // held by another thread: given up after the budget (and the time given way)
        let held = m.lock().unwrap();
        let t0 = std::time::Instant::now();
        let got = std::thread::scope(|s| s.spawn(|| try_lock_giving_way(&m, &waiting, Duration::from_millis(3), true).is_some()).join().unwrap());
        assert!(!got);
        assert!(t0.elapsed() >= Duration::from_millis(6), "gave up after {:?}", t0.elapsed());
        drop(held);
    }

    /// A mapping put off does not wait for the title's next submit: a CPU signal sends it, and
    /// without one it goes a few milliseconds after the commit (work the title submitted before
    /// may reach the memory once a value the CPU writes lets it run).
    #[test]
    fn a_mapping_put_off_is_sent_without_a_submit() {
        if super::super::backend() != super::super::Backend::Real {
            return;
        }
        unsafe {
            let _ = super::super::real::device_raw();
            // (a device without sparse windows: commits wait for one, nothing is put off)
            if STATE.lock().unwrap().is_none() {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            }
            let Some((at, len)) = alloc(64 * TILE, TILE) else { return };
            let put_off = || STATE.lock().unwrap().as_ref().is_some_and(|s| s.pending.iter().any(|t| (at..at + len).contains(t)));
            let mapped = || STATE.lock().unwrap().as_ref().is_some_and(|s| (at..at + len).step_by(TILE).all(|t| s.mapped.contains_key(&t)));
            send_pending();
            assert!(mapped() && !put_off(), "a CPU signal sends what is put off");
            decommitted(at, len);
            VirtualFree(at as *mut c_void, len, MEM_DECOMMIT);
            // (as a submit does: the tiles unmapped, so the next commit needs mappings again)
            if let Some(s) = STATE.lock().unwrap().as_mut() {
                s.flush_parked();
            }
            assert!(!VirtualAlloc(at as *const c_void, len, MEM_COMMIT, PAGE_READWRITE).is_null());
            committed(at, len);
            let t0 = std::time::Instant::now();
            while put_off() {
                assert!(t0.elapsed().as_secs() < 5, "nothing sent the mappings");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert!(mapped());
            assert!(free(at));
        }
    }

    #[test]
    fn exact_retirement_forgets_cache_but_inflight_batch_keeps_its_fence() {
        use std::sync::Arc;
        let old = Arc::new(7u32);
        let mut fences = FxHashMap::default();
        fences.insert(0xA, (old.clone(), 3));
        let batch_mark = fences[&0xA].0.clone();

        assert!(fences.contains_key(&0xA), "marker failure must leave the cache pinned");
        let gone = remove_queue_fence(&mut fences, 0xA).expect("exact retirement callback");
        assert!(fences.is_empty() && Arc::ptr_eq(&gone.0, &batch_mark));
        drop(gone);
        drop(old);
        assert_eq!(Arc::strong_count(&batch_mark), 1, "cache removal invalidated an in-flight batch mark");

        let replacement = Arc::new(9u32);
        fences.insert(0xA, (replacement.clone(), 1));
        assert!(!Arc::ptr_eq(&batch_mark, &fences[&0xA].0), "reused raw address kept the retired generation's fence");
    }

    #[test]
    fn retirement_fence_poll_is_unlocked_and_new_requirements_fail_closed() {
        use std::sync::{atomic::{AtomicU64, AtomicUsize, Ordering}, Arc};
        let first = Arc::new(AtomicU64::new(1));
        let late = Arc::new(AtomicU64::new(9));
        let state = Mutex::new(vec![first.clone(), first.clone()]);
        let polls = AtomicUsize::new(0);

        let snapshot = state.lock().unwrap().clone();
        let observed = poll_unique(snapshot, |fence| Arc::as_ptr(fence) as usize, |fence| {
            assert!(state.try_lock().is_ok(), "completion polling held sparse state");
            state.lock().unwrap().push(late.clone());
            polls.fetch_add(1, Ordering::Relaxed);
            fence.load(Ordering::Relaxed)
        });

        assert_eq!(polls.load(Ordering::Relaxed), 1, "the same fence is queried once");
        assert_eq!(observed_completion(&observed, &first, Arc::ptr_eq), 1);
        assert!(observed_completion(&observed, &first, Arc::ptr_eq) < 2, "a higher mark added during polling remains held");
        assert_eq!(observed_completion(&observed, &late, Arc::ptr_eq), 0, "a newly added fence remains held");
    }

    #[test]
    fn title_queue_wait_reservation_is_monotonic_and_per_queue() {
        let mut waited = FxHashMap::default();
        assert!(!reserve_queue_wait(&mut waited, 1, 0));
        assert!(reserve_queue_wait(&mut waited, 1, 4));
        assert!(!reserve_queue_wait(&mut waited, 1, 4));
        assert!(!reserve_queue_wait(&mut waited, 1, 3));
        assert!(reserve_queue_wait(&mut waited, 2, 3));
        assert!(reserve_queue_wait(&mut waited, 1, 5));
        assert_eq!(waited, FxHashMap::from_iter([(1, 5), (2, 3)]));
    }

    #[test]
    fn failed_write_watch_pages_survive_until_a_ready_retry() {
        let runs = [(0x1040, PAGE), (0x2000, CHUNK)];
        assert_eq!(retry_journal(PreparationOutcome::RetryRequired, vec![], &runs), [0x1000, 0x2000]);
        assert!(retry_journal(PreparationOutcome::Ready, vec![], &runs).is_empty());
        assert_eq!(retry_journal(PreparationOutcome::ExecutedUntracked, vec![0x3000], &runs), [0x1000, 0x2000, 0x3000]);
    }

    #[test]
    fn late_claim_completion_cannot_change_a_newer_owner() {
        let mut ledger = WorkClaimLedger::new(11);
        ledger.add_dirty_pages([0x8000]);
        let old = ledger.claim(1).unwrap();
        assert_eq!(ledger.commit(old.token), WorkClaimResolution::Committed);

        ledger.add_retry_pages([0x8000]);
        let newer = ledger.claim(1).unwrap();
        assert_eq!(ledger.restore(old.token), WorkClaimResolution::AlreadyFinished);
        assert_eq!(ledger.owners.get(&0x8000), Some(&newer.token.serial));
        assert_eq!(ledger.commit(newer.token), WorkClaimResolution::Committed);
        assert!(ledger.available.is_empty());
    }

    #[test]
    fn foreground_queue_cannot_pass_or_steal_an_unreceipted_claim() {
        let mut frontier = ClaimFrontier::new(23);
        let planner = frontier.claim(100).unwrap();
        assert_eq!(
            frontier.observe(23, planner.epoch, 200),
            Ok(ClaimFrontierAction::TakeOver(planner)),
            "another queue mistook pending planner work for an empty frontier",
        );

        let helper = frontier.take_over(planner, 200).unwrap();
        assert_eq!(frontier.accept(planner), Err(ClaimFrontierError::StaleToken));
        assert_eq!(frontier.receipt(planner, 7), Err(ClaimFrontierError::StaleToken));
        assert_eq!(frontier.consume_obsolete(planner), Err(ClaimFrontierError::StaleToken));
        assert_eq!(frontier.take_over(planner, 300), Err(ClaimFrontierError::StaleToken));
        assert_eq!(frontier.observe(23, helper.epoch, 200), Ok(ClaimFrontierAction::Progress(helper)));

        frontier.accept(helper).unwrap();
        assert_eq!(frontier.take_over(helper, 100), Err(ClaimFrontierError::WrongState));
        assert_eq!(frontier.consume_obsolete(helper), Err(ClaimFrontierError::WrongState));
        assert_eq!(
            frontier.observe(23, helper.epoch, 100),
            Ok(ClaimFrontierAction::WaitForReceipt(helper)),
            "accepted GPU work was exposed for retry",
        );
        frontier.receipt(helper, 7).unwrap();
        assert_eq!(
            frontier.observe(23, helper.epoch, 100),
            Ok(ClaimFrontierAction::Ready { through: helper.epoch, required_fence: Some(7) }),
        );
    }

    #[test]
    fn receipts_and_obsolete_holes_advance_only_a_contiguous_frontier() {
        let mut frontier = ClaimFrontier::new(29);
        let first = frontier.claim(1).unwrap();
        let second = frontier.claim(2).unwrap();
        let hole = frontier.claim(3).unwrap();
        frontier.accept(first).unwrap();
        frontier.accept(second).unwrap();
        frontier.receipt(second, 11).unwrap();
        frontier.consume_obsolete(hole).unwrap();

        assert_eq!(frontier.retired_through, 0, "out-of-order terminal work crossed the first claim");
        assert_eq!(
            frontier.observe(29, hole.epoch, 9),
            Ok(ClaimFrontierAction::WaitForReceipt(first)),
        );
        frontier.receipt(first, 8).unwrap();
        assert_eq!(frontier.retired_through, hole.epoch, "an obsolete hole left the frontier stuck");
        assert_eq!(
            frontier.observe(29, hole.epoch, 9),
            Ok(ClaimFrontierAction::Ready { through: hole.epoch, required_fence: Some(11) }),
        );
        assert_eq!(
            frontier.observe(29, first.epoch, 9),
            Ok(ClaimFrontierAction::Ready { through: first.epoch, required_fence: Some(8) }),
            "an older observer unnecessarily inherited a newer fence",
        );
    }

    #[test]
    fn frontier_rejects_wrong_generations_unknown_epochs_and_invalid_receipts() {
        let mut frontier = ClaimFrontier::new(31);
        let token = frontier.claim(4).unwrap();
        assert_eq!(frontier.observe(30, token.epoch, 4), Err(ClaimFrontierError::StaleGeneration));
        assert_eq!(frontier.observe(31, token.epoch + 1, 4), Err(ClaimFrontierError::UnknownEpoch));
        assert_eq!(frontier.receipt(token, 1), Err(ClaimFrontierError::WrongState));
        frontier.accept(token).unwrap();
        assert_eq!(frontier.receipt(token, 0), Err(ClaimFrontierError::Fence));
        assert_eq!(frontier.observe(31, 0, 4), Ok(ClaimFrontierAction::Ready { through: 0, required_fence: None }));

        let stale_generation = ClaimFrontierToken { state_generation: 32, ..token };
        assert_eq!(frontier.receipt(stale_generation, 1), Err(ClaimFrontierError::StaleGeneration));
    }

    #[test]
    fn frontier_epoch_and_lease_overflow_fail_without_partial_publication() {
        let mut epoch = ClaimFrontier::new(37);
        epoch.next_epoch = u64::MAX;
        assert_eq!(epoch.claim(1), Err(ClaimFrontierError::Exhausted));
        assert!(epoch.records.is_empty());

        let mut lease = ClaimFrontier::new(41);
        lease.next_lease = u64::MAX;
        assert_eq!(lease.claim(1), Err(ClaimFrontierError::Exhausted));
        assert_eq!(lease.next_epoch, 0);
        assert!(lease.records.is_empty());

        let mut takeover = ClaimFrontier::new(43);
        let token = takeover.claim(1).unwrap();
        takeover.next_lease = u64::MAX;
        assert_eq!(takeover.take_over(token, 2), Err(ClaimFrontierError::Exhausted));
        assert_eq!(takeover.observe(43, token.epoch, 1), Ok(ClaimFrontierAction::Progress(token)));
    }

    #[test]
    fn work_claim_ledger_merges_orders_and_restores_exact_out_of_order_work() {
        let mut ledger = WorkClaimLedger::new(7);
        ledger.add_zero_ranges(&[(0x3000, 2 * PAGE)]);
        ledger.add_retry_pages([0x3001, 0x1000, 0x1000]);
        ledger.add_dirty_pages([0x4000, 0x6001, 0x6000]);

        let first = ledger.claim(2).unwrap();
        assert_eq!(first.pages.iter().map(|(page, reasons)| (*page, reasons.0)).collect::<Vec<_>>(), [
            (0x1000, WorkReasons::RETRY.0),
            (0x3000, WorkReasons::ZERO.0 | WorkReasons::RETRY.0),
        ]);
        let second = ledger.claim(8).unwrap();
        assert_eq!(second.pages.iter().map(|(page, reasons)| (*page, reasons.0)).collect::<Vec<_>>(), [
            (0x4000, WorkReasons::ZERO.0 | WorkReasons::DIRTY.0),
            (0x6000, WorkReasons::DIRTY.0),
        ]);

        ledger.add_dirty_pages([0x1000]);
        assert!(ledger.claim(1).is_none(), "new work escaped an older owner of the same page");
        assert_eq!(ledger.cancel(second.token), WorkClaimResolution::Restored);
        let out_of_order = ledger.claim(1).unwrap();
        assert_eq!(out_of_order.pages[0].0, 0x4000);
        assert_eq!(ledger.commit(first.token), WorkClaimResolution::Committed);
        let later = ledger.claim(1).unwrap();
        assert_eq!(later.pages, [(0x1000, WorkReasons::DIRTY)]);
        assert_eq!(ledger.commit(later.token), WorkClaimResolution::Committed);
        assert_eq!(ledger.cancel(out_of_order.token), WorkClaimResolution::Restored);
        let restored = ledger.claim(8).unwrap();
        assert_eq!(restored.pages.iter().map(|(page, _)| *page).collect::<Vec<_>>(), [0x4000, 0x6000]);
    }

    #[test]
    fn replaced_sparse_batch_handles_reject_stale_additions() {
        let batches = SparseBatches::new(17);
        let stale_clone = batches.clone();
        let current = batches.replace_generation(18).unwrap();
        assert!(!batches.add_dirty_pages([0xC000]));
        assert!(!stale_clone.add_retry_pages([0xD000]));
        assert!(matches!(batches.seal(1, 1), Err(SparseBatchError::StaleGeneration)));
        assert!(matches!(stale_clone.replace_generation(19), Err(SparseBatchError::StaleGeneration)));
        assert!(current.add_dirty_pages([0xB000]));
        assert_eq!(current.seal(1, 1).unwrap().unwrap().pages()[0].0, 0xB000);
    }

    #[test]
    fn sparse_batch_takeover_revokes_without_restoring_or_duplicating_work() {
        let batches = SparseBatches::new(47);
        assert!(batches.add_dirty_pages([0x1_1000]));
        let old = batches.seal(1, 1).unwrap().unwrap();
        let through = batches.through().unwrap();
        assert_eq!(old.pages()[0].0, 0x1_1000);

        let replacement = old.take_over(2).unwrap();
        assert!(matches!(
            old.obsolete(),
            Err(SparseBatchError::Frontier(ClaimFrontierError::StaleToken)),
        ));
        assert_eq!(replacement.pages()[0].0, 0x1_1000);
        replacement.accept().unwrap().receipt(5).unwrap();
        assert_eq!(
            batches.observe(through, 9),
            Ok(ClaimFrontierAction::Ready { through, required_fence: Some(5) }),
            "the revoked owner restored or duplicated its work",
        );
        assert!(batches.seal(3, 1).unwrap().is_none());
    }

    #[test]
    fn dropped_pending_batch_remains_visible_for_foreground_takeover() {
        let batches = SparseBatches::new(53);
        assert!(batches.add_dirty_pages([0x2_1000]));
        let abandoned = batches.seal(10, 1).unwrap().unwrap();
        let token = abandoned.token;
        let through = batches.through().unwrap();
        drop(abandoned);
        assert_eq!(batches.observe(through, 20), Ok(ClaimFrontierAction::TakeOver(token)));

        let foreground = batches.take_over(token, 20).unwrap();
        foreground.accept().unwrap().receipt(7).unwrap();
        assert_eq!(
            batches.observe(through, 20),
            Ok(ClaimFrontierAction::Ready { through, required_fence: Some(7) }),
        );
    }

    #[test]
    fn sparse_batch_producers_are_collected_before_locking() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let batches = SparseBatches::new(57);
        let unlocked = AtomicBool::new(false);
        let ledger = batches.ledger.clone();
        let pages = std::iter::once_with(|| { unlocked.store(ledger.try_lock().is_ok(), Ordering::Relaxed); 0x2_2001 });
        assert!(batches.add_dirty_pages(pages));
        assert!(unlocked.load(Ordering::Relaxed), "a producer iterator ran under the batch lock");
        assert_eq!(batches.seal(1, 1).unwrap().unwrap().pages()[0].0, 0x2_2000);
    }

    #[test]
    fn sparse_batch_ready_requires_new_reasons_to_be_sealed() {
        let batches = SparseBatches::new(59);
        assert!(batches.add_dirty_pages([0x3_1000]));
        let first = batches.seal(1, 1).unwrap().unwrap();
        let first_epoch = batches.through().unwrap();
        assert!(batches.add_dirty_pages([0x3_1001]));
        assert_eq!(batches.through(), Err(SparseBatchError::Unsealed));
        assert_eq!(batches.observe(first_epoch, 2), Err(SparseBatchError::Unsealed));

        first.accept().unwrap().receipt(8).unwrap();
        assert_eq!(batches.through(), Err(SparseBatchError::Unsealed));
        let successor = batches.seal(2, 1).unwrap().unwrap();
        let through = batches.through().unwrap();
        assert!(through > first_epoch);
        successor.obsolete().unwrap();
        assert_eq!(
            batches.observe(through, 2),
            Ok(ClaimFrontierAction::Ready { through, required_fence: Some(8) }),
        );
    }

    #[test]
    fn sparse_batch_receipts_are_contiguous_and_acceptance_is_fail_closed() {
        let batches = SparseBatches::new(61);
        assert!(batches.add_dirty_pages([0x4_1000, 0x4_2000]));
        let first = batches.seal(1, 1).unwrap().unwrap();
        let second = batches.seal(2, 1).unwrap().unwrap();
        let through = batches.through().unwrap();
        let first_token = first.token;
        let first = first.accept().unwrap();
        let second = second.accept().unwrap();
        second.receipt(11).unwrap();
        assert_eq!(batches.observe(through, 3), Ok(ClaimFrontierAction::WaitForReceipt(first_token)));
        first.receipt(8).unwrap();
        assert_eq!(
            batches.observe(through, 3),
            Ok(ClaimFrontierAction::Ready { through, required_fence: Some(11) }),
        );

        assert!(batches.add_dirty_pages([0x4_3000]));
        let accepted = batches.seal(4, 1).unwrap().unwrap().accept().unwrap();
        let accepted_token = accepted.token.unwrap().0;
        assert!(matches!(
            batches.take_over(accepted_token, 5),
            Err(SparseBatchError::Frontier(ClaimFrontierError::WrongState)),
        ));
        drop(accepted);
        assert_eq!(batches.through(), Err(SparseBatchError::Poisoned));

        let current = batches.replace_generation(62).unwrap();
        assert_eq!(batches.through(), Err(SparseBatchError::StaleGeneration));
        assert!(matches!(current.replace_generation(61), Err(SparseBatchError::StaleGeneration)));
        assert!(matches!(current.take_over(accepted_token, 6), Err(SparseBatchError::StaleGeneration)));
        assert!(current.add_dirty_pages([0x5_1000]));
        let stale = current.seal(6, 1).unwrap().unwrap();
        let replacement = current.replace_generation(63).unwrap();
        assert_eq!(stale.obsolete(), Err(SparseBatchError::StaleGeneration));
        assert_eq!(replacement.through(), Ok(0));
        assert!(matches!(SparseBatches::new(u64::MAX).replace_generation(0), Err(SparseBatchError::StaleGeneration)));
    }

    #[test]
    fn live_sparse_coordination_owns_one_exact_generation() {
        let coordination = SparseCoordination::new(101, 501).unwrap();
        {
            let batches = coordination.batches.ledger.lock().unwrap();
            assert_eq!(batches.work.state_generation, 101);
            assert_eq!(batches.fence, Some(SparseFenceIdentity { state_generation: 101, allocation: 501 }));
        }
        assert_eq!(coordination.waits.fence(), SparseFenceIdentity { state_generation: 101, allocation: 501 });
        let reservation = coordination.reservations.reserve(0, RING_ALIGN).unwrap();
        assert_eq!(reservation.token.unwrap().state_generation, 101);
        assert!(SparseCoordination::new(0, 501).is_none());
        assert!(SparseCoordination::new(101, 0).is_none());
    }

    #[test]
    fn legacy_sparse_attempt_is_mirrored_and_retired_as_one_exact_batch() {
        let coordination = SparseCoordination::new(103, 503).unwrap();
        let snapshot = coordination.clone();
        assert!(Arc::ptr_eq(&coordination.batches.ledger, &snapshot.batches.ledger));

        let batch = snapshot.seal_legacy(
            0xA,
            &[(0x3000, 2 * PAGE)],
            &[0x3001, 0x1000],
            &[(0x4000 + CHUNK, CHUNK)],
        ).unwrap().unwrap();
        assert_eq!(batch.token.state_generation, 103);
        assert_eq!(
            batch.pages().iter().map(|(page, reasons)| (*page, reasons.0)).collect::<Vec<_>>(),
            [
                (0x1000, WorkReasons::RETRY.0),
                (0x3000, WorkReasons::ZERO.0 | WorkReasons::RETRY.0),
                (0x4000, WorkReasons::ZERO.0 | WorkReasons::DIRTY.0),
            ],
        );
        let through = snapshot.batches.through().unwrap();
        batch.obsolete().unwrap();
        assert_eq!(
            coordination.batches.observe(through, 0xB),
            Ok(ClaimFrontierAction::Ready { through, required_fence: None }),
            "the parity batch remained visible after the legacy path retired it",
        );
        assert!(coordination.batches.seal(0xB, usize::MAX).unwrap().is_none());
    }

    fn wait_ready(fence: SparseFenceIdentity, through: u64, value: Option<u64>) -> SparseReadyReceipt {
        SparseReadyReceipt { state_generation: fence.state_generation, fence, through, required_fence: value }
    }

    fn wait_reservation(decision: ForegroundWaitDecision) -> ForegroundWaitReservation {
        match decision { ForegroundWaitDecision::Reserve(reservation) => reservation, _ => panic!("wait not reserved") }
    }

    #[test]
    fn foreground_wait_pairs_out_of_order_frontier_with_exact_fence() {
        let batches = SparseBatches::with_fence(101, 501).unwrap();
        assert!(batches.add_dirty_pages([0x6_1000, 0x6_2000]));
        let first = batches.seal(1, 1).unwrap().unwrap();
        let first_token = first.token;
        let second = batches.seal(2, 1).unwrap().unwrap();
        let through = batches.through().unwrap();
        let first = first.accept().unwrap();
        second.accept().unwrap().receipt(11).unwrap();
        let waits = ForegroundWaits::new(101, 501).unwrap();
        let queue = waits.register_queue(0xA).unwrap();
        assert_eq!(waits.register_queue(0xA), Ok(queue));
        assert_eq!(batches.ready_receipt(through, 3), Err(ForegroundWaitError::NotReady));
        assert_eq!(batches.observe(through, 3), Ok(ClaimFrontierAction::WaitForReceipt(first_token)));
        first.receipt(8).unwrap();

        let ready = batches.ready_receipt(through, 3).unwrap();
        let wrong = SparseReadyReceipt { fence: SparseFenceIdentity { allocation: 502, ..ready.fence }, ..ready };
        assert!(matches!(waits.reserve(queue, wrong), Err(ForegroundWaitError::Fence)));
        let reservation = wait_reservation(waits.reserve(queue, ready).unwrap());
        assert_eq!(reservation.arm().unwrap().publish(), Ok((through, 11)));
        assert!(matches!(waits.reserve(queue, ready), Ok(ForegroundWaitDecision::Installed { through: t, value: 11 }) if t == through));
        assert!(matches!(waits.reserve(queue, wait_ready(waits.fence(), through + 1, None)),
            Ok(ForegroundWaitDecision::NoWait { through: t }) if t == through + 1));

        let other = SparseBatches::with_fence(101, 502).unwrap();
        let other_waits = ForegroundWaits::new(101, 502).unwrap();
        let other_queue = other_waits.register_queue(0xE).unwrap();
        assert!(matches!(other.ready_receipt(through, 3), Err(ForegroundWaitError::Batch(_))));
        assert!(matches!(other_waits.reserve(other_queue, ready), Err(ForegroundWaitError::Fence)));

        assert!(matches!(batches.replace_generation_with_fence(102, 501), Err(SparseBatchError::Fence)));
        let stale = batches.clone();
        let batches = batches.replace_generation_with_fence(102, 503).unwrap();
        assert_eq!(batches.ledger.lock().unwrap().fence.unwrap().allocation, 503);
        assert_eq!(stale.ready_receipt(through, 3), Err(ForegroundWaitError::Batch(SparseBatchError::StaleGeneration)))
    }

    #[test]
    fn foreground_wait_rollbacks_are_exact_and_queue_reuse_is_stale() {
        let waits = ForegroundWaits::new(103, 601).unwrap();
        let fence = waits.fence();
        let queue = waits.register_queue(0xB).unwrap();
        let ready = wait_ready(fence, 1, Some(5));
        let abandoned = wait_reservation(waits.reserve(queue, ready).unwrap());
        let stale = abandoned.token.unwrap();
        assert!(matches!(waits.reserve(queue, ready), Err(ForegroundWaitError::Pending(token)) if token == stale));
        assert!(matches!(waits.reserve(queue, wait_ready(fence, 0, None)), Err(ForegroundWaitError::Pending(_))));
        drop(abandoned);

        let accepted = wait_reservation(waits.reserve(queue, wait_ready(fence, 2, Some(6))).unwrap()).arm().unwrap();
        let accepted_token = accepted.token.unwrap();
        accepted.rollback().unwrap();
        let newer = wait_reservation(waits.reserve(queue, wait_ready(fence, 3, Some(7))).unwrap());
        assert_eq!(waits.0.lock().unwrap().transition(accepted_token, ForegroundWaitState::Accepted, None),
            Err(ForegroundWaitError::StaleToken));
        assert!(matches!(waits.reserve(queue, ready), Err(ForegroundWaitError::Pending(token)) if token == newer.token.unwrap()));
        drop(newer);
        waits.retire_queue(queue).unwrap();
        let replacement = waits.register_queue(0xB).unwrap();
        assert!(replacement.generation > queue.generation);
        assert!(matches!(waits.reserve(queue, ready), Err(ForegroundWaitError::Queue)));
        assert!(matches!(waits.reserve(replacement, ready), Ok(ForegroundWaitDecision::Reserve(_))));
    }

    #[test]
    fn foreground_wait_unpublished_acceptance_fails_closed() {
        let waits = ForegroundWaits::new(107, 701).unwrap();
        let queue = waits.register_queue(0xC).unwrap();
        let ready = wait_ready(waits.fence(), 1, Some(9));
        let accepted = wait_reservation(waits.reserve(queue, ready).unwrap()).arm().unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _accepted = accepted;
            panic!("after native wait acceptance");
        }));
        assert!(result.is_err());
        assert!(matches!(waits.reserve(queue, ready), Err(ForegroundWaitError::Uncertain)));
        assert!(matches!(waits.reserve(queue, wait_ready(waits.fence(), 0, None)), Err(ForegroundWaitError::Uncertain)));
    }

    #[test]
    fn foreground_wait_generation_replacement_rejects_old_guards_and_fences() {
        let waits = ForegroundWaits::new(109, 801).unwrap();
        let queue = waits.register_queue(0xD).unwrap();
        let old_fence = waits.fence();
        let old_ready = wait_ready(old_fence, 1, Some(3));
        let old = wait_reservation(waits.reserve(queue, old_ready).unwrap());
        waits.replace_generation(109, 110, 802).unwrap();
        drop(old);
        assert!(matches!(waits.reserve(queue, old_ready), Err(ForegroundWaitError::StaleGeneration)));
        assert_eq!(waits.replace_generation(110, 109, 803), Err(ForegroundWaitError::StaleGeneration));
        assert_eq!(waits.replace_generation(110, 111, 801), Err(ForegroundWaitError::Fence));
        let current = wait_ready(waits.fence(), 1, Some(4));
        assert!(matches!(waits.reserve(queue, current), Ok(ForegroundWaitDecision::Reserve(_))));
    }

    fn upload_observation(page: usize, pool: (u32, u32), source_word: u64) -> UploadPageObservation {
        let mut source = ZERO_HASHES;
        source[1] = source_word;
        source[2] = source_word;
        UploadPageObservation { page, pool, hole: false, sent: Some(ZERO_HASHES), source }
    }

    fn coordinated_upload_fixture(
        owner: u64,
    ) -> (SparseBatches, CoordinatedSparseUploadPlan, UploadPageObservation, ClaimFrontierToken) {
        let batches = SparseBatches::new(71);
        assert!(batches.add_dirty_pages([0x2_1000]));
        let lease = batches.seal(owner, 1).unwrap().unwrap();
        let token = lease.token;
        let observation = upload_observation(0x2_1000, (4, 9), 0xA5);
        let plan = CoordinatedSparseUploadPlan::build(lease, 71, 7, |_| Some(observation.clone())).unwrap();
        (batches, plan, observation, token)
    }

    fn commit_coordinated(
        plan: CoordinatedSparseUploadPlan, observation: &UploadPageObservation,
        reservations: &UploadReservations, slot: usize, fence: u64,
    ) -> Result<UploadSubmissionReceipt, CoordinatedSparseUploadError<(), (), ()>> {
        plan.commit_reserved(
            71, 7, |_| Some(observation.clone()), reservations, slot, 64,
            |_, _| Ok(vec![0; 64]), |_, _| Ok(()), |_, _| {}, || Ok(fence),
        )
    }

    #[test]
    fn shared_upload_page_builder_preserves_intent_order_and_dirty_ranges() {
        let claimed = [
            (0x1_1000, WorkReasons(WorkReasons::ZERO.0 | WorkReasons::DIRTY.0)),
            (0x1_2000, WorkReasons::RETRY),
        ];
        let first = upload_observation(0x1_1000, (2, 3), 0x91);
        let second = upload_observation(0x1_2000, (2, 3), ZERO_HASH);
        let pages = build_upload_pages(&claimed, |page| match page {
            0x1_1000 => Some(first.clone()),
            0x1_2000 => Some(second.clone()),
            _ => None,
        }).unwrap();

        assert_eq!(pages.iter().map(|page| page.page).collect::<Vec<_>>(), [0x1_1000, 0x1_2000]);
        assert!(pages[0].zero && pages[0].dirty);
        assert_eq!(pages[0].dirty_ranges, [(0x1_1000 + CHUNK, 2 * CHUNK)]);
        assert!(!pages[1].zero && pages[1].dirty);
        assert!(pages[1].dirty_ranges.is_empty(), "retry intent was lost when no chunks changed");
        assert!(build_upload_pages(&claimed, |page| {
            (page == 0x1_1000).then(|| first.clone())
        }).is_none());
    }

    #[test]
    fn coordinated_plan_revalidates_every_sparse_page_identity() {
        let (_, plan, observation, _) = coordinated_upload_fixture(1);
        assert_eq!(plan.validate(72, 7, |_| Some(observation.clone())), Err(UploadPlanInvalid::StateGeneration));
        assert_eq!(plan.validate(71, 8, |_| Some(observation.clone())), Err(UploadPlanInvalid::MappingEpoch));
        assert_eq!(plan.validate(71, 7, |_| None), Err(UploadPlanInvalid::MissingPage));

        let mut remapped = observation.clone();
        remapped.page += TILE;
        assert_eq!(plan.validate(71, 7, |_| Some(remapped.clone())), Err(UploadPlanInvalid::Mapping));
        let mut repooled = observation.clone();
        repooled.pool.1 += 1;
        assert_eq!(plan.validate(71, 7, |_| Some(repooled.clone())), Err(UploadPlanInvalid::Mapping));
        let mut hole = observation.clone();
        hole.hole = true;
        assert_eq!(plan.validate(71, 7, |_| Some(hole.clone())), Err(UploadPlanInvalid::Mapping));
        let mut resent = observation.clone();
        resent.sent.as_mut().unwrap()[3] ^= 1;
        assert_eq!(plan.validate(71, 7, |_| Some(resent.clone())), Err(UploadPlanInvalid::SentHashes));
        let mut rewritten = observation.clone();
        rewritten.source[3] ^= 1;
        assert_eq!(plan.validate(71, 7, |_| Some(rewritten.clone())), Err(UploadPlanInvalid::SourceHashes));
    }

    #[test]
    fn live_parity_snapshot_revalidates_concurrent_changes_and_retries() {
        let batches = SparseBatches::new(73);
        assert!(batches.add_dirty_pages([0x2_3000]));
        let lease = batches.seal(3, 1).unwrap().unwrap();
        let token = lease.token;
        let through = batches.through().unwrap();
        let observed = upload_observation(0x2_3000, (6, 11), 0xB7);
        let mut captures = 0;
        let snapshot = UploadPlanningSnapshot::capture(73, 9, lease.pages(), |_| {
            captures += 1;
            Some(observed.clone())
        }).unwrap();
        assert_eq!(captures, 1);

        // Construction consumes only the owned snapshot; no live State observation is available
        // while this work runs outside its lock.
        let plan = CoordinatedSparseUploadPlan::from_snapshot(lease, snapshot).unwrap();
        assert_eq!(plan.pages[0].source, observed.source);
        assert_eq!(plan.validate(74, 9, |_| Some(observed.clone())), Err(UploadPlanInvalid::StateGeneration));
        assert_eq!(plan.validate(73, 10, |_| Some(observed.clone())), Err(UploadPlanInvalid::MappingEpoch));
        let mut rewritten = observed.clone();
        rewritten.source[5] ^= 1;
        assert_eq!(plan.validate(73, 9, |_| Some(rewritten.clone())), Err(UploadPlanInvalid::SourceHashes));
        assert_eq!(plan.validate(73, 9, |_| Some(observed.clone())), Ok(()));

        let parity = LegacyParityUpload::Planned(plan);
        assert_eq!(
            parity.revalidate(74, 9, |_| Some(observed.clone())),
            LegacyParityRevalidation::RestartForGeneration,
        );
        assert_eq!(
            parity.revalidate(73, 10, |_| Some(observed.clone())),
            LegacyParityRevalidation::RetryCurrentGeneration,
        );
        assert_eq!(
            parity.revalidate(73, 9, |_| Some(rewritten.clone())),
            LegacyParityRevalidation::RetryCurrentGeneration,
        );
        assert_eq!(
            parity.revalidate(73, 9, |_| Some(observed.clone())),
            LegacyParityRevalidation::Ready,
        );
        assert_eq!(
            batches.observe(through, 4),
            Ok(ClaimFrontierAction::TakeOver(token)),
            "validation must not accept or publish the behavior-neutral parity lease",
        );
        parity.retry().unwrap();
        let restored = batches.seal(4, 1).unwrap().expect(
            "a stale unlocked plan did not restore its exact work for retry",
        );
        assert_eq!(restored.pages(), &[(0x2_3000, WorkReasons::DIRTY)]);
        let fallback = LegacyParityUpload::Lease {
            lease: restored,
            state_generation: 73,
            mapping_epoch: 9,
        };
        assert_eq!(
            fallback.revalidate(74, 9, |_| None),
            LegacyParityRevalidation::RestartForGeneration,
        );
        assert_eq!(
            fallback.revalidate(73, 10, |_| None),
            LegacyParityRevalidation::RetryCurrentGeneration,
        );
        fallback.retry().unwrap();
        let exact = batches.seal(5, 1).unwrap().unwrap();
        assert_eq!(exact.pages(), &[(0x2_3000, WorkReasons::DIRTY)]);
        exact.obsolete().unwrap();
    }

    #[test]
    fn coordinated_preaccept_unwind_cancels_reservation_and_leaves_batch_visible() {
        for panic_at in ["copy", "prepare"] {
            let (batches, plan, observation, token) = coordinated_upload_fixture(1);
            let reservations = UploadReservations::new(71, 1, 256, 64);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = plan.commit_reserved(
                    71, 7, |_| Some(observation.clone()), &reservations, 0, 64,
                    |_, _| -> Result<Vec<u8>, ()> {
                        if panic_at == "copy" { panic!("copy panicked") }
                        Ok(vec![0; 64])
                    },
                    |_, _| -> Result<(), ()> {
                        if panic_at == "prepare" { panic!("prepare panicked") }
                        Ok(())
                    },
                    |_, _| {}, || Ok::<_, ()>(9),
                );
            }));
            assert!(result.is_err());
            assert_eq!(batches.observe(1, 9), Ok(ClaimFrontierAction::TakeOver(token)));
            let ledger = reservations.0.lock().unwrap();
            assert!(!ledger.poisoned && ledger.records.is_empty());
            assert!(ledger.slots.iter().all(Option::is_none));
        }
    }

    #[test]
    fn coordinated_commit_orders_callbacks_and_owns_copied_staging() {
        let (_, plan, observation, _) = coordinated_upload_fixture(1);
        let reservations = UploadReservations::new(71, 1, 256, 64);
        let calls = std::cell::RefCell::new(Vec::new());
        let receipt = plan.commit_reserved(
            71, 7, |_| Some(observation.clone()), &reservations, 0, 4,
            |range, pages| {
                calls.borrow_mut().push("copy");
                assert_eq!(range, UploadRingRange { start: 0, end: 4 });
                assert_eq!(pages[0].page, observation.page);
                Ok::<_, ()>(vec![1, 2, 3, 4])
            },
            |staging, _| {
                calls.borrow_mut().push("prepare");
                assert_eq!(&*staging.bytes, &[1, 2, 3, 4]);
                Ok::<_, ()>(())
            },
            |staging, _| {
                calls.borrow_mut().push("execute");
                assert_eq!(&*staging.bytes, &[1, 2, 3, 4]);
            },
            || { calls.borrow_mut().push("signal"); Ok::<_, ()>(9) },
        ).unwrap();
        assert_eq!(&*calls.borrow(), &["copy", "prepare", "execute", "signal"]);
        assert_eq!(reservations.complete(receipt), Ok(4));
    }

    #[test]
    fn coordinated_plan_preaccept_failures_stay_visible_and_takeover_revokes() {
        for failure in ["validate", "reservation generation", "reserve", "copy", "size", "prepare"] {
            let (batches, plan, observation, token) = coordinated_upload_fixture(1);
            let reservations = UploadReservations::new(if failure == "reservation generation" { 72 } else { 71 }, 1, 256, 64);
            let occupied = (failure == "reserve").then(|| reservations.reserve(0, 64).unwrap());
            let result = plan.commit_reserved(
                if failure == "validate" { 72 } else { 71 }, 7, |_| Some(observation.clone()),
                &reservations, 0, 64,
                |_, _| if failure == "copy" { Err("copy") } else { Ok(vec![0; if failure == "size" { 63 } else { 64 }]) },
                |_, _| if failure == "prepare" { Err("prepare") } else { Ok(()) },
                |_, _| {}, || Ok::<_, &'static str>(9),
            );
            let expected = match failure {
                "validate" => matches!(result, Err(CoordinatedSparseUploadError::Invalid(UploadPlanInvalid::StateGeneration))),
                "reservation generation" => matches!(result, Err(CoordinatedSparseUploadError::Reserve(UploadReservationError::StaleGeneration))),
                "reserve" => matches!(result, Err(CoordinatedSparseUploadError::Reserve(UploadReservationError::SlotBusy))),
                "copy" => matches!(result, Err(CoordinatedSparseUploadError::Copy("copy"))),
                "size" => matches!(result, Err(CoordinatedSparseUploadError::StagingSize)),
                _ => matches!(result, Err(CoordinatedSparseUploadError::Prepare("prepare"))),
            };
            assert!(expected, "wrong {failure} boundary result");
            assert_eq!(batches.observe(1, 9), Ok(ClaimFrontierAction::TakeOver(token)));
            assert!(!reservations.0.lock().unwrap().poisoned);
            drop(occupied);
        }

        let (batches, stale, observation, token) = coordinated_upload_fixture(1);
        let replacement = batches.take_over(token, 2).unwrap();
        let result = commit_coordinated(stale, &observation, &UploadReservations::new(71, 1, 256, 64), 0, 9);
        assert!(matches!(result, Err(CoordinatedSparseUploadError::Accept(
            SparseBatchError::Frontier(ClaimFrontierError::StaleToken)
        ))));
        assert_eq!(replacement.pages()[0].0, 0x2_1000);
    }

    #[test]
    fn coordinated_submit_then_panic_poisons_both_generations() {
        let (batches, plan, observation, _) = coordinated_upload_fixture(1);
        let reservations = UploadReservations::new(71, 1, 256, 64);
        let submitted = std::cell::Cell::new(false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = plan.commit_reserved(
                71, 7, |_| Some(observation.clone()), &reservations, 0, 64,
                |_, _| Ok::<_, ()>(vec![0; 64]), |_, _| Ok::<_, ()>(()),
                |_, _| { submitted.set(true); panic!("execute panicked after submit") },
                || Ok::<_, ()>(9),
            );
        }));
        assert!(result.is_err() && submitted.get());
        assert!(reservations.0.lock().unwrap().poisoned);
        assert_eq!(batches.through(), Err(SparseBatchError::Poisoned));

        let (batches, plan, observation, _) = coordinated_upload_fixture(1);
        let reservations = UploadReservations::new(71, 1, 256, 64);
        let result = plan.commit_reserved(
            71, 7, |_| Some(observation.clone()), &reservations, 0, 64,
            |_, _| Ok::<_, ()>(vec![0; 64]), |_, _| Ok::<_, ()>(()), |_, _| {}, || Err::<u64, _>("signal"),
        );
        assert!(matches!(result, Err(CoordinatedSparseUploadError::Signal("signal"))));
        assert!(reservations.0.lock().unwrap().poisoned);
        assert_eq!(batches.through(), Err(SparseBatchError::Poisoned));

        let (batches, plan, observation, _) = coordinated_upload_fixture(1);
        let reservations = UploadReservations::new(71, 1, 256, 64);
        assert!(matches!(commit_coordinated(plan, &observation, &reservations, 0, 0),
            Err(CoordinatedSparseUploadError::PublishReservation(UploadReservationError::State))));
        assert!(reservations.0.lock().unwrap().poisoned);
        assert_eq!(batches.through(), Err(SparseBatchError::Poisoned));
    }

    #[test]
    fn coordinated_frontier_publication_failure_quarantines_submitted_reservation() {
        let (batches, plan, _, _) = coordinated_upload_fixture(1);
        let lease = plan.lease.unwrap();
        let batch = lease.accept().unwrap();
        let reservations = UploadReservations::new(71, 1, 256, 64);
        let reservation = reservations.reserve(0, 64).unwrap();
        let mut accepted = AcceptedCoordinatedUpload::new(reservation, batch);
        let receipt = accepted.reservation.take().unwrap().publish_after_accept(13).unwrap();
        accepted.batch_ledger.lock().unwrap().bindings.remove(&accepted.batch_token.epoch);
        assert_eq!(accepted.batch.take().unwrap().receipt(13), Err(SparseBatchError::Corrupt));
        drop(accepted);
        let ledger = reservations.0.lock().unwrap();
        assert!(ledger.poisoned);
        assert_eq!(ledger.records[&receipt.token.serial].state, UploadReservationState::Poisoned);
        drop(ledger);
        let ledger = batches.ledger.lock().unwrap();
        assert!(ledger.poisoned && ledger.work.claims.is_empty() && ledger.work.owners.is_empty());
        assert!(ledger.bindings.is_empty());
    }

    #[test]
    fn a_prior_poison_does_not_leave_later_accepted_work_owned() {
        let batches = SparseBatches::new(71);
        assert!(batches.add_dirty_pages([0x3_1000, 0x3_2000]));
        let first = batches.seal(1, 1).unwrap().unwrap().accept().unwrap();
        let second = batches.seal(2, 1).unwrap().unwrap().accept().unwrap();
        let reservations = UploadReservations::new(71, 2, 256, 64);
        let first = AcceptedCoordinatedUpload::new(reservations.reserve(0, 64).unwrap(), first);
        let second = AcceptedCoordinatedUpload::new(reservations.reserve(1, 64).unwrap(), second);
        drop(first);
        assert!(matches!(second.publish(17), Err(CoordinatedPublishError::Batch(SparseBatchError::Poisoned))));
        let ledger = batches.ledger.lock().unwrap();
        assert!(ledger.poisoned && ledger.work.claims.is_empty() && ledger.work.owners.is_empty());
        assert!(ledger.bindings.is_empty());
        drop(ledger);
        let reservations = reservations.0.lock().unwrap();
        assert!(reservations.poisoned && reservations.records.values().all(|r| r.state == UploadReservationState::Poisoned));
    }

    #[test]
    fn coordinated_receipts_publish_exact_out_of_order_fence_prefix() {
        let batches = SparseBatches::new(71);
        assert!(batches.add_dirty_pages([0x2_1000, 0x2_2000]));
        let observation = |page| upload_observation(page, (4, 9), 0xA5);
        let first_lease = batches.seal(1, 1).unwrap().unwrap();
        let first_token = first_lease.token;
        let second_lease = batches.seal(2, 1).unwrap().unwrap();
        let through = batches.through().unwrap();
        let first = CoordinatedSparseUploadPlan::build(first_lease, 71, 7, |page| Some(observation(page))).unwrap();
        let second = CoordinatedSparseUploadPlan::build(second_lease, 71, 7, |page| Some(observation(page))).unwrap();
        let reservations = UploadReservations::new(71, 2, 256, 64);
        let second_receipt = commit_coordinated(second, &observation(0x2_2000), &reservations, 1, 11).unwrap();
        assert_eq!(batches.observe(through, 3), Ok(ClaimFrontierAction::TakeOver(first_token)));
        let first_receipt = commit_coordinated(first, &observation(0x2_1000), &reservations, 0, 8).unwrap();
        assert_eq!(batches.observe(through, 3), Ok(ClaimFrontierAction::Ready { through, required_fence: Some(11) }));
        assert_eq!(reservations.complete(first_receipt), Ok(0));
        assert_eq!(reservations.complete(second_receipt), Ok(128));
    }

    #[test]
    fn upload_reservations_reclaim_only_a_contiguous_retired_prefix() {
        let reservations = UploadReservations::new(41, 2, 256, 64);
        let first = reservations.reserve(0, 80).unwrap();
        let second = reservations.reserve(1, 80).unwrap();
        assert_eq!(first.range(), UploadRingRange { start: 0, end: 80 });
        assert_eq!(second.range(), UploadRingRange { start: 128, end: 208 });
        let first = first.submit(5).unwrap();
        let second = second.submit(6).unwrap();

        assert_eq!(reservations.complete(second), Ok(0), "later retirement crossed an older owner");
        assert!(matches!(reservations.reserve(1, 64), Err(UploadReservationError::Full)));
        assert_eq!(reservations.complete(first), Ok(208));

        let wrapped = reservations.reserve(1, 64).unwrap();
        assert_eq!(wrapped.range(), UploadRingRange { start: 256, end: 320 });
        assert_eq!(wrapped.submit(7).and_then(|receipt| reservations.complete(receipt)), Ok(320));
    }

    #[test]
    fn an_empty_ring_discards_tail_padding_before_a_full_capacity_reservation() {
        let reservations = UploadReservations::new(42, 1, 256, 64);
        drop(reservations.reserve(0, 64).unwrap());
        {
            let ledger = reservations.0.lock().unwrap();
            assert_eq!((ledger.cursor, ledger.reclaimed), (64, 64));
        }

        let full = reservations.reserve(0, 256).unwrap();
        assert_eq!(full.range(), UploadRingRange { start: 256, end: 512 });
        let ledger = reservations.0.lock().unwrap();
        assert_eq!((ledger.cursor, ledger.reclaimed), (512, 256));
    }

    #[test]
    fn failed_empty_ring_reservation_does_not_publish_boundary_normalization() {
        let mut ledger = UploadReservationLedger::new(42, 1, 256, 64);
        let token = ledger.reserve(42, 0, 64).unwrap().0;
        assert_eq!(ledger.cancel(token), Ok(64));
        ledger.next_serial = u64::MAX;

        assert!(matches!(ledger.reserve(42, 0, 256), Err(UploadReservationError::Exhausted)));
        assert_eq!((ledger.cursor, ledger.reclaimed), (64, 64));
        assert_eq!(ledger.next_serial, u64::MAX);
        assert!(ledger.order.is_empty());
        assert!(ledger.records.is_empty());
        assert!(ledger.slots[0].is_none());
    }

    #[test]
    fn cancelling_an_older_reservation_never_rewinds_newer_ring_work() {
        let reservations = UploadReservations::new(43, 3, 512, 64);
        let first = reservations.reserve(0, 64).unwrap();
        let middle = reservations.reserve(1, 64).unwrap();
        let third = reservations.reserve(2, 64).unwrap();
        let middle_token = middle.token.unwrap();
        drop(middle);
        assert_eq!(reservations.0.lock().unwrap().reclaimed, 0);

        let later = reservations.reserve(1, 64).unwrap();
        assert!(later.range().start >= third.range().end, "a cancellation rewound across newer work");
        drop(first);
        assert_eq!(reservations.0.lock().unwrap().reclaimed, 128);
        assert_eq!(reservations.0.lock().unwrap().cancel(middle_token), Err(UploadReservationError::Missing));

        let third = third.submit(8).unwrap();
        assert_eq!(reservations.complete(UploadSubmissionReceipt { fence: 9, ..third }), Err(UploadReservationError::Fence));
        assert_eq!(reservations.complete(third), Ok(192));
        drop(later);
        assert_eq!(reservations.0.lock().unwrap().reclaimed, 256);
    }

    #[test]
    fn dropped_reservation_cancels_exactly_it_and_signal_failure_poisons_reuse() {
        let reservations = UploadReservations::new(47, 2, 256, 64);
        let abandoned = reservations.reserve(0, 64).unwrap();
        let stale = abandoned.token.unwrap();
        drop(abandoned);
        assert_eq!(reservations.0.lock().unwrap().reclaimed, 64);
        assert_eq!(reservations.0.lock().unwrap().cancel(stale), Err(UploadReservationError::Missing));

        let poisoned = reservations.reserve(0, 64).unwrap();
        poisoned.poison().unwrap();
        let ledger = reservations.0.lock().unwrap();
        assert!(ledger.poisoned);
        assert_eq!(ledger.slots[0], Some(2), "poison released an allocator with untracked GPU work");
        drop(ledger);
        assert!(matches!(reservations.reserve(1, 64), Err(UploadReservationError::Poisoned)));
    }

    #[test]
    fn upload_reservation_bounds_generation_and_monotonic_overflow_fail_closed() {
        let mut ledger = UploadReservationLedger::new(53, 1, 256, 64);
        assert!(matches!(ledger.reserve(52, 0, 64), Err(UploadReservationError::StaleGeneration)));
        assert!(matches!(ledger.reserve(53, 1, 0), Err(UploadReservationError::Slot)));
        assert!(matches!(ledger.reserve(53, 0, 0), Err(UploadReservationError::Empty)));
        assert!(matches!(ledger.reserve(53, 0, 257), Err(UploadReservationError::TooLarge)));

        ledger.cursor = u64::MAX - 31;
        assert!(matches!(ledger.reserve(53, 0, 64), Err(UploadReservationError::Exhausted)));
        ledger.cursor = 0;
        ledger.next_serial = u64::MAX;
        assert!(matches!(ledger.reserve(53, 0, 64), Err(UploadReservationError::Exhausted)));
        assert!(ledger.records.is_empty());
        assert!(ledger.order.is_empty());
        assert!(ledger.slots[0].is_none());
    }

    #[test]
    fn a_failed_title_queue_wait_rolls_back_only_its_reservation() {
        let mut waited = FxHashMap::from_iter([(1, 4), (2, 7)]);
        assert!(reserve_queue_wait(&mut waited, 1, 5));
        rollback_queue_wait(&mut waited, 1, 5, Some(4));
        assert_eq!(waited, FxHashMap::from_iter([(1, 4), (2, 7)]));
        rollback_queue_wait(&mut waited, 1, 9, None);
        assert_eq!(waited.get(&1), Some(&4), "a stale failure changed a newer reservation");
    }

    #[test]
    fn quarantine_uses_installed_retirement_marks_and_fails_closed_without_one() {
        use std::sync::Arc;
        let own = Arc::new(1u32);
        let retired = Arc::new(2u32);
        let marks = initial_quarantine_marks((own.clone(), 3), &[(retired.clone(), 4)], false).unwrap();
        assert_eq!(marks.iter().map(|(_, value)| *value).collect::<Vec<_>>(), [3, 4]);
        drop(retired);
        assert_eq!(Arc::strong_count(&marks[1].0), 1, "batch did not own the retirement marker");
        assert!(initial_quarantine_marks((own, 3), &[(marks[1].0.clone(), 4)], true).is_none(), "Preparing/Failed reused tiles without a completion source");
    }

    /// A block freed on one thread while another allocates: its address goes back to the
    /// windows only once it is decommitted (released first, the decommit came after another
    /// thread had the same address again: that thread's writes faulted).
    #[test]
    fn a_freed_block_is_decommitted_before_its_address_is_reused() {
        if crate::gpu::uma::get().is_none_or(|u| !u.has_sparse()) {
            return;
        }
        let threads: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| unsafe {
                    for i in 0..2000u64 {
                        let Some((p, n)) = alloc(TILE, TILE) else { continue };
                        std::ptr::write_volatile(p as *mut u64, i);
                        std::ptr::write_volatile((p + n - 8) as *mut u64, i);
                        assert_eq!(std::ptr::read_volatile(p as *const u64), i);
                        assert!(free(p));
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
    }

    /// `f` on another thread while this one holds the sparse state (as `before_execute` does
    /// around its D3D12 calls, which can wait for that thread): did `f` finish without waiting
    /// for the state? The thread runs before the state is taken, so its start is not timed.
    fn returns_while_the_state_is_held(f: impl FnOnce() + Send + 'static) -> bool {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let (tx, rx) = std::sync::mpsc::channel();
        let t = std::thread::spawn(move || {
            let _ = ready_tx.send(());
            let _ = go_rx.recv();
            f();
            let _ = tx.send(());
        });
        ready_rx.recv().unwrap();
        let held = STATE.lock().unwrap();
        go_tx.send(()).unwrap();
        let done = rx.recv_timeout(std::time::Duration::from_secs(5));
        drop(held);
        let joined = t.join();
        assert!(joined.is_ok(), "the thread panicked: {joined:?}");
        done.is_ok()
    }

    /// A placed buffer released while another thread holds the sparse state: the release does
    /// not wait for the state, and the next `lock_state` lets the buffer go.
    #[test]
    fn a_buffer_released_while_the_state_is_held_does_not_wait_for_it() {
        const RES: usize = 0x7ff0_dead_0000;
        assert!(returns_while_the_state_is_held(|| forget_buffer(RES)), "forget_buffer waited for the sparse state's lock");
        drop(lock_state());
        assert!(!FORGOTTEN.lock().unwrap().contains(&RES));
    }


    /// Freeing the last tile in use of a pool and taking a tile again takes the retired pool
    /// back, not a new heap each time (#913: 288 pools of 64 MB, 18 GB asked of a 12 GB card).
    #[test]
    fn a_pool_retired_a_moment_ago_is_taken_back() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");
            let readbacks = Arc::new(SharedReadbacks::new(rb_fence.clone(), None));
            let rb_submit = Arc::new(Mutex::new(RbSubmit::new(dev.clone())));
            let mut s = State {
                generation: 1,
                mapping_epoch: 0,
                coordination: SparseCoordination::new(1, 1).unwrap(),
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                rbq: None,
                rbq_value: 0,
                readbacks,
                rb_submit,
                windows: Vec::new(),
                pools: Vec::new(),
                free: FreePools::new(),
                cooling: VecDeque::new(),
                retired: Vec::new(),
                retire_tried: std::time::Instant::now(),
                parked: FxHashMap::default(),
                kept: FxHashMap::default(),
                pending: Vec::new(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                quarantined: VecDeque::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: BTreeSet::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                uploads_ordered: true,
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: BTreeMap::new(),
                scratch_runs: Vec::new(),
                own_pages: Vec::new(),
                mirrors: FxHashMap::default(),
                mirror_src: FxHashMap::default(),
                src_runs: BTreeMap::new(),
                sources: &SOURCES,
                fill: Vec::new(),
                asks: Vec::new(),
                remaps: Vec::new(),
                later: VecDeque::new(),
                fill_pages: fill_pages(),
                mirrored: 0,
                mirror_cap: mirror_cap(),
                refused: FxHashSet::default(),
                no_memory: None,
                starved: Vec::new(),
                counts: MirrorCounts::default(),
                diag: ViewDiag::default(),
                poison: None,
                retry_pages: Vec::new(),
            };

            let (first, reused) = s.take_tile().expect("a tile from a new pool");
            assert!(!reused, "a new pool's tile needs no clearing");
            s.free.push(first);
            s.retire_free_pools();
            assert_eq!(s.retired.len(), 1, "an all-free pool is retired");
            for round in 0..8 {
                let (tile, reused) = s.take_tile().expect("a tile");
                assert!(reused, "round {round}: a tile of a retired pool is cleared before use");
                s.free.push(tile);
                s.retire_free_pools();
                assert_eq!(s.retired.len(), 1, "round {round}: the retired pool was taken back, not a new one made");
                assert_eq!(s.pools.iter().filter(|p| p.is_some()).count(), 0, "round {round}");
            }
        }
    }

    /// A title frees its tiles in runs spread over all its pools: the pools of the freed runs go
    /// back. With pools of 64 MiB none of these did (every pool kept the tiles still in use).
    #[test]
    fn pools_of_freed_runs_go_back() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(1);
            // 192 MiB of tiles, taken in order as a title's first commits take them
            const TILES: usize = 3072;
            let tiles: Vec<(u32, u32)> = (0..TILES).map(|_| s.take_tile().expect("a tile").0).collect();
            let made = s.pools.iter().filter(|p| p.is_some()).count();
            assert_eq!(made * POOL_TILES as usize, TILES, "the tiles fill whole pools");
            // of every 64 MiB the title keeps the first 4 MiB and frees the rest
            let (mut kept, mut freed) = (0, 0);
            for (i, &t) in tiles.iter().enumerate() {
                if i % 1024 < 64 {
                    kept += 1;
                } else {
                    s.free.push(t);
                    freed += 1;
                }
            }
            s.retire_free_pools();
            let left = s.pools.iter().filter(|p| p.is_some()).count() * POOL_TILES as usize;
            assert_eq!(s.free.len() + s.retired.len() * POOL_TILES as usize, freed, "every freed tile is free or its pool retired");
            assert!(left <= kept * 2, "{left} tiles of video memory stay in pools for {kept} tiles in use ({} MiB for {} MiB)", left * TILE >> 20, kept * TILE >> 20);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn aligned_tile_addresses_spread_across_hash_buckets() {
        use std::hash::BuildHasher;

        let hasher = BuildHasherDefault::<FxHasher>::default();
        let mut used = [false; 256];
        for tile in 0..used.len() {
            let address = (1usize << 32) + tile * TILE;
            let bucket = hasher.hash_one(address) as usize & (used.len() - 1);
            used[bucket] = true;
        }
        let occupied = used.iter().filter(|&&bucket| bucket).count();
        assert!(occupied >= used.len() / 2, "256 aligned addresses used {occupied} buckets");
    }

    /// Only the 64-byte chunks the CPU changed are sent again – the rest of the page may hold
    /// what the GPU wrote.
    #[test]
    fn only_changed_chunks_are_sent() {
        let mut page = vec![0u8; PAGE];
        let mut sent = ZERO_HASHES;
        assert!(changed_chunks(&page, &mut sent).is_empty(), "a zero page is what the GPU has");
        page[10] = 1;
        page[200] = 2;
        page[255] = 3;
        page[4000] = 4;
        assert_eq!(changed_chunks(&page, &mut sent), vec![(0, 64), (192, 64), (3968, 64)]);
        assert!(changed_chunks(&page, &mut sent).is_empty(), "nothing changed since");
        page[256] = 5; // next to the chunk at 192: one run
        page[200] = 9;
        assert_eq!(changed_chunks(&page, &mut sent), vec![(192, 128)]);
    }

    #[test]
    fn clean_chunks_take_gpu_copy_back_bytes() {
        let mut page = vec![0x11u8; PAGE];
        let mut sent = Box::new(ZERO_HASHES);
        changed_chunks(&page, &mut sent);
        let mut slot = Some(sent);
        let gpu = vec![0x22u8; 2 * CHUNK];

        land_page(&mut slot, &mut page, CHUNK, &gpu, false);

        assert!(page[..CHUNK].iter().all(|&byte| byte == 0x11));
        assert!(page[CHUNK..3 * CHUNK].iter().all(|&byte| byte == 0x22));
        assert!(changed_chunks(&page, slot.as_deref_mut().unwrap()).is_empty(), "landed bytes were left dirty");
    }

    #[test]
    fn dirty_chunks_survive_gpu_copy_back_and_upload_next() {
        let mut page = vec![0x11u8; PAGE];
        let mut sent = Box::new(ZERO_HASHES);
        changed_chunks(&page, &mut sent);
        let old_dirty_hash = sent[1];
        let mut slot = Some(sent);
        page[CHUNK..2 * CHUNK].fill(0x33);
        let gpu = vec![0x22u8; 2 * CHUNK];

        land_page(&mut slot, &mut page, CHUNK, &gpu, false);

        assert!(page[CHUNK..2 * CHUNK].iter().all(|&byte| byte == 0x33), "copy-back overwrote newer CPU bytes");
        assert_eq!(slot.as_ref().unwrap()[1], old_dirty_hash, "dirty chunk lost its last-sent hash");
        assert!(page[2 * CHUNK..3 * CHUNK].iter().all(|&byte| byte == 0x22), "clean neighbour did not land");
        assert_eq!(changed_chunks(&page, slot.as_deref_mut().unwrap()), vec![(CHUNK, CHUNK)]);
    }

    /// A wait's cells and the write after it share a chunk (#3093, the test title's Wait32 / Wait64
    /// at +0x300 / +0x308 and Write32 at +0x310): the CPU wrote the cells, so the chunk changed,
    /// and the GPU's own write still lands; the chunk keeps its old hash and uploads next.
    #[test]
    fn gpu_written_bytes_land_in_a_chunk_the_cpu_changed() {
        let mut page = vec![0u8; PAGE];
        let mut sent = Box::new(ZERO_HASHES);
        changed_chunks(&page, &mut sent);
        let old_hash = sent[0x300 / CHUNK];
        let mut slot = Some(sent);
        page[0x300..0x304].copy_from_slice(&0xDEAD_00AAu32.to_le_bytes());
        page[0x308..0x310].copy_from_slice(&0x0000_0003_0000_0000u64.to_le_bytes());

        land_page(&mut slot, &mut page, 0x310, &0x55AA_55AAu32.to_le_bytes(), true);

        assert_eq!(page[0x310..0x314], 0x55AA_55AAu32.to_le_bytes(), "the GPU's write was dropped");
        assert_eq!(page[0x300..0x304], 0xDEAD_00AAu32.to_le_bytes(), "the CPU's cells were overwritten");
        assert_eq!(page[0x308..0x310], 0x0000_0003_0000_0000u64.to_le_bytes());
        assert_eq!(slot.as_ref().unwrap()[0x300 / CHUNK], old_hash, "the changed chunk lost its last-sent hash");
        assert_eq!(changed_chunks(&page, slot.as_deref_mut().unwrap()), vec![(0x300, CHUNK)]);
    }

    /// `scan_page` leaves published hashes unchanged, skips allocation for zero pages, and
    /// coalesces adjacent dirty chunks.
    #[test]
    fn scan_page_plans_without_publishing_and_coalesces() {
        let mut page = vec![0u8; PAGE];
        let slot: Option<Box<[u64; CHUNKS]>> = None;
        let mut out = Vec::new();
        scan_page(&page, 0x1_0000, &slot, &mut out);
        assert!(slot.is_none(), "all-zero page must not allocate hash box");
        assert!(out.is_empty());

        page[PAGE - 64] = 0xAA;
        scan_page(&page, 0x1_0000, &slot, &mut out);
        assert!(slot.is_none(), "planning must not publish a last-sent hash");
        assert_eq!(out, vec![(0x1_0FC0, 64)]);

        // Adjacent chunk at the start of the next page coalesces into a single run.
        let mut next = vec![0u8; PAGE];
        next[0] = 0xBB;
        let next_slot: Option<Box<[u64; CHUNKS]>> = None;
        scan_page(&next, 0x1_1000, &next_slot, &mut out);
        assert_eq!(out, vec![(0x1_0FC0, 128)]);
    }

    #[test]
    fn hash_plan_is_published_only_when_applied() {
        let base = 0x2_0000;
        let mut mapped = FxHashMap::default();
        mapped.insert(base, TileEntry::with_holes((0, 0), 0));
        let mut prefix = Vec::new();
        append_hash_updates(base + CHUNK, &[1; CHUNK], &mut prefix);
        let mut suffix = Vec::new();
        append_hash_updates(base + PAGE + 2 * CHUNK, &[2; CHUNK], &mut suffix);
        assert!(mapped[&base].sent[0].is_none());
        assert!(mapped[&base].sent[1].is_none());
        publish_hash_updates(&mut mapped, &prefix);
        assert_eq!(mapped[&base].sent[0].as_ref().unwrap()[1], chunk_hash(&[1; CHUNK]));
        assert!(mapped[&base].sent[1].is_none(), "an unacknowledged suffix was published");
        publish_hash_updates(&mut mapped, &suffix);
        assert_eq!(mapped[&base].sent[1].as_ref().unwrap()[2], chunk_hash(&[2; CHUNK]));
    }

    #[test]
    fn cpu_write_after_planning_is_dirty_after_receipt_publication() {
        let base = 0x3_0000;
        let mut mapped = FxHashMap::default();
        mapped.insert(base, TileEntry::with_holes((0, 0), 0));
        let mut page = vec![0u8; PAGE];
        page[0] = 1;
        let staged = page[..CHUNK].to_vec();
        let mut planned = Vec::new();
        append_hash_updates(base, &staged, &mut planned);
        page[0] = 2;
        publish_hash_updates(&mut mapped, &planned);

        let mut runs = Vec::new();
        scan_page(&page, base, &mapped[&base].sent[0], &mut runs);
        assert_eq!(runs, [(base, CHUNK)]);
    }

    #[test]
    fn batch_hash_plan_is_bounded_by_staging() {
        assert_eq!(MAX_BATCH_HASH_UPDATES, STAGING / CHUNK);
        assert_eq!(MAX_BATCH_HASH_UPDATES * std::mem::size_of::<HashUpdate>(), STAGING / 4);
        assert!(!batch_hash_plan_would_overflow(MAX_BATCH_HASH_UPDATES - 1, CHUNK));
        assert!(batch_hash_plan_would_overflow(MAX_BATCH_HASH_UPDATES, CHUNK));
        assert!(batch_hash_plan_would_overflow(MAX_BATCH_HASH_UPDATES / 2 + 1, STAGING / 2));
    }

    #[test]
    fn prepared_dirty_batch_keeps_exact_hash_plan_and_rollback_point() {
        let mut batch = PreparedDirtyBatch::new(3, 0x12_3400);
        let bytes = [0xA5; 2 * CHUNK];
        batch.append_hashes(0x20_0000, &bytes);
        assert_eq!(batch.slot, 3);
        assert_eq!(batch.ring_start, 0x12_3400);
        assert_eq!(batch.updates.len(), 2);
        assert_eq!(batch.updates[0], HashUpdate { at: 0x20_0000, hash: chunk_hash(&bytes[..CHUNK]) });
        assert_eq!(batch.updates[1], HashUpdate { at: 0x20_0000 + CHUNK, hash: chunk_hash(&bytes[CHUNK..]) });
        assert!(!batch.hash_plan_would_overflow(CHUNK));

        let storage = batch.updates.as_ptr();
        let capacity = batch.updates.capacity();
        for index in 0..3 {
            batch.clear_after_submit();
            batch = batch.reopen(4 + index, 0x20_0000 + index as u64 * PAGE as u64);
            batch.append_hashes(0x30_0000, &bytes);
            assert_eq!(batch.updates.as_ptr(), storage, "batch {index} replaced its hash-plan allocation");
            assert_eq!(batch.updates.capacity(), capacity);
        }
    }

    #[test]
    fn prepared_dirty_copy_splits_at_windows_and_ring_wrap() {
        let window = 4 * PAGE;
        let base = 0x10_0000;
        let first = prepare_dirty_copy(base + window - CHUNK, 2 * CHUNK, base, window, 0, 8 * PAGE);
        assert_eq!(first, PreparedDirtyCopy {
            ring_start: 0,
            staging_offset: 0,
            window_offset: window - CHUNK,
            bytes: CHUNK,
        });
        let second = prepare_dirty_copy(base + window, CHUNK, base + window, window, CHUNK as u64, 8 * PAGE);
        assert_eq!(second.window_offset, 0);
        assert_eq!(second.bytes, CHUNK);

        let cap = 8 * PAGE;
        let wrapped = prepare_dirty_copy(base, PAGE, base, window, (cap - 512) as u64, cap);
        assert_eq!(wrapped.ring_start, cap as u64);
        assert_eq!(wrapped.staging_offset, 0);
        assert_eq!(wrapped.bytes, PAGE);
    }

    /// Ring-buffer planner keeps every chunk contiguous, 64-byte aligned, <= cap/2, and splits
    /// cleanly across the end of the ring buffer without wrapping mid-copy.
    #[test]
    fn ring_planner_handles_wrap_and_large_chunks() {
        let cap = 64 << 20;
        // Normal aligned chunk
        let (a, n) = plan_ring_chunk(0, 4096, cap);
        assert_eq!((a, n), (0, 4096));
        // Unaligned cursor aligns up to 64
        let (a, n) = plan_ring_chunk(17, 256, cap);
        assert_eq!((a, n), (64, 256));
        // Oversized chunk capped at cap / 2
        let (a, n) = plan_ring_chunk(0, cap * 2, cap);
        assert_eq!((a, n), (0, cap / 2));
        // Near end of ring with >= 4 KiB left: shrinks to fill tail
        let near = (cap - 8192) as u64;
        let (a, n) = plan_ring_chunk(near, 16384, cap);
        assert_eq!((a, n), (near, 8192));
        assert_eq!((a as usize % cap) + n, cap);
        // Tiny tail (< 4 KiB): wraps cleanly to 0 of next cycle
        let tiny = (cap - 512) as u64;
        let (a, n) = plan_ring_chunk(tiny, 4096, cap);
        assert_eq!((a, n), (cap as u64, 4096));
        assert_eq!(a as usize % cap, 0);
    }

    /// `query_tile_holes` marks reserved, free, and `PAGE_NOACCESS` pages as holes while clearing
    /// bits only for committed, accessible 4 KiB pages.
    #[test]
    fn query_tile_holes_detects_reserved_noaccess_and_committed_pages() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            // SAFETY: Reserving 2 tiles (128 KiB) with VirtualAlloc returns a 64 KiB-aligned VA.
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);

            // Initially every page in both reserved tiles is a hole.
            assert_eq!(query_tile_holes(base), 0xFFFF);
            assert_eq!(query_tile_holes(base + TILE), 0xFFFF);

            // Commit pages 4..8 as PAGE_READWRITE and page 9 as PAGE_NOACCESS.
            let p4 = VirtualAlloc((base + 4 * PAGE) as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert!(!p4.is_null());
            let p9 = VirtualAlloc((base + 9 * PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_NOACCESS);
            assert!(!p9.is_null());

            // Only bits 4..8 are cleared; reserved [0..4), PAGE_NOACCESS [9], and reserved [10..16) remain holes.
            assert_eq!(query_tile_holes(base), 0xFFFF & !(0b1111u16 << 4));
            assert_eq!(query_tile_holes(base + TILE), 0xFFFF);

            // Commit pages 0..4 and confirm bits 0..4 clear too.
            let p0 = VirtualAlloc(base as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert!(!p0.is_null());
            assert_eq!(query_tile_holes(base), 0xFF00);

            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
            // Released (MEM_FREE) memory reports all holes.
            assert_eq!(query_tile_holes(base), 0xFFFF);
        }
    }

    /// Mapping tiles for a placed buffer over partially-committed/reserved memory must preserve
    /// uncommitted hole bits so `readback_ranges` does not access-violate,
    /// and later commits clear those hole bits so subsequent writes scan cleanly.
    #[test]
    fn ensure_mapped_with_query_holes_skips_uncommitted_pages_on_readback() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");

            // Reserve 2 tiles; commit only [base + 0x4000, base + 0x8000) to simulate a
            // non-64KiB-aligned placed buffer with uncommitted prefix [base, base + 0x4000) and
            // uncommitted second tile [base + TILE, base + 2*TILE).
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            assert!(!VirtualAlloc((base + 4 * PAGE) as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());

            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc((2 * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
                .expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");

            let readbacks = Arc::new(SharedReadbacks::new(rb_fence.clone(), None));
            let rb_submit = Arc::new(Mutex::new(RbSubmit::new(dev.clone())));
            let mut s = State {
                generation: 1,
                mapping_epoch: 0,
                coordination: SparseCoordination::new(1, 1).unwrap(),
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                rbq: None,
                rbq_value: 0,
                readbacks,
                rb_submit,
                windows: vec![(base, reserved)],
                pools: Vec::new(),
                free: FreePools::new(),
                cooling: VecDeque::new(),
                retired: Vec::new(),
                retire_tried: std::time::Instant::now(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                kept: FxHashMap::default(),
                pending: Vec::new(),
                quarantined: VecDeque::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: BTreeSet::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                uploads_ordered: true,
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: BTreeMap::new(),
                scratch_runs: Vec::new(),
                own_pages: Vec::new(),
                mirrors: FxHashMap::default(),
                mirror_src: FxHashMap::default(),
                src_runs: BTreeMap::new(),
                sources: &SOURCES,
                fill: Vec::new(),
                asks: Vec::new(),
                remaps: Vec::new(),
                later: VecDeque::new(),
                fill_pages: fill_pages(),
                mirrored: 0,
                mirror_cap: mirror_cap(),
                refused: FxHashSet::default(),
                no_memory: None,
                starved: Vec::new(),
                counts: MirrorCounts::default(),
                diag: ViewDiag::default(),
                poison: None,
                retry_pages: Vec::new(),
            };

            // Mirrors `place_buffer`: map both tiles probing real OS page commitment.
            s.ensure_mapped(base, 2 * TILE, |t| {
                let holes = query_tile_holes(t);
                (holes, !holes)
            });
            assert_eq!(s.mapped[&base].holes, 0xFF0F, "uncommitted prefix and suffix pages must stay holes");
            assert_eq!(s.mapped[&(base + TILE)].holes, 0xFFFF, "fully reserved tile must have all 16 holes set");

            // Reading back the full 2-tile span must not fault on uncommitted pages.
            s.readback_ranges(&[(base, 2 * TILE)], false);

            // Now incrementally commit [base, base + 4*PAGE) and clear its hole bits (as `committed` does).
            assert!(!VirtualAlloc(base as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            let entry = s.mapped.get_mut(&base).unwrap();
            for i in 0..4 {
                entry.holes &= !(1u16 << i);
            }
            assert_eq!(entry.holes, 0xFF00);

            // Writing to the newly committed page is detected by `scan_page`.
            *(base as *mut u8) = 0x5A;
            let mut dirty = Vec::new();
            let page0 = std::slice::from_raw_parts(base as *const u8, PAGE);
            scan_page(page0, base, &entry.sent[0], &mut dirty);
            assert_eq!(dirty, vec![(base, CHUNK)]);

            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// A tile decommitted and committed again before the next submit gets its pool tile back
    /// with no new tile mapping (cleared instead). One that is not is unmapped and freed by
    /// `flush_parked` (before a submit), or when the free list runs out; a placed buffer's
    /// release does not free it while it is parked.
    #[test]
    fn a_tile_decommitted_and_committed_again_keeps_its_video_memory() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc((2 * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
                .expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");

            let readbacks = Arc::new(SharedReadbacks::new(rb_fence.clone(), None));
            let rb_submit = Arc::new(Mutex::new(RbSubmit::new(dev.clone())));
            let mut s = State {
                generation: 1,
                mapping_epoch: 0,
                coordination: SparseCoordination::new(1, 1).unwrap(),
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                rbq: None,
                rbq_value: 0,
                readbacks,
                rb_submit,
                windows: vec![(base, reserved)],
                pools: Vec::new(),
                free: FreePools::new(),
                cooling: VecDeque::new(),
                retired: Vec::new(),
                retire_tried: std::time::Instant::now(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                kept: FxHashMap::default(),
                pending: Vec::new(),
                quarantined: VecDeque::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: BTreeSet::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                uploads_ordered: true,
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: BTreeMap::new(),
                scratch_runs: Vec::new(),
                own_pages: Vec::new(),
                mirrors: FxHashMap::default(),
                mirror_src: FxHashMap::default(),
                src_runs: BTreeMap::new(),
                sources: &SOURCES,
                fill: Vec::new(),
                asks: Vec::new(),
                remaps: Vec::new(),
                later: VecDeque::new(),
                fill_pages: fill_pages(),
                mirrored: 0,
                mirror_cap: mirror_cap(),
                refused: FxHashSet::default(),
                no_memory: None,
                starved: Vec::new(),
                counts: MirrorCounts::default(),
                diag: ViewDiag::default(),
                poison: None,
                retry_pages: Vec::new(),
            };

            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            assert_eq!(s.pending, vec![base], "a new tile's mapping waits for the next flush");
            assert_eq!(s.value, 0, "nothing sent yet");
            s.flush_pending();
            assert!(s.pending.is_empty());
            let v = s.value;
            assert!(v > 0, "a new tile is mapped");
            mark_protection(&mut s.mapped, base + PAGE, PAGE, false);
            assert_eq!(s.mapped[&base].not_rw, 1 << 1);

            // decommitted: parked, still mapped, nothing sent to the GPU
            s.decommit_range(base, TILE);
            assert!(!s.mapped.contains_key(&base));
            assert_eq!(s.parked.get(&base), Some(&tile));
            assert!(!s.free.contains(&tile));
            assert_eq!(s.value, v, "no tile mapping changed");
            s.release_if_unused(base, tile);
            assert!(!s.free.contains(&tile) && s.cooling.is_empty(), "a parked tile is not free");

            assert!(s.runs.is_empty() && runs_of(&s).is_empty());
            // committed again: the same tile, cleared, no new mapping
            s.to_zero.clear();
            let gen = mapped_gen();
            s.commit_range(base, TILE);
            assert_eq!(s.runs, BTreeMap::from([(base, base + TILE)]));
            assert_eq!(s.mapped[&base].pool, tile);
            assert_eq!(s.mapped[&base].holes, 0);
            assert_eq!(s.mapped[&base].not_rw, 0, "recommitted pages use their commit protection");
            assert!(s.parked.is_empty());
            assert_eq!(s.to_zero, BTreeSet::from([(base, TILE)]));
            assert_eq!(s.value, v, "no tile mapping changed");
            assert!(mapped_gen() > gen, "readers of `mapped` see it changed");
            mark_protection(&mut s.mapped, base + 2 * PAGE, PAGE, false);
            s.decommit_range(base + 2 * PAGE, PAGE);
            s.commit_range(base + 2 * PAGE, PAGE);
            assert_eq!(s.mapped[&base].not_rw, 0, "a partially recommitted page is writable again");
            s.decommit_range(base + 3 * PAGE, PAGE);
            s.commit_range_with_protection(base + 3 * PAGE, PAGE, false);
            assert_eq!(s.mapped[&base].not_rw, 1 << 3, "commit and protection publish as one record update");
            s.commit_range(base + 3 * PAGE, PAGE);

            // decommitted and flushed (as before a submit): unmapped and quarantined until the
            // unmap fence completes, then cooling, not free yet
            s.decommit_range(base, TILE);
            assert!(s.flush_parked());
            assert!(s.parked.is_empty());
            assert!(s.quarantined.iter().any(|b| b.tiles.contains(&tile)));
            assert!(!s.cooling.iter().any(|c| c.0 == tile) && !s.free.contains(&tile));
            assert!(s.value > v, "the unmapping is sent");
            assert!(!s.flush_parked(), "nothing left to flush");
            s.wait_for(s.value);
            assert!(s.drain_quarantined());
            assert!(s.cooling.iter().any(|c| c.0 == tile) && !s.free.contains(&tile));

            // no free tile left: neither the parked tile (mapped at its address until the next
            // flush) nor the one cooling is taken for the next commit, but a new pool's. Before,
            // the parked one was unmapped and taken at once.
            s.commit_range(base, TILE);
            let parked = s.mapped[&base].pool;
            assert_ne!(parked, tile, "the cooling tile is not taken");
            s.decommit_range(base, TILE);
            s.free.clear();
            s.commit_range(base + TILE, TILE);
            assert_eq!(s.parked.get(&base), Some(&parked), "still parked");
            let tile = s.mapped[&(base + TILE)].pool;
            assert_ne!(tile.0, parked.0, "a tile of a new pool");
            assert!(!s.mapped.contains_key(&base));
            assert!(s.flush_parked());
            assert!(!s.cooling.iter().any(|c| c.0 == parked), "unmapped at the flush, quarantined");
            s.wait_for(s.value);
            assert!(s.drain_quarantined());
            assert_eq!(s.cooling.back().map(|c| c.0), Some(parked), "unmapped at the flush, then cooling");

            // a placed buffer uses the tile: the window keeps it when it is decommitted and
            // flushed, memory committed there again is the buffer's, and it is free only once
            // the buffer is gone and it was decommitted
            *s.pinned.entry(tile).or_default() += 1;
            let at = base + TILE;
            assert!(s.pending.contains(&at), "its mapping is not sent yet");
            s.decommit_range(at, TILE);
            assert!(!s.flush_parked(), "nothing to unmap");
            assert_eq!(s.kept.get(&at), Some(&tile));
            assert!(!s.free.contains(&tile));
            // kept before its mapping was sent: that stays pending, and the commit that takes the
            // tile back sends it
            let v = s.value;
            s.flush_pending();
            assert_eq!((s.value, &s.pending), (v, &vec![at]), "no mapping of a kept tile");
            s.commit_range(at, TILE);
            s.flush_pending();
            assert!(s.value > v && s.pending.is_empty(), "mapped with the commit");
            s.decommit_range(at, TILE);
            assert!(!s.flush_parked(), "nothing to unmap");
            assert_eq!(s.kept.get(&at), Some(&tile));
            let v = s.value;
            s.to_zero.clear();
            s.commit_range(at, TILE);
            assert_eq!(s.mapped[&at].pool, tile, "the buffer's tile");
            assert_eq!(s.to_zero, BTreeSet::from([(at, TILE)]));
            assert_eq!(s.value, v, "no tile mapping changed");
            s.decommit_range(at, TILE);
            assert!(!s.flush_parked());
            s.pinned.remove(&tile);
            s.release_if_unused(at, tile);
            assert!(s.kept.is_empty() && !s.cooling.iter().any(|c| c.0 == tile));
            assert!(s.flush_parked());
            s.wait_for(s.value);
            assert!(s.drain_quarantined());
            assert!(s.cooling.iter().any(|c| c.0 == tile) && !s.free.contains(&tile));

            assert_eq!(s.runs, runs_of(&s));
            // committed and decommitted again before a flush: no mapping is sent for it
            s.flush_pending();
            s.flush_parked();
            let v = s.value;
            s.commit_range(base, TILE);
            assert_eq!(s.pending, vec![base]);
            s.decommit_range(base, TILE);
            s.flush_pending(); // as for an upload or readback before the title's next submit
            assert_eq!(s.value, v, "a tile gone before the flush is not mapped");
            assert_eq!(s.pending, vec![base], "parked, its mapping never sent: still pending");
            // taken back from `parked`: the flush now sends its mapping
            s.commit_range(base, TILE);
            s.flush_pending();
            assert!(s.pending.is_empty());
            assert!(s.value > v, "a tile parked before its mapping was sent is mapped");

            // neighbours are one run, a tile taken out of it leaves the other
            s.commit_range(base, 2 * TILE);
            assert_eq!(s.runs, BTreeMap::from([(base, base + 2 * TILE)]));
            s.decommit_range(base, TILE);
            assert_eq!(s.runs, BTreeMap::from([(base + TILE, base + 2 * TILE)]));
            s.decommit_range(base + TILE, PAGE);
            assert_eq!(s.runs, runs_of(&s), "a tile with pages left stays");
            s.flush_pending();

            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    struct SignalOnDrop(ID3D12Fence, u64);

    impl Drop for SignalOnDrop {
        fn drop(&mut self) {
            unsafe {
                let _ = self.0.Signal(self.1);
            }
        }
    }

    unsafe fn test_state_with_reserved(tiles: usize) -> (usize, State) {
        use windows_sys::Win32::System::Memory::MEM_RESERVE;

        let dev = test_device();
        let base = VirtualAlloc(std::ptr::null(), tiles * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
        assert!(base != 0 && base % TILE == 0);
        let mut reserved: Option<ID3D12Resource> = None;
        dev.CreateReservedResource(&buffer_desc((tiles * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
            .expect("CreateReservedResource");
        let reserved = reserved.expect("reserved buffer");
        let s = new_state(&dev, vec![(base, reserved)]).expect("state");
        (base, s)
    }

    unsafe fn add_upload_slot(s: &mut State, fence_val: u64, close: bool) {
        let alloc = s.dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("upload allocator");
        let list: ID3D12GraphicsCommandList = s.dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            .expect("upload list");
        if close {
            list.Close().expect("upload list close");
        }
        s.slots.push(UploadSlot { alloc, list, fence_val });
    }

    #[test]
    fn begin_slot_skips_a_busy_head_for_a_ready_slot() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(1);
            // (`new_state` makes `UPLOAD_SLOTS` ready slots: these tests lay out their own)
            s.slots.clear();
            add_upload_slot(&mut s, 9, true);
            add_upload_slot(&mut s, 0, true);

            assert!(matches!(s.begin_slot(), BeginSlot::Ready(1)));
            assert_eq!(s.next_slot, 0);
            s.slots[1].list.Close().expect("selected list close");

            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn begin_slot_waits_for_the_earliest_busy_slot_without_advancing() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(1);
            // (`new_state` makes `UPLOAD_SLOTS` ready slots: these tests lay out their own)
            s.slots.clear();
            add_upload_slot(&mut s, 9, true);
            add_upload_slot(&mut s, 7, true);
            add_upload_slot(&mut s, 8, true);
            let ticket = match s.begin_slot() {
                BeginSlot::Wait(ticket) => ticket,
                _ => panic!("all busy slots did not return a retirement ticket"),
            };

            assert_eq!(s.next_slot, 0);
            assert_eq!(ticket.target, 7);
            assert!(matches!(ticket.kind, WaitKind::Slot { index: 1, next_slot: 0, fence: 7 }));
            assert!(s.wait_ticket_owns_upload(&ticket));
            s.slots[1].fence_val = 10;
            assert!(!s.wait_ticket_owns_upload(&ticket), "a changed selected slot accepted its old ticket");

            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn begin_slot_reset_failure_does_not_open_or_advance_a_slot() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(1);
            // (`new_state` makes `UPLOAD_SLOTS` ready slots: these tests lay out their own)
            s.slots.clear();
            add_upload_slot(&mut s, 0, false);

            assert!(matches!(s.begin_slot(), BeginSlot::RetryRequired));
            assert_eq!(s.next_slot, 0);

            s.slots[0].list.Close().expect("failed selected list close");
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn upload_wait_releases_and_reacquires_its_ownership_gate() {
        let (acquired_tx, acquired_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (mut guard, mut timing) = UploadGateTiming::acquire(false);
        let worker = std::thread::spawn(move || {
            let _crossing_uploader = UPLOAD_GATE.lock().unwrap();
            acquired_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        let released = timing.with_released(&mut guard, || {
            let acquired = acquired_rx.recv_timeout(std::time::Duration::from_secs(1)).is_ok();
            release_tx.send(()).unwrap();
            worker.join().unwrap();
            acquired
        });
        assert!(released, "another uploader could not cross the fence wait");
        assert!(UPLOAD_GATE.try_lock().is_err(), "the process upload gate was not reacquired");
        let times = timing.finish();
        assert_eq!((times.wait_us, times.hold_us, times.released_us), (0, 0, 0), "disabled timing read the clock");
        drop(guard);
    }

    #[test]
    fn replacement_waits_for_every_upload_generation_reader() {
        let _placement = PLACEMENT.lock().unwrap();
        let lower_target = UPLOAD_STATE_LIFETIME.read().unwrap();
        let higher_target = UPLOAD_STATE_LIFETIME.read().unwrap();
        let (acquired_tx, acquired_rx) = std::sync::mpsc::channel();
        let replacement = std::thread::spawn(move || {
            let _exclusive = UPLOAD_STATE_LIFETIME.write().unwrap();
            acquired_tx.send(()).unwrap();
        });

        assert!(acquired_rx.recv_timeout(std::time::Duration::from_millis(50)).is_err());
        drop(lower_target);
        assert!(acquired_rx.recv_timeout(std::time::Duration::from_millis(50)).is_err(),
            "replacement passed the lower target while a higher upload was still in flight");
        drop(higher_target);
        acquired_rx.recv_timeout(std::time::Duration::from_secs(1)).expect("replacement stayed blocked after every upload reader retired");
        replacement.join().unwrap();
    }

    #[test]
    fn a_stalled_upload_ring_yields_without_publishing_hashes() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE};

        unsafe {
            let (base, mut s) = test_state_with_reserved(1);
            assert_eq!(VirtualAlloc(base as *mut c_void, TILE, MEM_COMMIT, PAGE_READWRITE) as usize, base);
            std::ptr::write_bytes(base as *mut u8, 0x5a, PAGE);
            s.commit_range(base, TILE);
            s.wait_for(s.value);
            let page = std::slice::from_raw_parts(base as *const u8, PAGE);
            let mut sent = Box::new(ZERO_HASHES);
            changed_chunks(page, &mut sent);
            s.mapped.get_mut(&base).unwrap().sent[0] = Some(sent);
            let before = *s.mapped[&base].sent[0].as_ref().unwrap().clone();

            s.ring_write = STAGING as u64;
            s.ring_submitted = STAGING as u64;
            s.ring_free = 0;
            s.inflight.push_back((s.value + 1, STAGING as u64));
            let ticket = match s.upload_dirty_with_timeout(&[(base + CHUNK, 2 * CHUNK)], 20) {
                UploadStep::Wait(ticket) => ticket,
                UploadStep::Done(_) => panic!("a full ring did not return its retirement ticket"),
            };
            assert_eq!(ticket.target, s.value + 1);
            assert!(s.wait_ticket_valid(&ticket));
            s.mapping_epoch += 1;
            assert!(!s.wait_ticket_valid(&ticket), "a remapped destination accepted an old upload ticket");
            assert!(s.wait_ticket_generation(&ticket), "a remap invalidated the live state generation");
            let after = s.mapped[&base].sent[0].as_ref().unwrap();
            assert_eq!(after.as_ref(), &before, "a failed upload changed the published hash state");

            let alloc = s.dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = s.dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            list.Close().unwrap();
            // (only this busy slot: `new_state` made `UPLOAD_SLOTS` ready ones)
            s.slots.clear();
            s.slots.push(UploadSlot { alloc, list, fence_val: s.value + 1 });
            s.next_slot = 0;
            let before_slot = s.next_slot;
            let ticket = match s.begin_slot() {
                BeginSlot::Wait(ticket) => ticket,
                _ => panic!("a busy slot did not return its retirement ticket"),
            };
            assert_eq!(s.next_slot, before_slot, "a busy slot advanced before its fence completed");
            assert!(matches!(ticket.kind, WaitKind::Slot { index: 0, next_slot: 0, .. }));
            assert!(s.wait_ticket_owns_upload(&ticket));
            s.slots[0].fence_val += 1;
            assert!(!s.wait_ticket_owns_upload(&ticket), "a reused slot accepted its predecessor ticket");
            assert!(s.wait_ticket_generation(&ticket), "safe slot reuse invalidated the live generation");
            s.generation += 1;
            assert!(!s.wait_ticket_owns_upload(&ticket), "a replacement state accepted an old upload ticket");
            s.uploads_ordered = false;
            assert!(matches!(
                s.upload_dirty_with_timeout(&[(base + CHUNK, CHUNK)], 20),
                UploadStep::Done(PreparationOutcome::ExecutedUntracked)
            ));

            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn placement_create_is_unlocked_drains_releases_and_rejects_a_new_generation() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE};

        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            assert_eq!(VirtualAlloc(base as *mut c_void, 2 * TILE, MEM_COMMIT, PAGE_READWRITE) as usize, base);
            s.commit_range(base, 2 * TILE);
            s.wait_for(s.value);
            let old_tile = s.mapped[&base].pool;
            let new_tile = s.mapped[&(base + TILE)].pool;

            let mut placed: Option<ID3D12Resource> = None;
            s.dev.CreateReservedResource(&buffer_desc(TILE as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut placed)
                .expect("placed resource");
            let placed = placed.unwrap();
            let ptr = placed.as_raw() as usize;
            s.buffers.insert(ptr, vec![(base, old_tile)]);
            s.pinned.insert(old_tile, 1);

            let state = Mutex::new(Some(s));
            let forgotten = Mutex::new(Vec::new());
            let desc = buffer_desc(TILE as u64);
            let returned = placed.clone().into_raw();
            let mut unlocked = false;
            let result = place_buffer_in(
                &state,
                &forgotten,
                base + TILE,
                &desc,
                0,
                |_| {
                    unlocked = state.try_lock().is_ok();
                    forget_buffer_in(&state, &forgotten, ptr);
                    forgotten.lock().unwrap().push(ptr);
                    Some((0, returned))
                },
                |_| panic!("same-generation placement aborted"),
            );
            assert_eq!(result, Some((0, 0)));
            assert!(unlocked, "resource creation held sparse STATE");
            {
                let g = lock_state_in(&state, &forgotten);
                let s = g.as_ref().unwrap();
                assert_eq!(s.buffers.get(&ptr), Some(&vec![(base + TILE, new_tile)]));
                assert!(!s.pinned.contains_key(&old_tile));
                assert_eq!(s.pinned.get(&new_tile), Some(&1));
            }

            let original_fence = state.lock().unwrap().as_ref().unwrap().fence.clone();
            let replacement = state.lock().unwrap().as_ref().unwrap().dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let returned2 = placed.clone().into_raw();
            let mut aborted = false;
            let result = place_buffer_in(
                &state,
                &forgotten,
                base + TILE,
                &desc,
                0,
                |_| {
                    state.lock().unwrap().as_mut().unwrap().fence = replacement;
                    Some((0, returned2))
                },
                |raw| {
                    aborted = true;
                    drop(ID3D12Resource::from_raw(raw));
                },
            );
            assert!(result.is_none() && aborted, "a replacement state accepted the old generation's resource");
            state.lock().unwrap().as_mut().unwrap().fence = original_fence;

            drop(ID3D12Resource::from_raw(returned));
            let s = state.lock().unwrap().take().unwrap();
            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// `XCL_SPARSE_LAZY` on (or off) for this thread while it lives, and `take_tile` finding no
    /// video memory while `no_tiles` is set.
    struct Lazy;
    impl Lazy {
        fn on(on: bool) -> Lazy {
            LAZY_COMMITS_FOR_TEST.with(|c| c.set(Some(on)));
            Lazy
        }
        fn no_tiles(&self, none: bool) {
            NO_TILES.with(|c| c.set(none));
        }
    }
    impl Drop for Lazy {
        fn drop(&mut self) {
            LAZY_COMMITS_FOR_TEST.with(|c| c.set(None));
            NO_TILES.with(|c| c.set(false));
        }
    }

    /// A state of `tiles` tiles for the lazy tests, its window's CPU pages committed and the title's
    /// queue. None: no GPU whose unmapped tiles read zeros (the lazy path is left out there).
    unsafe fn lazy_state(tiles: usize) -> Option<(usize, State, ID3D12CommandQueue)> {
        let dev = test_device();
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).ok()?;
        if software(&dev) || o.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
            return None;
        }
        let (base, s) = test_state_with_reserved(tiles);
        assert!(!VirtualAlloc(base as *const c_void, tiles * TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
        let queue: ID3D12CommandQueue = s.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).ok()?;
        Some((base, s, queue))
    }

    /// `len` bytes of the window at `va` as the GPU has them, after what was sent to it.
    unsafe fn window_bytes(s: &mut State, va: usize, len: usize) -> Vec<u8> {
        let mut d = buffer_desc(len as u64);
        d.Flags = D3D12_RESOURCE_FLAG_NONE;
        let mut back: Option<ID3D12Resource> = None;
        s.dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back).expect("readback buffer");
        let back = back.expect("readback buffer");
        let (base, window) = (s.windows[0].0, s.windows[0].1.clone());
        let alloc: ID3D12CommandAllocator = s.dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
        let list: ID3D12GraphicsCommandList = s.dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
        list.CopyBufferRegion(&back, 0, &window, (va - base) as u64, len as u64);
        list.Close().expect("Close");
        s.flush_pending();
        s.queue.ExecuteCommandLists(&[Some(list.cast().expect("list"))]);
        s.signal();
        s.wait_for(s.value);
        let mut p = std::ptr::null_mut();
        back.Map(0, None, Some(&mut p)).expect("Map");
        let out = std::slice::from_raw_parts(p as *const u8, len).to_vec();
        back.Unmap(0, None);
        out
    }

    /// Tile `t`'s CPU pages written with bytes made from `seed`; its pages, for a submit's dirty list.
    unsafe fn write_tile(t: usize, seed: u64) -> Vec<usize> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        for i in 0..TILE / 8 {
            *(t as *mut u64).add(i) = next(&mut x);
        }
        (t..t + TILE).step_by(PAGE).collect()
    }

    /// With `XCL_SPARSE_LAZY` a committed tile gets no video memory until the CPU's first bytes
    /// go up to it; they reach the GPU, and a later submit with nothing written leaves them there
    /// (a tile cleared one submit after its bytes went up would read zeros). A tile never written
    /// keeps none and reads zeros.
    #[test]
    fn a_lazy_tile_gets_memory_at_its_first_bytes_and_keeps_them() {
        let lazy = Lazy::on(true);
        unsafe {
            let Some((base, mut s, queue)) = lazy_state(2) else { return crate::gpu::recomp::runner::skip_test("no GPU with tiled resources tier 2") };
            s.commit_range(base, 2 * TILE);
            assert_eq!(s.mapped[&base].pool, LAZY, "committed: listed with no video memory");
            assert_eq!(s.mapped[&(base + TILE)].pool, LAZY);
            // a used pool tile on hand, so that its old bytes would show if it were not cleared
            s.no_memory = None;
            let dirty = write_tile(base, 7);
            s.before_execute_on(&queue, &dirty, PAGE);
            assert_ne!(s.mapped[&base].pool, LAZY, "its first bytes gave it a pool tile");
            assert_eq!(s.mapped[&(base + TILE)].pool, LAZY, "the tile never written has none");
            let cpu = std::slice::from_raw_parts(base as *const u8, TILE).to_vec();
            assert!(window_bytes(&mut s, base, TILE) == cpu, "the bytes reached the GPU");
            s.before_execute_on(&queue, &[], PAGE);
            s.before_execute_on(&queue, &[], PAGE);
            assert!(window_bytes(&mut s, base, TILE) == cpu, "and are still there two submits later");
            assert!(window_bytes(&mut s, base + TILE, TILE).iter().all(|&b| b == 0), "a tile with no memory reads zeros");
        }
        drop(lazy);
    }

    /// Bytes that found no video memory at an upload are owed and go up whole once there is some.
    #[test]
    fn bytes_that_found_no_memory_go_up_once_there_is_some() {
        let lazy = Lazy::on(true);
        unsafe {
            let Some((base, mut s, queue)) = lazy_state(1) else { return crate::gpu::recomp::runner::skip_test("no GPU with tiled resources tier 2") };
            s.commit_range(base, TILE);
            let dirty = write_tile(base, 11);
            lazy.no_tiles(true);
            s.before_execute_on(&queue, &dirty, PAGE);
            assert_eq!(s.mapped[&base].pool, LAZY, "no video memory to be had: still none");
            assert!(s.starved.contains(&base), "its bytes are owed");
            lazy.no_tiles(false);
            s.no_memory = None;
            s.before_execute_on(&queue, &[], PAGE);
            assert_ne!(s.mapped[&base].pool, LAZY, "a submit with nothing written gives it its memory");
            let cpu = std::slice::from_raw_parts(base as *const u8, TILE).to_vec();
            assert!(window_bytes(&mut s, base, TILE) == cpu, "and all its bytes");
        }
        drop(lazy);
    }

    /// A memory write the title records on a list gives the tile its memory before the list runs
    /// (`give_memory`); a placed buffer over a tile that finds none leaves it out of its mapping
    /// instead of mapping pool tile u32::MAX.
    #[test]
    fn a_recorded_write_gives_memory_and_a_buffer_never_maps_the_sentinel() {
        let lazy = Lazy::on(true);
        unsafe {
            let Some((base, mut s, _queue)) = lazy_state(2) else { return crate::gpu::recomp::runner::skip_test("no GPU with tiled resources tier 2") };
            s.commit_range(base, 2 * TILE);
            s.give_memory(&[(base + 8, 8)]);
            assert_ne!(s.mapped[&base].pool, LAZY, "the recorded write's tile got memory");
            assert_eq!(s.mapped[&(base + TILE)].pool, LAZY, "the other did not");
            lazy.no_tiles(true);
            let tiles = s.buffer_tiles(base, 2 * TILE);
            assert!(tiles.iter().all(|&(_, tile)| tile != LAZY), "no tile of a buffer is the sentinel: {tiles:?}");
            assert!(tiles.iter().any(|&(t, _)| t == base), "the tile with memory is the buffer's");
        }
        drop(lazy);
    }

    /// Decommitting a tile with no video memory frees and parks nothing; committed again it is
    /// listed again. With the switch off a commit maps at once, as before.
    #[test]
    fn a_lazy_tile_decommits_without_a_pool_tile_and_off_maps_at_commit() {
        unsafe {
            let lazy = Lazy::on(true);
            let Some((base, mut s, _queue)) = lazy_state(1) else { return crate::gpu::recomp::runner::skip_test("no GPU with tiled resources tier 2") };
            s.commit_range(base, TILE);
            let (free, parked) = (s.free.len(), s.parked.len());
            s.decommit_range(base, TILE);
            assert!(!s.mapped.contains_key(&base), "gone from the window");
            assert_eq!((s.free.len(), s.parked.len()), (free, parked), "no pool tile freed or parked");
            s.commit_range(base, TILE);
            assert_eq!(s.mapped[&base].pool, LAZY, "listed again");
            drop(lazy);
            let off = Lazy::on(false);
            let Some((base2, mut s2, _q2)) = lazy_state(1) else { return };
            s2.commit_range(base2, TILE);
            assert_ne!(s2.mapped[&base2].pool, LAZY, "off: mapped at the commit");
            drop(off);
        }
    }

    /// The tiles `flush_parked` quarantined, once the sparse queue and the title queues are past
    /// them (`drain_quarantined`), waited for up to 5 s.
    unsafe fn settle_quarantined(s: &mut State) {
        s.wait_for(s.value);
        let start = std::time::Instant::now();
        while !s.quarantined.is_empty() && start.elapsed() < std::time::Duration::from_secs(5) {
            if !s.drain_quarantined() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    /// `flush_parked` must keep unmapped pool tiles out of `free` (and `take_tile` must not reuse
    /// them) while the sparse queue's unmap fence is still in flight on the GPU.
    #[test]
    fn flush_parked_does_not_free_or_reuse_tiles_while_unmap_is_in_flight() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            s.wait_for(s.value);

            let gate: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate fence");
            let _unblock = SignalOnDrop(gate.clone(), 1);
            s.queue.Wait(&gate, 1).expect("gate sparse queue");

            s.decommit_range(base, TILE);
            s.free.clear();
            assert!(s.flush_parked());
            assert!(
                !s.free.contains(&tile) && !s.cooling.iter().any(|c| c.0 == tile),
                "unmapped tile must remain quarantined until unmap fence completes"
            );

            // While the unmap is still blocked on `gate`, committing another tile must not steal `tile`.
            s.commit_range(base + TILE, TILE);
            assert_ne!(s.mapped[&(base + TILE)].pool, tile, "in-flight unmapped tile must not be reused before GPU completion");

            gate.Signal(1).expect("release sparse queue");
            s.wait_for(s.value);
            assert!(s.drain_quarantined(), "completed unmap batch should drain after fence completion");
            assert!(
                s.cooling.iter().any(|c| c.0 == tile) && !s.free.contains(&tile),
                "unmapped tile must go on to cool (`cooling`) once unmap fence completes"
            );
            drop(_unblock);
            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// `flush_parked` must keep unmapped pool tiles quarantined until every active title queue has
    /// finished its prior work, even after `s.queue` itself finishes unmapping.
    #[test]
    fn flush_parked_quarantines_tiles_until_title_queues_finish() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            let title_q: ID3D12CommandQueue = s.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");

            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            s.wait_for(s.value);

            let title_gate: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("title gate fence");
            let _unblock = SignalOnDrop(title_gate.clone(), 1);
            title_q.Wait(&title_gate, 1).expect("gate title queue");

            s.decommit_range(base, TILE);
            s.free.clear();
            assert!(s.flush_parked_for(std::slice::from_ref(&title_q), &[], false));

            // Even after `s.queue` finishes `unmap_tiles`, `title_q` is still gated behind `title_gate`.
            s.wait_for(s.value);
            assert!(!s.drain_quarantined(), "tile must remain quarantined while title queue has work in flight");
            assert!(!s.free.contains(&tile) && !s.cooling.iter().any(|c| c.0 == tile));

            s.commit_range(base + TILE, TILE);
            assert_ne!(s.mapped[&(base + TILE)].pool, tile, "must not reuse tile still in use by title queue");

            title_gate.Signal(1).expect("release title queue");
            let start = std::time::Instant::now();
            while !s.drain_quarantined() && start.elapsed() < std::time::Duration::from_secs(5) {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            assert!(
                s.cooling.iter().any(|c| c.0 == tile) && !s.free.contains(&tile),
                "tile must go on to cool (`cooling`) after title queue completes"
            );
            drop(_unblock);
            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// Sub-tile decommits (two 32 KiB halves, sixteen 4 KiB pages, or decommitting the last
    /// remaining committed page) must unmap the tile and return unpinned pool tiles to `free`
    /// once `holes` reaches `0xFFFF`, pruning stale `to_zero` entries for unmapped tiles.
    #[test]
    fn decommit_sub_tile_reclaims_pool_tile_when_all_pages_become_holes() {
        let mut mapped: FxHashMap<usize, TileEntry> = FxHashMap::default();
        let mut pinned: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        let mut free: Vec<(u32, u32)> = Vec::new();
        let mut to_zero: BTreeSet<(usize, usize)> = BTreeSet::new();

        // 1. Two 32 KiB decommits across a full 64 KiB tile.
        let t0 = 0x10_0000usize;
        mapped.insert(t0, TileEntry::with_holes((0, 10), 0));
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t0, TILE / 2).is_empty());
        assert_eq!(mapped[&t0].holes, 0x00FF);
        assert_eq!(to_zero.len(), 8);
        assert!(free.is_empty());

        let unmapped = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t0 + TILE / 2, TILE / 2);
        assert_eq!(unmapped, vec![t0]);
        assert!(!mapped.contains_key(&t0));
        assert_eq!(free, vec![(0, 10)]);
        assert!(to_zero.is_empty(), "stale to_zero entries for the unmapped tile must be pruned");

        // 2. Sixteen individual 4 KiB decommits in reverse order.
        let t1 = 0x20_0000usize;
        free.clear();
        mapped.insert(t1, TileEntry::with_holes((0, 11), 0));
        for i in (1..PAGES_PER_TILE).rev() {
            let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t1 + i * PAGE, PAGE);
            assert!(u.is_empty(), "tile still has committed pages after step {i}");
        }
        assert!(free.is_empty());
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t1, PAGE);
        assert_eq!(u, vec![t1]);
        assert!(!mapped.contains_key(&t1));
        assert_eq!(free, vec![(0, 11)]);
        assert!(to_zero.is_empty());

        // 3. Pinned tile decommitted in two halves: unmapped from `mapped`, NOT returned to `free`.
        let t2 = 0x30_0000usize;
        free.clear();
        mapped.insert(t2, TileEntry::with_holes((0, 12), 0));
        pinned.insert((0, 12), 1);
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t2, TILE / 2).is_empty());
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t2 + TILE / 2, TILE / 2);
        assert_eq!(u, vec![t2]);
        assert!(!mapped.contains_key(&t2));
        assert!(free.is_empty(), "pinned pool tile must not return to free list");
        assert!(to_zero.is_empty());

        // 4. Single-committed-page tile decommit reclaims tile without dropping another tile's to_zero.
        let t3 = 0x40_0000usize;
        let t4 = 0x50_0000usize;
        mapped.insert(t3, TileEntry::with_holes((0, 13), 0xFFFF & !(1u16 << 2)));
        mapped.insert(t4, TileEntry::with_holes((0, 14), 0));
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t4, PAGE).is_empty());
        assert_eq!(to_zero, BTreeSet::from([(t4, PAGE)]));
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t3 + 2 * PAGE, PAGE);
        assert_eq!(u, vec![t3]);
        assert_eq!(free, vec![(0, 13)]);
        assert_eq!(to_zero, BTreeSet::from([(t4, PAGE)]), "unrelated mapped tile's to_zero must remain");

        // 5. The pieces as runs: neighbours together, pieces inside a whole tile with it.
        let t5 = 0x60_0000usize;
        to_zero.extend([(t5, TILE), (t5 + 3 * PAGE, PAGE), (t5 + TILE, TILE), (t5 + 2 * TILE + PAGE, PAGE)]);
        assert_eq!(zero_runs(to_zero), vec![(t4, PAGE), (t5, 2 * TILE), (t5 + 2 * TILE + PAGE, PAGE)]);
    }

    /// The pages in `early`, checking that its ranges are page-aligned, not empty, disjoint and
    /// merged (no two touch).
    fn early_pages(early: &BTreeMap<usize, usize>) -> std::collections::BTreeSet<usize> {
        let mut last_end = None;
        let mut pages = std::collections::BTreeSet::new();
        for (&a, &b) in early {
            assert!(a % PAGE == 0 && b % PAGE == 0 && a < b, "range {a:#x}..{b:#x}");
            assert!(last_end.is_none_or(|e| e < a), "range {a:#x}..{b:#x} touches the one before");
            last_end = Some(b);
            pages.extend((a..b).step_by(PAGE));
        }
        pages
    }

    /// xorshift64*: the same sequence on every run
    fn next(x: &mut u64) -> u64 {
        *x ^= *x >> 12;
        *x ^= *x << 25;
        *x ^= *x >> 27;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Before the device exists, commits and decommits leave the pages a plain page set would
    /// hold, with the live path's rule (a range counts every page it touches): random ranges,
    /// unaligned too, over 300 pages.
    #[test]
    fn early_commits_are_the_pages_a_page_set_holds() {
        let base = 0x4_2000_0000usize;
        for seed in 1..=8u64 {
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let (mut early, mut model) = (BTreeMap::new(), std::collections::BTreeSet::new());
            for _ in 0..4000 {
                let a = base + (next(&mut x) % (300 * PAGE as u64)) as usize;
                let n = match next(&mut x) % 4 {
                    0 => PAGE,
                    1 => 1 + (next(&mut x) % 64) as usize,
                    _ => 1 + (next(&mut x) % (24 * PAGE as u64)) as usize,
                };
                let (start, end) = page_span(a, n);
                let pages = (start..end).step_by(PAGE);
                if next(&mut x) % 3 == 0 {
                    early_remove(&mut early, start, end);
                    pages.for_each(|p| {
                        model.remove(&p);
                    });
                } else {
                    early_add(&mut early, start, end);
                    model.extend(pages);
                }
                assert_eq!(early_pages(&early), model, "seed {seed}");
            }
        }
    }

    /// The same pages as the list this replaced, for what that list handled exactly: one
    /// entry per committed page (`XMemMapPhysicalPagesScatter`, `home_again`) and page-aligned
    /// decommits of any length.
    #[test]
    fn early_commits_match_the_old_list_for_page_entries() {
        let base = 0x4_2000_0000usize;
        let mut x = 0x1234_5678_9ABC_DEF1u64;
        let (mut early, mut old): (BTreeMap<usize, usize>, Vec<(usize, usize)>) = (BTreeMap::new(), Vec::new());
        for _ in 0..20000 {
            let a = base + (next(&mut x) % 512) as usize * PAGE;
            if next(&mut x) % 2 == 0 {
                old.push((a, PAGE));
                early_add(&mut early, a, a + PAGE);
            } else {
                let n = (1 + next(&mut x) % 8) as usize * PAGE;
                old.retain(|&(y, m)| y + m <= a || y >= a + n);
                early_remove(&mut early, a, a + n);
            }
        }
        let old_pages: std::collections::BTreeSet<usize> = old.iter().map(|&(y, _)| y).collect();
        assert_eq!(early_pages(&early), old_pages);
    }

    /// A range committed in one call keeps its other pages when one page of it is decommitted
    /// (`commit_homes` commits several pages, `move_home_away` gives one back). The list this
    /// replaced dropped the whole entry, so `init` left those still-committed pages as holes.
    #[test]
    fn early_decommit_of_one_page_keeps_the_rest() {
        let a = 0x4_2000_0000usize;
        let mut early = BTreeMap::new();
        early_add(&mut early, a, a + 4 * TILE);
        early_remove(&mut early, a + TILE, a + 2 * TILE);
        assert_eq!(early.into_iter().collect::<Vec<_>>(), vec![(a, a + TILE), (a + 2 * TILE, a + 4 * TILE)]);
    }

    /// Time for n 64 KiB pages committed one at a time and decommitted one at a time in
    /// reverse, before the device exists (`cargo test --release -p xcl --lib --
    /// --ignored --exact hle::d3d12x::sparse::tests::early_commit_timing --nocapture`,
    /// `XCL_EARLY_PAGES=16384,32768`). Run it alone: a test that made the device keeps a state.
    #[test]
    #[ignore]
    fn early_commit_timing() {
        if STATE.lock().unwrap().is_some() {
            return eprintln!("early_commit_timing: a sparse state exists – run it alone");
        }
        let sizes: Vec<usize> = std::env::var("XCL_EARLY_PAGES").ok().map_or(vec![4096, 8192, 16384, 32768], |v| v.split(',').map(|s| s.trim().parse().unwrap()).collect());
        let base = 0x4_2000_0000usize;
        for n in sizes {
            let t = std::time::Instant::now();
            for i in 0..n {
                unsafe { committed(base + i * TILE, TILE) };
            }
            let commit = t.elapsed();
            let t = std::time::Instant::now();
            for i in (0..n).rev() {
                unsafe { decommitted(base + i * TILE, TILE) };
            }
            let decommit = t.elapsed();
            println!("early: {n} pages: commit {:.1} ms, decommit {:.1} ms", commit.as_secs_f64() * 1e3, decommit.as_secs_f64() * 1e3);
        }
    }

    /// A sparse window of the test's own over a GPU view, and the CPU view the view shows, paired
    /// by `kernelx::mirrors` as a retail title's are (`tiles` tiles each, a size no other test
    /// pairs: pairs are made by size). The window's address space (4 GiB, a window's) is reserved
    /// with no access: a CPU touch of it faults the test. The CPU view is reserved, and committed
    /// where the test says. The window's buffer has `tiles` tiles.
    struct Views {
        s: State,
        dev: ID3D12Device,
        buf: ID3D12Resource,
        /// the title's queue its submits go to
        title: ID3D12CommandQueue,
        w: usize,
        c: usize,
    }

    impl Views {
        /// None: the device has no reserved buffers
        unsafe fn new(tiles: usize) -> Option<Views> {
            use windows_sys::Win32::System::Memory::MEM_RESERVE;
            let dev = test_device();
            let mut buf: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc((tiles * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut buf).ok()?;
            let buf = buf?;
            let w = VirtualAlloc(std::ptr::null(), crate::gpu::uma::WINDOW, MEM_RESERVE, PAGE_NOACCESS) as usize;
            let c = VirtualAlloc(std::ptr::null(), tiles * TILE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert!(w != 0 && c != 0 && w % TILE == 0 && c % TILE == 0);
            crate::hle::kernelx::mirrors::reserved(c, tiles * TILE, 0x2040_2000, PAGE_READWRITE);
            crate::hle::kernelx::mirrors::reserved(w, tiles * TILE, 0x2040_2000, 0x801);
            assert_eq!(cpu_of(w + 0x1_2345), c + 0x1_2345, "the views are paired");
            let title: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
            let mut s = new_state(&dev, vec![(w, buf.clone())]).expect("state");
            // (what its decommits are told of is the test's own)
            s.sources = Box::leak(Box::default());
            Some(Views { s, dev, buf, title, w, c })
        }

        /// Commit tiles `k..k + n` of the CPU view.
        unsafe fn commit(&self, k: usize, n: usize) {
            assert!(!VirtualAlloc((self.c + k * TILE) as *const c_void, n * TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
        }

        /// Write tiles `k..k + n` of the CPU view with bytes made from `seed`.
        unsafe fn write(&self, k: usize, n: usize, seed: u64) {
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            for i in 0..n * TILE / 8 {
                *((self.c + k * TILE) as *mut u64).add(i) = next(&mut x);
            }
        }

        /// The bytes of tile `k` of the CPU view.
        unsafe fn cpu_tile(&self, k: usize) -> Vec<u8> {
            std::slice::from_raw_parts((self.c + k * TILE) as *const u8, TILE).to_vec()
        }

        /// Tiles `0..n` of the GPU view asked for (`map_mirror`), taken by the test's state as a
        /// submit takes them (other tests' asks stay queued).
        fn ask(&mut self, n: usize) {
            let (lo, hi) = (self.w, self.w + n * TILE);
            map_mirror(lo, hi - lo);
            MIRROR_ASKS.lock().unwrap().retain(|&t| t < lo || t >= hi);
            self.s.asks.extend((lo..hi).step_by(TILE));
        }

        /// The tiles of `0..n` queued by `remapped`, taken by the test's state as a submit takes
        /// them (other tests' stay queued).
        fn take_remaps(&mut self, n: usize) {
            let (lo, hi) = (self.w, self.w + n * TILE);
            let mut q = REMAPS.lock().unwrap();
            self.s.remaps.extend(q.iter().copied().filter(|&t| t >= lo && t < hi));
            q.retain(|&t| t < lo || t >= hi);
        }

        /// A submit of the title, with the CPU pages `dirty` written since the last.
        unsafe fn submit(&mut self, dirty: &[usize]) {
            let title = self.title.clone();
            let _ = self.s.before_execute_on(&title, dirty, PAGE);
        }

        /// `len` bytes at `off` of `res` as the GPU has them, after what was sent to it.
        unsafe fn gpu_bytes(&mut self, res: &ID3D12Resource, off: usize, len: usize) -> Vec<u8> {
            let mut d = buffer_desc(len as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            let mut back: Option<ID3D12Resource> = None;
            self.dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back).expect("readback buffer");
            let back = back.expect("readback buffer");
            let alloc: ID3D12CommandAllocator = self.dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = self.dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            list.CopyBufferRegion(&back, 0, res, off as u64, len as u64);
            list.Close().expect("Close");
            self.s.flush_pending();
            self.s.queue.ExecuteCommandLists(&[Some(list.cast().expect("list"))]);
            self.s.signal();
            self.s.wait_for(self.s.value);
            let mut p = std::ptr::null_mut();
            back.Map(0, None, Some(&mut p)).expect("Map");
            let out = std::slice::from_raw_parts(p as *const u8, len).to_vec();
            back.Unmap(0, None);
            out
        }
    }

    impl Drop for Views {
        fn drop(&mut self) {
            use windows_sys::Win32::System::Memory::MEM_RELEASE;
            unsafe {
                self.s.wait_for(self.s.value);
                crate::hle::kernelx::mirrors::released(self.c);
                crate::hle::kernelx::mirrors::released(self.w);
                VirtualFree(self.c as *mut c_void, 0, MEM_RELEASE);
                VirtualFree(self.w as *mut c_void, 0, MEM_RELEASE);
            }
        }
    }

    /// The tiles of a GPU view are read and written at the CPU view they show, never at their own
    /// addresses (reserved with no access, as the title's are: a touch faults): the uploads'
    /// sources are in one piece only as far as the CPU view is, and immediates and copies back
    /// land in the CPU view. Before, the scan and the copies read the window's own addresses.
    #[test]
    fn a_mirrored_tile_is_read_and_written_at_its_cpu_view() {
        unsafe {
            let Some(mut v) = Views::new(37) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 5);
            v.write(0, 5, 1);
            v.s.asks.extend((0..4).map(|k| w + k * TILE));
            v.s.map_mirrors();
            for k in 0..4 {
                let e = &v.s.mapped[&(w + k * TILE)];
                assert_eq!((e.src, e.holes), (c + k * TILE, 0), "tile {k} shows its CPU tile");
            }
            assert_eq!(v.s.mirrored, 4);
            assert!(v.s.runs.is_empty() && runs_of(&v.s).is_empty(), "the view's own addresses are no CPU memory");
            // the CPU bytes of a run over the tiles are in one piece …
            assert_eq!(v.s.cpu_span(w + 0x100, 3 * TILE), (c + 0x100, 3 * TILE));
            // … up to a tile shown over another part of the CPU view …
            v.s.mapped.get_mut(&(w + 2 * TILE)).unwrap().src = c + 4 * TILE;
            assert_eq!(v.s.cpu_span(w + 0x100, 3 * TILE), (c + 0x100, 2 * TILE - 0x100));
            assert_eq!(v.s.cpu_span(w + 2 * TILE + 0x40, 0x80), (c + 4 * TILE + 0x40, 0x80));
            v.s.mapped.get_mut(&(w + 2 * TILE)).unwrap().src = c + 2 * TILE;
            // … or to one of the window's own (the one after the 4th has nothing: its address is
            // its own)
            assert_eq!(v.s.cpu_span(w + 3 * TILE, 2 * TILE), (c + 3 * TILE, TILE));

            // filled whole from the CPU view at the next submit
            v.submit(&[]);
            assert!(v.s.fill.is_empty());
            assert!(v.s.mapped[&w].sent.iter().all(|p| p.is_some()), "every page written went up");
            // an immediate of the GPU lands in the CPU view, and stays sent
            let gen = |s: &State, t: usize| s.mapped[&t].gen;
            v.s.store_imm(w + 2 * TILE + 0x40, 0x1122_3344_5566_7788, 8, gen(&v.s, w + 2 * TILE));
            assert_eq!(*((c + 2 * TILE + 0x40) as *const u64), 0x1122_3344_5566_7788);
            v.s.store_imm(w + 2 * TILE + 0x48, 0xAABB_CCDD, 4, gen(&v.s, w + 2 * TILE));
            assert_eq!(*((c + 2 * TILE + 0x48) as *const u32), 0xAABB_CCDD);
            assert!(v.s.chunk_clean(w + 2 * TILE + 0x40), "what landed is what the GPU has: not sent again");
            // so does a copy back …
            let back: Vec<u8> = (0..256u32).map(|i| i as u8 ^ 0x5A).collect();
            v.s.copy_back(back.as_ptr() as *mut u8, w + 3 * TILE + 0x200, 0, 256, gen(&v.s, w + 3 * TILE), true);
            assert_eq!(std::slice::from_raw_parts((c + 3 * TILE + 0x200) as *const u8, 256), &back[..]);
            // … but one recorded while the tile showed another part of the CPU view
            let before = gen(&v.s, w + TILE);
            v.s.drop_mirror(w + TILE);
            v.s.link_mirror(w + TILE, c + 4 * TILE);
            assert!(v.s.new_mirror(w + TILE, c + 4 * TILE, 0, false));
            let tile = v.cpu_tile(4);
            v.s.copy_back(back.as_ptr() as *mut u8, w + TILE, 0, 256, before, true);
            assert_eq!(v.cpu_tile(4), tile, "a copy back of what the tile showed before is left out");
        }
    }

    /// What the title's CPU writes has to reach the tiles over it: the watched ranges are the CPU
    /// tiles (one with no committed page yet too: a write there gives its tile video memory),
    /// never the GPU view's own addresses. Forgotten, the tiles give their video memory back.
    #[test]
    fn the_watch_ranges_of_mirrored_tiles_are_their_cpu_views() {
        unsafe {
            let Some(mut v) = Views::new(41) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 2);
            v.write(0, 2, 2);
            v.s.asks.extend((0..3).map(|k| w + k * TILE));
            let gen = mapped_gen();
            v.s.map_mirrors();
            assert!(mapped_gen() > gen, "readers of the ranges see them changed");
            assert_eq!(v.s.watched_ranges(), vec![(c, c + 3 * TILE)]);
            assert!(!v.s.mapped.contains_key(&(w + 2 * TILE)), "no committed page: no video memory yet");
            assert_eq!(v.s.mirrors[&(c + 2 * TILE)], vec![w + 2 * TILE]);
            assert_eq!(v.s.mirrored, 2);
            let gen = mapped_gen();
            v.s.forget_mirrors_in(c, c + 41 * TILE);
            assert!(mapped_gen() > gen);
            assert!(v.s.watched_ranges().is_empty() && v.s.mirrors.is_empty() && v.s.mirror_src.is_empty());
            assert_eq!(v.s.mirrored, 0);
            assert!(v.s.parked.contains_key(&w) && v.s.parked.contains_key(&(w + TILE)), "their video memory goes back");
            v.submit(&[]);
        }
    }

    /// The CPU view decommitted under a tile (told before Windows does it): its pages are holes,
    /// cleared on the GPU and not read again (a read would fault); a page written once committed
    /// again is one no more (a lazily committed page of the view). A tile with no committed page
    /// left gives its video memory back, and gets it again when its CPU tile is written.
    #[test]
    fn a_decommit_of_the_cpu_view_makes_holes_and_parks_the_tile() {
        unsafe {
            let Some(mut v) = Views::new(43) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 1);
            v.write(0, 1, 3);
            // the second tile: its first 8 pages committed, the rest not yet
            assert!(!VirtualAlloc((c + TILE) as *const c_void, 8 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::write_bytes((c + TILE) as *mut u8, 0x3C, 8 * PAGE);
            v.s.asks.extend([w, w + TILE]);
            v.s.map_mirrors();
            assert_eq!(v.s.mapped[&(w + TILE)].holes, 0xFF00);
            v.submit(&[]);
            assert!(v.s.mapped[&w].sent[3].is_some());

            v.s.source_decommitted_in(c + 3 * PAGE, PAGE);
            assert_ne!(VirtualFree((c + 3 * PAGE) as *mut c_void, PAGE, MEM_DECOMMIT), 0);
            assert_eq!(v.s.mapped[&w].holes, 1 << 3);
            assert!(v.s.mapped[&w].sent[3].is_none(), "the GPU holds zeros there");
            assert!(v.s.to_zero.contains(&(w + 3 * PAGE, PAGE)));
            // written before the decommit (the look reports it): asked again, still a hole, not read
            v.submit(&[c + 3 * PAGE]);
            assert_eq!((v.s.mapped[&w].holes, v.s.counts.rechecks), (1 << 3, 1));
            // a page of the second tile committed on its first write: no hole from that submit on
            assert!(!VirtualAlloc((c + TILE + 9 * PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            *((c + TILE + 9 * PAGE + 0x80) as *mut u32) = 0x600D_F00D;
            v.submit(&[c + TILE + 9 * PAGE]);
            assert_eq!(v.s.mapped[&(w + TILE)].holes, 0xFF00 & !(1 << 9));
            assert!(v.s.mapped[&(w + TILE)].sent[9].is_some(), "and it went up");

            // the first tile decommitted whole: parked, still wanted over its CPU tile
            let pool = v.s.mapped[&w].pool;
            v.s.source_decommitted_in(c, TILE);
            assert_ne!(VirtualFree(c as *mut c_void, TILE, MEM_DECOMMIT), 0);
            assert!(!v.s.mapped.contains_key(&w));
            assert_eq!(v.s.parked.get(&w), Some(&pool));
            assert_eq!((v.s.mirrored, v.s.counts.given_back), (1, 1));
            assert_eq!(v.s.mirrors[&c], vec![w]);
            // committed and written again: back from parked, filled whole at the submit
            v.commit(0, 1);
            *((c + 5 * PAGE) as *mut u8) = 1;
            v.s.mirror_writes(&[c + 5 * PAGE], PAGE);
            assert_eq!((v.s.mapped[&w].pool, v.s.mapped[&w].holes), (pool, 0));
            assert!(v.s.fill.contains(&w) && v.s.to_zero.contains(&(w, TILE)));
            assert_eq!(v.s.mirrored, 2);
            v.submit(&[c + 5 * PAGE]);
            assert!(v.s.fill.is_empty() && v.s.mapped[&w].sent[5].is_some());
        }
    }

    /// `map_mirror` is called while the title records, on any thread: it does not wait for the
    /// state (held by a submit around its D3D12 calls), and asks for each tile once.
    #[test]
    fn map_mirror_does_not_wait_for_the_state() {
        // (an address nothing reserves: only the asking is looked at)
        const AT: usize = 0x7ff3_0000_0000;
        let asked = returns_while_the_state_is_held(|| {
            assert_eq!(map_mirror(AT + 0x10, 2 * TILE), 3, "three tiles queued");
            assert_eq!(map_mirror(AT + TILE, 3 * TILE), 1, "the fourth alone is new");
            assert_eq!(map_mirror(AT + 0x20, 0x100), 0, "a range in tiles asked for queues none");
            // (with the state held no submit takes them)
            let asks = MIRROR_ASKS.lock().unwrap();
            for k in 0..4 {
                assert_eq!(asks.iter().filter(|&&t| t == AT + k * TILE).count(), 1, "tile {k} is asked for once");
            }
        });
        assert!(asked, "map_mirror waited for the sparse state's lock");
        // (other tests forget tiles too: the count only goes up)
        let before = mirror_asks_forgotten();
        forget_mirrors(AT, 4 * TILE);
        assert!(!MIRROR_ASKED.contains(AT) && !MIRROR_ASKS.lock().unwrap().contains(&AT), "forgotten tiles may be asked for again");
        assert!(mirror_asks_forgotten() >= before + 4, "whoever remembers having asked for them is told");
    }

    /// A thread asks for a tile while `forget_mirrors` forgets it: once the forget is done the
    /// tile is queued or may be asked for again, and the thread is told to. Before, the asked set
    /// lost the tile first, the ask in between queued it anew and the queue lost it after: asked
    /// for, in no queue, never mapped (the GPU read zeros there).
    #[test]
    fn a_tile_asked_for_while_it_is_forgotten_is_not_lost() {
        // (an address nothing reserves: only the asking is looked at)
        const AT: usize = 0x7ff3_0010_0000;
        assert_eq!(map_mirror(AT, TILE), 1);
        let between = std::rc::Rc::new(std::cell::Cell::new(None));
        let b = between.clone();
        BETWEEN_FORGETS.with(|f| f.set(Some(Box::new(move || b.set(Some(map_mirror(AT, TILE)))))));
        let (counted, before) = (ASKS_FORGOTTEN.with(|n| n.get()), mirror_asks_forgotten());
        forget_mirrors(AT, TILE);
        assert_eq!(between.get(), Some(0), "asked for in between: still asked for, not queued anew");
        assert_eq!(ASKS_FORGOTTEN.with(|n| n.get()), counted + 1, "the forget after it is counted");
        assert!(mirror_asks_forgotten() > before);
        let queued = MIRROR_ASKS.lock().unwrap().contains(&AT);
        assert!(!MIRROR_ASKED.contains(AT) || queued, "asked for and in no queue: never mapped");
        assert_eq!(map_mirror(AT, TILE), 1, "asked for again: queued");
        forget_mirrors(AT, TILE);
    }

    /// An asked tile that `take_asks` leaves out (no pair: here past the GPU view's span) leaves
    /// the asked set uncounted, and may be asked for again. Counted, an address bound at every
    /// draw emptied every thread's memory of its asks at every submit, and every thread asked
    /// again.
    #[test]
    fn an_ask_left_out_is_not_counted_and_may_be_asked_for_again() {
        unsafe {
            let Some(mut v) = Views::new(127) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let g = v.w + 127 * TILE;
            assert_eq!(cpu_of(g), g, "no pair past the views");
            assert_eq!(map_mirror(g + 0x40, 0x100), 1);
            // (taken by the test's state as a submit takes them; other tests' asks stay queued)
            MIRROR_ASKS.lock().unwrap().retain(|&t| t != g);
            v.s.asks.push(g);
            let counted = ASKS_FORGOTTEN.with(|n| n.get());
            v.s.map_mirrors();
            assert!(v.s.asks.is_empty() && v.s.mirror_src.is_empty() && !v.s.mapped.contains_key(&g), "left out");
            assert_eq!(ASKS_FORGOTTEN.with(|n| n.get()), counted, "not counted");
            assert_eq!(map_mirror(g, TILE), 1, "asked for again");
            MIRROR_ASKS.lock().unwrap().retain(|&t| t != g);
            MIRROR_ASKED.remove(g);
        }
    }

    /// An ask a submit took that `forget_mirrors_in` drops (its CPU view forgotten) may be asked
    /// for again, and the threads that remember asking for it are told. Before, it stayed in the
    /// asked set: asked for again, it was not queued.
    #[test]
    fn an_ask_dropped_by_a_forget_may_be_asked_for_again() {
        unsafe {
            let Some(mut v) = Views::new(131) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            assert_eq!(map_mirror(w + 0x40, 0x100), 1);
            MIRROR_ASKS.lock().unwrap().retain(|&t| t != w);
            v.s.asks.push(w);
            let (counted, before) = (ASKS_FORGOTTEN.with(|n| n.get()), mirror_asks_forgotten());
            v.s.forget_mirrors_in(c, c + TILE);
            assert!(v.s.asks.is_empty(), "the ask over the CPU tile forgotten goes");
            assert_eq!(ASKS_FORGOTTEN.with(|n| n.get()), counted + 1, "and is counted");
            assert!(mirror_asks_forgotten() > before);
            assert_eq!(map_mirror(w, TILE), 1, "asked for again");
            MIRROR_ASKS.lock().unwrap().retain(|&t| t != w);
            MIRROR_ASKED.remove(w);
        }
    }

    /// Past `mirror_cap` tiles over CPU views get no video memory (counted once each); room
    /// comes back when a CPU tile is decommitted whole, and a refused tile gets it when its CPU
    /// tile is written.
    #[test]
    fn mirrored_tiles_stop_at_the_cap() {
        unsafe {
            let Some(mut v) = Views::new(47) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 4);
            v.write(0, 4, 5);
            v.s.mirror_cap = 2;
            v.s.asks.extend((0..4).map(|k| w + k * TILE));
            v.s.map_mirrors();
            assert_eq!(v.s.mirrored, 2);
            assert!(v.s.mapped.contains_key(&w) && v.s.mapped.contains_key(&(w + TILE)));
            assert!(!v.s.mapped.contains_key(&(w + 2 * TILE)) && !v.s.mapped.contains_key(&(w + 3 * TILE)));
            assert_eq!(v.s.counts.refused, 2);
            assert_eq!(v.s.mirror_src.len(), 4, "the refused ones are still wanted");
            v.s.mirror_writes(&[c + 2 * TILE], PAGE);
            assert_eq!((v.s.mirrored, v.s.counts.refused), (2, 2), "refused again, counted once");
            v.s.source_decommitted_in(c, TILE);
            assert_ne!(VirtualFree(c as *mut c_void, TILE, MEM_DECOMMIT), 0);
            assert_eq!(v.s.mirrored, 1);
            v.s.mirror_writes(&[c + 2 * TILE + PAGE], PAGE);
            assert!(v.s.mapped.contains_key(&(w + 2 * TILE)), "room again: written, it gets video memory");
            assert_eq!((v.s.mirrored, v.s.counts.refused, v.s.refused.len()), (2, 2, 1));
            v.submit(&[c + 2 * TILE + PAGE]);
        }
    }

    /// The holes of a CPU view's tiles are asked of Windows once for each region of pages in one
    /// state, not once a tile: VirtualQuery walks a region to its end, and a call a tile over a
    /// long committed range took the square of its length. They are what `query_tile_holes` says.
    #[test]
    fn the_holes_of_a_cpu_view_take_a_query_a_region() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        const TILES: usize = 64;
        unsafe {
            let c = VirtualAlloc(std::ptr::null(), TILES * TILE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert!(c != 0 && !VirtualAlloc(c as *const c_void, TILES * TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
            // (pages never touched are not in the working set: VirtualQuery answers for them)
            assert_ne!(VirtualFree((c + 10 * TILE + 5 * PAGE) as *mut c_void, PAGE, MEM_DECOMMIT), 0);
            *((c + 20 * TILE) as *mut u8) = 1;
            let tiles: Vec<usize> = (0..TILES).map(|k| c + k * TILE).collect();
            let q0 = VIRTUAL_QUERIES.with(|n| n.get());
            let holes = source_holes(&tiles);
            let queries = VIRTUAL_QUERIES.with(|n| n.get()) - q0;
            for (k, &t) in tiles.iter().enumerate() {
                assert_eq!(holes[k], query_tile_holes(t), "tile {k}");
            }
            assert_eq!(holes[10], 1 << 5);
            assert!(queries <= 3, "{queries} VirtualQuery calls for {TILES} tiles in 3 regions");
            assert_ne!(VirtualFree(c as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// A tile of a GPU view holds what its CPU view holds once it is asked for (the CPU wrote it
    /// before), and follows the CPU's writes, the many small ones too (dispatches) – far into the
    /// window and across the 1 GiB a dispatch reaches on a GPU. Before, the GPU read zeros there.
    #[test]
    fn a_gpu_view_tile_is_filled_from_its_cpu_view_and_follows_its_writes() {
        unsafe {
            let soft = software(&test_device());
            // (the software device takes small reserved buffers only)
            let tiles = if soft { 53 } else { (3 << 30) / TILE };
            let ks: [usize; 5] = if soft { [0, 1, 2, 3, 52] } else { [0, 1, (1 << 30) / TILE - 1, (1 << 30) / TILE, tiles - 1] };
            let Some(mut v) = Views::new(tiles) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            for &k in &ks {
                v.commit(k, 1);
                v.write(k, 1, 7 + k as u64);
            }
            v.s.asks.extend(ks.iter().map(|&k| v.w + k * TILE));
            v.submit(&[]);
            let buf = v.buf.clone();
            for &k in &ks {
                assert!(v.gpu_bytes(&buf, k * TILE, TILE) == v.cpu_tile(k), "tile {k} is filled from its CPU view");
            }
            // the CPU writes runs of each again
            let mut x = 0x1234_5678_9ABC_DEF1u64;
            let (mut dirty, mut small) = (Vec::new(), 0usize);
            for &k in &ks {
                let t = v.c + k * TILE;
                let mut o = 0;
                loop {
                    o += CHUNK * (1 + next(&mut x) as usize % 9);
                    let n = if next(&mut x) % 23 == 0 { SCATTER_RUN + CHUNK * (next(&mut x) as usize % 40) } else { CHUNK * (1 + next(&mut x) as usize % 12) };
                    if o + n > TILE {
                        break;
                    }
                    small += (n < SCATTER_RUN) as usize;
                    for i in 0..n / 8 {
                        *((t + o) as *mut u64).add(i) = next(&mut x);
                    }
                    dirty.extend(((t + o) & !(PAGE - 1)..t + o + n).step_by(PAGE));
                    o += n;
                }
            }
            dirty.sort_unstable();
            dirty.dedup();
            assert!(small > SCATTER_FROM * 4, "{small} small runs");
            let before = SCATTERED[1].load(Ordering::Relaxed);
            v.submit(&dirty);
            if !soft {
                assert!(SCATTERED[1].load(Ordering::Relaxed) - before >= SCATTER_FROM as u64, "the small runs went up in dispatches");
            }
            for &k in &ks {
                assert!(v.gpu_bytes(&buf, k * TILE, TILE) == v.cpu_tile(k), "tile {k} follows its CPU view");
            }
        }
    }

    /// What the title's lists write into a GPU view (a copy, an immediate) lands in the CPU view
    /// behind it, where the title's CPU reads it. Before, it was copied to the view's own
    /// addresses, which fault.
    #[test]
    fn a_gpu_write_into_a_gpu_view_lands_in_its_cpu_view() {
        unsafe {
            let Some(mut v) = Views::new(59) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 1);
            v.s.asks.push(w);
            v.submit(&[]);
            assert!(v.s.mapped.contains_key(&w));
            let mut up: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(256);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            v.dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut up).expect("upload buffer");
            let up = up.expect("upload buffer");
            let mut p: *mut c_void = std::ptr::null_mut();
            up.Map(0, None, Some(&mut p)).expect("Map");
            for i in 0..64u32 {
                *(p as *mut u32).add(i as usize) = 0xC0DE_0000 | i;
            }
            let alloc: ID3D12CommandAllocator = v.dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = v.dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            list.CopyBufferRegion(&v.buf, 0x100, &up, 0, 256);
            list.Close().expect("close");
            let title = v.title.clone();
            title.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            v.s.after_execute(&title, &[(w + 0x100, 256)], &[(w + 0x800, 0x0123_4567_89AB_CDEF, 8)]);
            let t = std::time::Instant::now();
            while *((c + 0x100) as *const u32) != 0xC0DE_0000 || *((c + 0x800) as *const u64) == 0 {
                v.s.drain_readbacks(Drain::TitleDone);
                assert!(t.elapsed() < std::time::Duration::from_secs(5), "the copy back did not land");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            for i in 0..64u32 {
                assert_eq!(*((c + 0x100) as *const u32).add(i as usize), 0xC0DE_0000 | i, "dword {i}");
            }
            assert_eq!(*((c + 0x800) as *const u64), 0x0123_4567_89AB_CDEF, "the immediate");
        }
    }

    /// A buffer the title places in a GPU view shows the CPU view behind it: its tiles are the
    /// window's, filled from the CPU view (one with no committed page yet too, which the buffer
    /// needs: it shows the CPU's writes once they come). Before, its tiles were the view's own
    /// uncommitted pages, zeros for good.
    #[test]
    fn a_buffer_placed_in_a_gpu_view_shows_its_cpu_view() {
        unsafe {
            let Some(mut v) = Views::new(61) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 1);
            v.write(0, 1, 11);
            let mut res: Option<ID3D12Resource> = None;
            v.dev.CreateReservedResource(&buffer_desc(2 * TILE as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut res).expect("CreateReservedResource");
            let res = res.expect("reserved buffer");
            let tiles = v.s.buffer_tiles(w, 2 * TILE);
            assert_eq!(tiles.iter().map(|t| t.0).collect::<Vec<_>>(), vec![w, w + TILE], "both tiles, the one with no committed page too");
            assert_eq!((v.s.mapped[&w].src, v.s.mapped[&(w + TILE)].src), (c, c + TILE));
            assert_eq!(v.s.mapped[&(w + TILE)].holes, 0xFFFF);
            v.s.map_tiles(&res, w, &tiles);
            for &(_, tile) in &tiles {
                *v.s.pinned.entry(tile).or_default() += 1;
            }
            v.submit(&[]);
            assert!(v.gpu_bytes(&res, 0, TILE) == v.cpu_tile(0), "the buffer shows the CPU view");
            assert!(v.gpu_bytes(&res, TILE, TILE) == vec![0u8; TILE]);
            v.commit(1, 1);
            v.write(1, 1, 12);
            let dirty: Vec<usize> = (0..PAGES_PER_TILE).map(|i| c + TILE + i * PAGE).collect();
            v.submit(&dirty);
            assert!(v.gpu_bytes(&res, TILE, TILE) == v.cpu_tile(1), "and the CPU's writes to it");
        }
    }

    /// `n` physical pages of the title mapped at their home, one after the other (a pool for the
    /// queue's page mappings whose pages a tile may show): (the pages, the home). None: no unified
    /// memory.
    unsafe fn home_pages(n: usize) -> Option<(Vec<usize>, usize)> {
        use crate::hle::kernelx::{AllocateTitlePhysicalPages, MapTitlePhysicalPages};
        crate::gpu::uma::get()?;
        let (mut k, mut pages) = (n, vec![0usize; n]);
        assert_eq!(AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut k, pages.as_mut_ptr()), 1);
        let home = MapTitlePhysicalPages(std::ptr::null_mut(), n, 0, 4, pages.as_mut_ptr()) as usize;
        assert!(home != 0 && home % TILE == 0);
        Some((pages, home))
    }

    /// The 64 KiB at `at` written with bytes made from `seed`; returns them.
    unsafe fn fill_tile(at: usize, seed: u64) -> Vec<u8> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        for i in 0..TILE / 8 {
            *(at as *mut u64).add(i) = next(&mut x);
        }
        std::slice::from_raw_parts(at as *const u8, TILE).to_vec()
    }

    /// One call of the queue's page-mapping method: `ranges` (kind, start, count) from GPU-view
    /// address `dest` on, in the pool of `pages` pages at `pool`.
    unsafe fn map_pages_at(dest: usize, ranges: &[[u32; 3]], pool: usize, pages: u32) {
        // (a batch: the first destination, the number of ranges, a pointer to them)
        let batch = [dest as u64, ranges.len() as u64, ranges.as_ptr() as u64];
        assert!(crate::hle::kernelx::mirrors::apply_batches(1, batch.as_ptr() as usize, pool, pages).is_some());
    }

    /// A tile of a GPU view shows the page the queue's page mappings put there, from the submit
    /// after the call: a mapped page, nothing for an unmapped one (no video memory, still asked
    /// for), the CPU view for one never named; the page's later writes, a remap, a page mapped
    /// where nothing was (with no new ask) and an unmap. No ask is forgotten on the way, and a
    /// range over two pages kept apart is not one piece on the CPU.
    #[test]
    fn a_tile_of_a_gpu_view_shows_the_page_mapped_there() {
        unsafe {
            let Some(mut v) = Views::new(163) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let Some((mut pages, h)) = home_pages(3) else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let (w, buf) = (v.w, v.buf.clone());
            let mut pool: Vec<Vec<u8>> = (0..3).map(|k| fill_tile(h + k * TILE, 40 + k as u64)).collect();
            // three tiles over their CPU view, asked for and filled
            v.commit(0, 3);
            v.write(0, 3, 44);
            v.ask(3);
            v.submit(&[]);
            assert!(v.s.mapped.contains_key(&(w + TILE)));
            let forgotten = ASKS_FORGOTTEN.with(|n| n.get());
            let shown = page_tiles()[0];

            // pool page 0 at view page 0, view page 1 unmapped, view page 2 left as it is
            map_pages_at(w, &[[0, 0, 1], [2, 0, 1]], h, 3);
            v.take_remaps(3);
            assert_eq!(v.s.remaps, vec![w, w + TILE], "the asked tiles of the changed pages");
            v.submit(&[]);
            assert!(v.gpu_bytes(&buf, 0, TILE) == pool[0], "the pool page");
            assert!(!v.s.mapped.contains_key(&(w + TILE)) && !v.s.mirror_src.contains_key(&(w + TILE)), "unmapped: no video memory");
            assert!(MIRROR_ASKED.contains(w + TILE), "and still asked for");
            assert!(v.gpu_bytes(&buf, 2 * TILE, TILE) == v.cpu_tile(2), "never named: its CPU tile");
            assert!(page_tiles()[0] >= shown + 2);
            // a write to the pool page goes up
            pool[0] = fill_tile(h, 50);
            v.submit(&(0..PAGES_PER_TILE).map(|i| h + i * PAGE).collect::<Vec<_>>());
            assert!(v.gpu_bytes(&buf, 0, TILE) == pool[0], "the pool page's write");

            // view page 0 remapped to pool page 2, pool page 1 mapped at view page 1 (asked for
            // while unmapped: no new ask)
            assert_eq!(map_mirror(w + TILE, 0x100), 0, "still asked for");
            map_pages_at(w, &[[0, 2, 1], [0, 1, 1]], h, 3);
            assert_eq!(super::super::view_want::cpu_run((w + TILE - 8) as u64, 16), None, "two pages kept apart");
            v.take_remaps(3);
            v.submit(&[]);
            assert!(v.gpu_bytes(&buf, 0, TILE) == pool[2], "the new page");
            assert!(v.gpu_bytes(&buf, TILE, TILE) == pool[1], "the page mapped where nothing was");
            // an unmap takes the video memory
            map_pages_at(w, &[[2, 0, 1]], h, 3);
            v.take_remaps(3);
            v.submit(&[]);
            assert!(!v.s.mapped.contains_key(&w) && MIRROR_ASKED.contains(w));
            assert_eq!(ASKS_FORGOTTEN.with(|n| n.get()), forgotten, "no ask forgotten on the way");

            // freed: the tile over a freed page shows nothing
            assert_eq!(crate::hle::kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), 3, pages.as_mut_ptr()), 1);
            v.take_remaps(3);
            v.submit(&[]);
            assert!(!v.s.mapped.contains_key(&(w + TILE)));
        }
    }

    /// A buffer placed over GPU-view tiles keeps its video memory while the pages under it
    /// change: a remap shows the new page on the same pool tile, an unmap zeros, and a buffer
    /// placed over an unmapped page shows the page once it is mapped there.
    #[test]
    fn a_buffer_over_a_remapped_tile_reads_the_new_page() {
        unsafe {
            let Some(mut v) = Views::new(167) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let Some((mut pages, h)) = home_pages(2) else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let w = v.w;
            let pool: Vec<Vec<u8>> = (0..2).map(|k| fill_tile(h + k * TILE, 60 + k as u64)).collect();
            // pool page 0 at view page 0, view page 1 unmapped; the buffer's tiles are asked for
            // (as a placed buffer's are, `view_want`)
            map_pages_at(w, &[[0, 0, 1], [2, 0, 1]], h, 2);
            v.ask(2);
            let mut res: Option<ID3D12Resource> = None;
            v.dev.CreateReservedResource(&buffer_desc(2 * TILE as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut res).expect("CreateReservedResource");
            let res = res.expect("reserved buffer");
            let tiles = v.s.buffer_tiles(w, 2 * TILE);
            assert_eq!(tiles.len(), 2);
            assert_eq!((v.s.mapped[&w].src, v.s.mapped[&(w + TILE)].src), (h, no_source()), "the unmapped one over nothing");
            v.s.map_tiles(&res, w, &tiles);
            for &(_, tile) in &tiles {
                *v.s.pinned.entry(tile).or_default() += 1;
            }
            v.submit(&[]);
            assert!(v.gpu_bytes(&res, 0, TILE) == pool[0], "the buffer shows the pool page");
            assert!(v.gpu_bytes(&res, TILE, TILE) == vec![0u8; TILE]);
            let (p0, p1) = (v.s.mapped[&w].pool, v.s.mapped[&(w + TILE)].pool);

            // remapped: the new page on the same pool tile
            map_pages_at(w, &[[0, 1, 1]], h, 2);
            v.take_remaps(2);
            v.submit(&[]);
            assert!(v.gpu_bytes(&res, 0, TILE) == pool[1], "the buffer reads the new page");
            assert_eq!(v.s.mapped[&w].pool, p0, "on the same pool tile");
            // unmapped: zeros, the pool tile kept
            map_pages_at(w, &[[2, 0, 1]], h, 2);
            v.take_remaps(2);
            v.submit(&[]);
            assert!(v.gpu_bytes(&res, 0, TILE) == vec![0u8; TILE], "unmapped: zeros");
            assert_eq!((v.s.mapped[&w].src, v.s.mapped[&w].pool), (no_source(), p0));
            // the page mapped under the buffer's second tile, placed while it was unmapped
            map_pages_at(w + TILE, &[[0, 0, 1]], h, 2);
            v.take_remaps(2);
            v.submit(&[]);
            assert!(v.gpu_bytes(&res, TILE, TILE) == pool[0], "shown once mapped");
            assert_eq!((v.s.mapped[&(w + TILE)].src, v.s.mapped[&(w + TILE)].pool), (h, p1));

            assert_eq!(crate::hle::kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), 2, pages.as_mut_ptr()), 1);
            v.take_remaps(2);
            v.submit(&[]);
        }
    }

    /// A tile follows its page when the page moves (`retarget`): the link moves, and its pool
    /// tile, gen and sent hashes stay, with nothing cleared or filled; a scan of the new place
    /// sends nothing when its bytes are the same and the chunk written when they differ; a copy
    /// back recorded before the move lands at the new place; a page never used, not committed at
    /// its new place, gives the video memory up.
    #[test]
    fn a_tile_follows_its_page_without_being_filled_again() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        unsafe {
            let Some(mut v) = Views::new(173) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let Some((mut pages, h)) = home_pages(1) else { return crate::gpu::recomp::runner::skip_test("no unified memory") };
            let (w, buf) = (v.w, v.buf.clone());
            let bytes = fill_tile(h, 70);
            map_pages_at(w, &[[0, 0, 1]], h, 1);
            v.ask(1);
            v.submit(&[]);
            assert!(v.gpu_bytes(&buf, 0, TILE) == bytes, "the page at its home");
            // its bytes copied to another place (as kernelx moves a page), and the tile follows
            let r = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert!(r != 0 && r % TILE == 0 && !VirtualAlloc(r as *const c_void, TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::copy_nonoverlapping(h as *const u8, r as *mut u8, TILE);
            let e = &v.s.mapped[&w];
            let (pool, gen, sent) = (e.pool, e.gen, e.sent.clone());
            let (fills, followed) = (v.s.counts.fills, page_tiles()[1]);
            v.s.retarget(w, h, r, true);
            let e = &v.s.mapped[&w];
            assert_eq!((e.src, e.holes, e.pool, e.gen), (r, 0, pool, gen), "the link moves, the video memory stays");
            assert!(e.sent == sent, "and what was sent");
            assert_eq!(v.s.mirror_src[&w], r);
            assert!(!v.s.mirrors.contains_key(&h) && v.s.mirrors[&r] == vec![w]);
            assert!(v.s.fill.is_empty() && v.s.later.is_empty() && v.s.to_zero.range((w, 0)..(w + TILE, 0)).next().is_none(), "nothing cleared or filled");
            assert!(page_tiles()[1] > followed);
            v.s.retarget(w, h, r + TILE, true);
            assert_eq!(v.s.mirror_src[&w], r, "a tile linked elsewhere stays");

            // the new place's pages written (kernelx's known write): the same bytes, nothing sent
            let dirty: Vec<usize> = (0..PAGES_PER_TILE).map(|i| r + i * PAGE).collect();
            v.submit(&dirty);
            assert_eq!(v.s.counts.fills, fills, "no fill");
            assert!(v.s.mapped[&w].sent == sent, "nothing sent");
            // a chunk that differs goes up
            *((r + 3 * PAGE + 0x40) as *mut u64) ^= 0xFFFF;
            v.submit(&[r + 3 * PAGE]);
            assert!(v.s.mapped[&w].sent != sent);
            assert!(v.gpu_bytes(&buf, 0, TILE) == std::slice::from_raw_parts(r as *const u8, TILE), "the chunk written");
            // a copy back recorded before the move lands at the new place
            let back = [0x5Au8; 256];
            v.s.copy_back(back.as_ptr() as *mut u8, w + 0x200, 0, 256, gen, true);
            assert_eq!(std::slice::from_raw_parts((r + 0x200) as *const u8, 256), &back[..]);

            // a page never used: not committed at its new place, no video memory for it
            v.s.retarget(w, r, r + TILE, false);
            assert!(!v.s.mapped.contains_key(&w) && v.s.parked.contains_key(&w) && v.s.later.contains(&(w, r + TILE)));
            v.submit(&[]);
            assert!(!v.s.mapped.contains_key(&w) && v.s.mirror_src[&w] == r + TILE, "linked, with no video memory until its page is written");

            assert_eq!(crate::hle::kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), 1, pages.as_mut_ptr()), 1);
            v.take_remaps(1);
            assert_ne!(VirtualFree(r as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// A decommit of the CPU view under way – the state told, Windows not done yet – when a submit
    /// asks Windows again about pages of it written before (the look reports them): they stay
    /// holes and are not read, and a tile with no page left gets no video memory back. Before,
    /// Windows still had them: the submit read them while they were freed, and kept them as
    /// committed for good.
    #[test]
    fn pages_being_decommitted_stay_holes_for_a_submit() {
        unsafe {
            let Some(mut v) = Views::new(67) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 2);
            v.write(0, 2, 13);
            v.s.asks.extend([w, w + TILE]);
            v.submit(&[]);
            assert_eq!((v.s.mapped[&w].holes, v.s.mapped[&(w + TILE)].holes), (0, 0));
            // a page of the first tile and the whole second: the decommitter tells the state …
            let p = c + 3 * PAGE;
            let (one, shown) = freeing_begins(v.s.sources, p, PAGE);
            assert!(shown, "tiles show it");
            v.s.source_decommitted_in(p, PAGE);
            let (two, _) = freeing_begins(v.s.sources, c + TILE, TILE);
            v.s.source_decommitted_in(c + TILE, TILE);
            assert!(!v.s.mapped.contains_key(&(w + TILE)));
            // … a submit, with pages of both written before …
            v.submit(&[p, c + TILE + 2 * PAGE]);
            assert_eq!(v.s.mapped[&w].holes, 1 << 3, "Windows has the page still: a hole all the same");
            assert!(v.s.mapped[&w].sent[3].is_none(), "and not read");
            assert!(!v.s.mapped.contains_key(&(w + TILE)), "no page left: no video memory");
            // … then Windows
            assert_ne!(VirtualFree(p as *mut c_void, PAGE, MEM_DECOMMIT), 0);
            freeing_ends(v.s.sources, one);
            assert_ne!(VirtualFree((c + TILE) as *mut c_void, TILE, MEM_DECOMMIT), 0);
            freeing_ends(v.s.sources, two);
            assert!(lock_sources(v.s.sources).freeing.is_empty());
            v.submit(&[p, c + TILE + 2 * PAGE]);
            assert_eq!(v.s.mapped[&w].holes, 1 << 3);
            assert!(!v.s.mapped.contains_key(&(w + TILE)));
        }
    }

    /// A tile asked for while a decommit of its CPU view is under way (nothing showed that memory
    /// when the decommit began, so the state was not told): the pages being decommitted are holes
    /// in it and are not read, and a decommit that begins once it is linked finds it. Before,
    /// Windows had them still, and the submit filled the tile from them while they were freed.
    #[test]
    fn a_tile_linked_during_a_decommit_of_its_cpu_view_takes_those_pages_as_holes() {
        unsafe {
            let Some(mut v) = Views::new(71) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 1);
            v.write(0, 1, 17);
            let (span, shown) = freeing_begins(v.s.sources, c + 2 * PAGE, 2 * PAGE);
            assert!(!shown, "nothing shows it yet: the decommit does not wait for the state");
            v.s.asks.push(w);
            v.submit(&[]);
            assert_eq!(v.s.mapped[&w].holes, 0b1100);
            assert!(v.s.mapped[&w].sent[2].is_none() && v.s.mapped[&w].sent[3].is_none(), "not read");
            assert!(v.s.mapped[&w].sent[4].is_some(), "the rest went up");
            assert_ne!(VirtualFree((c + 2 * PAGE) as *mut c_void, 2 * PAGE, MEM_DECOMMIT), 0);
            freeing_ends(v.s.sources, span);
            let (span, shown) = freeing_begins(v.s.sources, c + 8 * PAGE, PAGE);
            assert!(shown, "linked: a decommit from here on tells the state");
            freeing_ends(v.s.sources, span);
        }
    }

    /// The pair of a tile asked for goes between its look up and its link (the CPU view released,
    /// and its `forget_mirrors` found nothing linked): the tile is not linked, and decommits of
    /// that memory do not wait for the state. Before, it stayed linked over memory that may be
    /// given out anew.
    #[test]
    fn a_tile_whose_pair_went_before_its_link_is_left_out() {
        unsafe {
            let Some(mut v) = Views::new(73) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 1);
            v.write(0, 1, 19);
            v.s.asks.push(w);
            BEFORE_LINK.with(|f| f.set(Some(Box::new(move || crate::hle::kernelx::mirrors::released(c)))));
            v.s.map_mirrors();
            assert!(v.s.mirror_src.is_empty() && v.s.mirrors.is_empty() && !v.s.mapped.contains_key(&w));
            assert!(lock_sources(v.s.sources).runs.is_empty());
            assert_eq!(v.s.mirrored, 0);
        }
    }

    /// `source_decommit` and `source_release` hold the pages as being freed while Windows frees
    /// them (the caller's call, whose result they return), and not after.
    #[test]
    fn a_decommit_of_cpu_memory_is_under_way_while_windows_does_it() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        unsafe {
            let a = VirtualAlloc(std::ptr::null(), TILE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            assert!(a != 0);
            let under_way = |s: (usize, usize)| lock_sources(&SOURCES).freeing.contains(&s);
            let r = source_decommit(a + PAGE + 0x10, PAGE, || {
                assert!(under_way((a + PAGE, a + 3 * PAGE)), "the pages it touches");
                VirtualFree((a + PAGE) as *mut c_void, 2 * PAGE, MEM_DECOMMIT)
            });
            assert_ne!(r, 0);
            assert!(!under_way((a + PAGE, a + 3 * PAGE)));
            let r = source_release(a, TILE, || {
                assert!(under_way((a, a + TILE)));
                VirtualFree(a as *mut c_void, 0, MEM_RELEASE)
            });
            assert_ne!(r, 0);
            assert!(!under_way((a, a + TILE)));
        }
    }

    /// A CPU view that lies in a sparse window itself (a title's reservation of 64 KiB pages)
    /// decommitted there: the tiles over it get the holes too, and a tile with no page left gives
    /// its video memory back; the pages are held as being freed while Windows frees them
    /// (`decommit`). Before, only the view's own tiles got them, and copies back, immediates and
    /// scans through the tiles over it touched the decommitted pages.
    #[test]
    fn a_decommit_of_a_cpu_view_in_a_window_makes_holes_in_the_tiles_over_it() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        unsafe {
            let Some(mut v) = Views::new(79) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            let mut own: Option<ID3D12Resource> = None;
            v.dev.CreateReservedResource(&buffer_desc((79 * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut own).expect("CreateReservedResource");
            // (after the GPU view's window: a tile in both is the view's)
            v.s.windows.push((c, own.expect("reserved buffer")));
            v.commit(0, 1);
            v.s.commit_range(c, TILE);
            v.write(0, 1, 23);
            v.s.asks.push(w);
            v.submit(&[]);
            assert_eq!((v.s.mapped[&c].src, v.s.mapped[&w].src, v.s.mapped[&w].holes), (0, c, 0));
            v.s.decommit_range(c + 3 * PAGE, PAGE);
            assert_ne!(VirtualFree((c + 3 * PAGE) as *mut c_void, PAGE, MEM_DECOMMIT), 0);
            assert_eq!(v.s.mapped[&c].holes, 1 << 3);
            assert_eq!(v.s.mapped[&w].holes, 1 << 3, "the tile over it has the hole too");
            assert!(v.s.mapped[&w].sent[3].is_none() && v.s.to_zero.contains(&(w + 3 * PAGE, PAGE)));
            v.s.decommit_range(c, TILE);
            assert_ne!(VirtualFree(c as *mut c_void, TILE, MEM_DECOMMIT), 0);
            assert!(!v.s.mapped.contains_key(&w) && v.s.parked.contains_key(&w), "no page left: its video memory goes back");
            assert_eq!(v.s.mirrored, 0);
            // (written before: the look reports it, and nothing reads it)
            v.submit(&[c + 5 * PAGE]);
            assert!(!v.s.mapped.contains_key(&w));

            let a = VirtualAlloc(std::ptr::null(), TILE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            assert!(a != 0);
            let r = decommit(a, PAGE, || {
                assert!(lock_sources(&SOURCES).freeing.contains(&(a, a + PAGE)), "being freed while Windows does it");
                VirtualFree(a as *mut c_void, PAGE, MEM_DECOMMIT)
            });
            assert_ne!(r, 0);
            assert!(!lock_sources(&SOURCES).freeing.contains(&(a, a + PAGE)));
            assert_ne!(VirtualFree(a as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// A copy back or an immediate recorded before the CPU memory a tile shows changed – a page
    /// decommitted and committed again, or the tile shown over another CPU tile – lands nowhere:
    /// the GPU's bytes were of the memory before. What is recorded now lands. Before, a copy
    /// back compared only the CPU address and an immediate nothing, and the old bytes went into
    /// the zero chunks of the new memory.
    #[test]
    fn a_copy_back_or_immediate_of_memory_since_changed_lands_nowhere() {
        unsafe {
            let Some(mut v) = Views::new(83) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 2);
            v.s.asks.extend([w, w + TILE]);
            v.submit(&[]);
            let (g0, g1) = (v.s.mapped[&w].gen, v.s.mapped[&(w + TILE)].gen);
            // a page of the first CPU tile decommitted, committed again and written in part: asked
            // again at the submit, it is no hole
            let p = c + 3 * PAGE;
            v.s.source_decommitted_in(p, PAGE);
            assert_ne!(VirtualFree(p as *mut c_void, PAGE, MEM_DECOMMIT), 0);
            assert!(!VirtualAlloc(p as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            *(p as *mut u64) = 0x600D;
            v.submit(&[p]);
            assert_eq!(v.s.mapped[&w].holes, 0);
            let back = vec![0xABu8; 256];
            v.s.copy_back(back.as_ptr() as *mut u8, w + 3 * PAGE, 0, 256, g0, true);
            assert_eq!(*(p as *const u64), 0x600D);
            assert!(std::slice::from_raw_parts((p + 8) as *const u8, 248).iter().all(|&b| b == 0), "the GPU's old bytes do not land in the new memory");
            // the second tile shown over the first CPU tile: an immediate of before lands nowhere …
            v.s.drop_mirror(w + TILE);
            v.s.link_mirror(w + TILE, c);
            assert!(v.s.new_mirror(w + TILE, c, 0, false));
            v.s.store_imm(w + TILE + 0x80, 0x1234_5678, 8, g1);
            assert_eq!(*((c + 0x80) as *const u64), 0);
            // … one of now does
            v.s.store_imm(w + TILE + 0x80, 0x1234_5678, 8, v.s.mapped[&(w + TILE)].gen);
            assert_eq!(*((c + 0x80) as *const u64), 0x1234_5678);
            // (a submission's immediates are recorded with the memory they are for)
            if v.s.rbq.is_some() {
                let title = v.title.clone();
                v.s.after_execute(&title, &[], &[(w + 0x100, 0x77, 8)]);
                assert_eq!(v.s.readbacks.lock().unwrap().imm_jobs.back().map(|j| j.writes[0].3), Some(v.s.mapped[&w].gen));
                let t = std::time::Instant::now();
                while !v.s.readbacks.lock().unwrap().imm_jobs.is_empty() {
                    v.s.drain_readbacks(Drain::TitleDone);
                    assert!(t.elapsed() < std::time::Duration::from_secs(5), "the immediate did not land");
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                assert_eq!(*((c + 0x100) as *const u64), 0x77);
            }
        }
    }

    /// A buffer placed over a tile of a GPU view at `mirror_cap` gets the tile all the same: it
    /// maps its tiles once, and the window tile could get video memory later that the buffer
    /// would not have. Before, it was left out of the buffer for good.
    #[test]
    fn a_buffer_placed_at_the_cap_gets_its_tiles() {
        unsafe {
            let Some(mut v) = Views::new(97) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 2);
            v.write(0, 2, 31);
            v.s.mirror_cap = 1;
            v.s.asks.push(w);
            v.s.map_mirrors();
            assert_eq!(v.s.mirrored, 1);
            let tiles = v.s.buffer_tiles(w + TILE, TILE);
            assert_eq!(tiles.iter().map(|t| t.0).collect::<Vec<_>>(), vec![w + TILE], "the buffer has its tile");
            assert_eq!(v.s.mapped[&(w + TILE)].src, c + TILE);
            assert_eq!((v.s.mirrored, v.s.counts.refused), (2, 0), "it counts");
            v.submit(&[]);
        }
    }

    /// A tile kept mapped for a placed buffer while its CPU tile was decommitted whole takes its
    /// video memory back once the CPU tile is committed and written again, at `mirror_cap` too:
    /// the memory is held already. Before, it was refused, and stayed mapped in the window and the
    /// buffer with the old bytes while the new ones never went up.
    #[test]
    fn a_kept_tile_is_taken_back_at_the_cap() {
        unsafe {
            let Some(mut v) = Views::new(101) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 2);
            v.write(0, 2, 37);
            v.s.asks.extend([w, w + TILE]);
            v.submit(&[]);
            let pool = v.s.mapped[&w].pool;
            // (a placed buffer uses it)
            *v.s.pinned.entry(pool).or_default() += 1;
            v.s.source_decommitted_in(c, TILE);
            assert_ne!(VirtualFree(c as *mut c_void, TILE, MEM_DECOMMIT), 0);
            v.s.flush_parked();
            assert_eq!((v.s.kept.get(&w), v.s.mirrored), (Some(&pool), 1));
            v.s.mirror_cap = 1;
            v.commit(0, 1);
            v.write(0, 1, 41);
            v.s.mirror_writes(&[c + PAGE], PAGE);
            assert_eq!(v.s.mapped.get(&w).map(|e| e.pool), Some(pool), "taken back");
            assert_eq!((v.s.mirrored, v.s.counts.refused), (2, 0));
            assert!(v.s.fill.contains(&w));
            v.submit(&[c + PAGE]);
        }
    }

    /// A tile refused at `mirror_cap` gets video memory at a submit once there is room again,
    /// filled with what its CPU view held (written once before, maybe never again); while there
    /// is none, a write to its CPU tile does not ask Windows about it. Before, it waited for a
    /// write after the room came back, and asked Windows at every write meanwhile.
    #[test]
    fn a_tile_refused_at_the_cap_gets_video_memory_when_there_is_room() {
        unsafe {
            let Some(mut v) = Views::new(103) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 2);
            v.write(0, 2, 43);
            // (the third CPU tile half committed: asked about, the rest takes a VirtualQuery)
            assert!(!VirtualAlloc((c + 2 * TILE) as *const c_void, 8 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::write_bytes((c + 2 * TILE) as *mut u8, 0x5A, 8 * PAGE);
            v.s.mirror_cap = 2;
            v.s.asks.extend([w, w + TILE, w + 2 * TILE]);
            v.submit(&[]);
            assert!(!v.s.mapped.contains_key(&(w + 2 * TILE)) && v.s.refused.contains(&(w + 2 * TILE)));
            let q0 = VIRTUAL_QUERIES.with(|n| n.get());
            v.submit(&[c + 2 * TILE]);
            assert_eq!(VIRTUAL_QUERIES.with(|n| n.get()), q0, "at the cap, Windows is not asked about it");
            assert!(!v.s.mapped.contains_key(&(w + 2 * TILE)));
            // room again (a CPU tile decommitted whole), and no write: it gets its video memory
            v.s.source_decommitted_in(c, TILE);
            assert_ne!(VirtualFree(c as *mut c_void, TILE, MEM_DECOMMIT), 0);
            assert_eq!(v.s.mirrored, 1);
            v.submit(&[]);
            assert_eq!(v.s.mapped.get(&(w + 2 * TILE)).map(|e| e.holes), Some(0xFF00));
            assert!(v.s.refused.is_empty() && v.s.mirrored == 2 && v.s.counts.refused == 1);
            assert!((0..8).all(|i| v.s.mapped[&(w + 2 * TILE)].sent[i].is_some()), "what the CPU view held went up");
        }
    }

    /// A tile that found no video memory is refused like one at the cap (not asked for again at
    /// every ask), and gets it at a submit once there is some. Before, it was not even recorded:
    /// it read zeros until its CPU tile was written again.
    #[test]
    fn a_tile_refused_for_want_of_video_memory_gets_it_later() {
        unsafe {
            let Some(mut v) = Views::new(107) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let w = v.w;
            v.commit(0, 1);
            v.write(0, 1, 47);
            // (as after a heap that could not be made)
            v.s.no_memory = Some(std::time::Instant::now());
            v.s.asks.push(w);
            v.submit(&[]);
            assert!(!v.s.mapped.contains_key(&w) && v.s.refused.contains(&w));
            assert_eq!(v.s.counts.refused, 1);
            v.s.no_memory = None;
            v.submit(&[]);
            assert!(v.s.mapped.contains_key(&w) && v.s.refused.is_empty());
            assert!(v.s.mapped[&w].sent.iter().all(|p| p.is_some()), "what the CPU view held went up");
        }
    }

    /// A pool tile freed is used again `COOL` after at the earliest, the oldest first and before
    /// a tile of `free`. Before, it went to `free` and was the next one taken.
    #[test]
    fn a_freed_pool_tile_cools_before_it_is_used_again() {
        let t0 = std::time::Instant::now();
        let ms = |n: u64| t0 + std::time::Duration::from_millis(n);
        let mut free = FreePools::new();
        let mut cooling: VecDeque<((u32, u32), std::time::Instant)> = VecDeque::new();
        cooling.push_back(((0, 5), ms(0)));
        cooling.push_back(((0, 6), ms(10)));
        assert_eq!(reuse_tile(&mut free, &mut cooling, ms(1999)), None, "freed 1999 ms ago: still cooling");
        assert!(!cooled(&cooling, ms(1999)));
        free.push((1, 9));
        assert_eq!(reuse_tile(&mut free, &mut cooling, ms(1999)), Some((1, 9)), "a free tile meanwhile");
        assert!(cooled(&cooling, ms(2000)));
        free.push((1, 8));
        assert_eq!(reuse_tile(&mut free, &mut cooling, ms(2005)), Some((0, 5)), "cooled: before a free tile");
        assert_eq!(reuse_tile(&mut free, &mut cooling, ms(2005)), Some((1, 8)), "the next one freed 10 ms later is not");
        cooling.push_back(((2, 1), ms(20)));
        assert_eq!(reuse_tile(&mut free, &mut cooling, ms(5000)), Some((0, 6)), "the oldest first");
        assert_eq!(reuse_tile(&mut free, &mut cooling, ms(5000)), Some((2, 1)));
        assert_eq!(reuse_tile(&mut free, &mut cooling, ms(5000)), None);
    }

    /// The tile mapping line gives the calls and tiles since the counts before, by kind of window.
    #[test]
    fn the_tile_mapping_line_counts_since_the_last() {
        let before = [1, 10, 2, 20, 3, 30, 4, 40];
        assert_eq!(tile_calls_line(&before, &before), None, "no call: no line");
        let now = [2, 74, 2, 20, 3, 30, 7, 41];
        assert_eq!(
            tile_calls_line(&now, &before).as_deref(),
            Some("sparse windows 1 calls mapping 64 tiles, 0 calls unmapping 0 tiles; GPU views 0 calls mapping 0 tiles, 3 calls unmapping 1 tiles")
        );
    }

    /// The GPU-view diagnostics: the poison pattern a switch names, the keep decision within the
    /// cap, the buckets of the submits a tile was away, and the line of the counts since the last.
    #[test]
    fn view_diag_switches_buckets_and_line() {
        assert_eq!(poison_pattern(None), None);
        assert_eq!(poison_pattern(Some("1")), Some(0x7FC0_DEAD));
        assert_eq!(poison_pattern(Some("0x0000BEEF")), Some(0xBEEF));
        assert_eq!(poison_pattern(Some("0x0")), None, "a zero pattern is no poison");
        assert_eq!(poison_pattern(Some("0xnope")), None);
        assert!(!keep_view_tile(false, 1, 10), "off: given back");
        assert!(keep_view_tile(true, 10, 10), "at the cap with this tile: kept");
        assert!(!keep_view_tile(true, 11, 10), "past it (a placed buffer's tiles): given back");
        let buckets: Vec<usize> = [0, 1, 2, 8, 9, 64, 65, u64::MAX].into_iter().map(back_bucket).collect();
        assert_eq!(buckets, vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let mut before = [0u64; VS];
        before[VS_NOW] = 5;
        assert_eq!(view_stats_line(&before, &before), None, "nothing counted: no line");
        let mut now = [7, 3, 2, 1, 4, 0, 1, 0, 3 << 19, 20, 6, 9, 1, 12, 0];
        for (n, b) in now.iter_mut().zip(before.iter_mut()).take(VS_NOW) {
            *b = 1;
            *n += 1;
        }
        assert_eq!(
            view_stats_line(&now, &before).as_deref(),
            Some("GPU-view tiles (2 s): 7 asked for, 3 filled, 2 given back, 1 kept; back after 0-1 / 2-8 / 9-64 / more submits: 4 / 0 / 1 / 0 (1.5 MB filled again); 6 of 20 submits left asked tiles waiting; now 9 with video memory, 1 kept, 12 waiting, 0 refused")
        );
    }

    /// `XCL_VIEW_KEEP`: a tile whose CPU view is decommitted stays mapped where it is (kept, not
    /// parked, so no submit unmaps it), and is taken back from there once its CPU tile is written
    /// again, counted in the bucket of the submits it was away.
    #[test]
    fn a_kept_tile_stays_mapped_and_comes_back() {
        unsafe {
            let Some(mut v) = Views::new(191) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            (v.s.diag.keep, v.s.diag.stats) = (true, true);
            v.commit(0, 1);
            v.write(0, 1, 9);
            v.s.asks.push(w);
            v.s.map_mirrors();
            v.submit(&[]);
            let pool = v.s.mapped[&w].pool;
            v.s.source_decommitted_in(c, TILE);
            assert_ne!(VirtualFree(c as *mut c_void, TILE, MEM_DECOMMIT), 0);
            assert_eq!((v.s.kept.get(&w), v.s.parked.get(&w)), (Some(&pool), None));
            assert_eq!((v.s.mirrored, v.s.counts.given_back, v.s.diag.kept), (0, 0, 1));
            v.submit(&[]);
            assert_eq!(v.s.kept.get(&w), Some(&pool), "no submit unmaps it");
            v.commit(0, 1);
            *((c + 5 * PAGE) as *mut u8) = 1;
            v.s.mirror_writes(&[c + 5 * PAGE], PAGE);
            assert_eq!((v.s.mapped[&w].pool, v.s.kept.get(&w)), (pool, None), "back from kept");
            assert_eq!((v.s.diag.back, v.s.diag.back_bytes), ([1, 0, 0, 0], TILE as u64), "away one submit");
            v.submit(&[c + 5 * PAGE]);
        }
    }

    /// `XCL_VIEW_POISON`: a tile of the window without video memory reads the pattern, one with
    /// it what was filled there, and one given back the pattern again once a submit unmaps it.
    #[test]
    fn a_tile_without_video_memory_reads_the_poison() {
        unsafe {
            let Some(mut v) = Views::new(193) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.s.poison_windows(0x7FC0_DEAD);
            assert!(v.s.poison.is_some(), "a poison tile");
            let buf = v.buf.clone();
            let poison: Vec<u8> = std::iter::repeat_n(0x7FC0_DEADu32.to_le_bytes(), 8).flatten().collect();
            assert_eq!(v.gpu_bytes(&buf, 0, 32), poison);
            assert_eq!(v.gpu_bytes(&buf, 192 * TILE + 0x40, 32), poison, "every tile");
            v.commit(0, 1);
            v.write(0, 1, 11);
            v.s.asks.push(w);
            v.s.map_mirrors();
            v.submit(&[]);
            assert_eq!(v.gpu_bytes(&buf, 0, 32), v.cpu_tile(0)[..32].to_vec(), "filled");
            assert_eq!(v.gpu_bytes(&buf, TILE, 32), poison, "the next tile still has none");
            v.s.source_decommitted_in(c, TILE);
            assert_ne!(VirtualFree(c as *mut c_void, TILE, MEM_DECOMMIT), 0);
            v.submit(&[]);
            assert_eq!(v.gpu_bytes(&buf, 0, 32), poison, "given back");
        }
    }

    /// A pool goes when all its tiles are free or cooled: one with a tile still cooling stays.
    #[test]
    fn a_pool_with_a_tile_still_cooling_is_not_retired() {
        let t0 = std::time::Instant::now();
        let ms = |n: u64| t0 + std::time::Duration::from_millis(n);
        // pool 1 free but its last tile, freed at 0 ms; pool 0 free but its first, freed at 1500 ms
        let mut free = FreePools::new();
        for t in 0..POOL_TILES - 1 {
            free.push((1, t));
        }
        for t in 1..POOL_TILES {
            free.push((0, t));
        }
        let cooling = VecDeque::from([((1, POOL_TILES - 1), ms(0)), ((0, 0), ms(1500))]);
        assert!(free_heaps(&free, &cooling, 2, ms(1999)).is_empty(), "their last tiles are cooling");
        assert_eq!(free_heaps(&free, &cooling, 2, ms(2000)), vec![1], "pool 1's cooled");
        assert_eq!(free_heaps(&free, &cooling, 2, ms(3500)), vec![0, 1]);
        let mut partial = FreePools::new();
        for t in 1..POOL_TILES - 1 {
            partial.push((1, t));
        }
        for t in 1..POOL_TILES {
            partial.push((0, t));
        }
        assert_eq!(free_heaps(&partial, &cooling, 2, ms(3500)), vec![0], "a tile of pool 1 in use");
    }

    /// `reuse_tile` packs the fullest partially-used pool heap rather than handing out the oldest
    /// cooled tile from a nearly-empty heap, so draining heaps become completely empty and retire.
    /// Before, FIFO `cooling.pop_front()` re-pinned the nearly-empty heap with new allocations.
    #[test]
    fn reuse_tile_packs_fuller_heaps_so_nearly_empty_pools_retire() {
        let t0 = std::time::Instant::now();
        let ms = |n: u64| t0 + std::time::Duration::from_millis(n);
        let mut free = FreePools::new();
        let mut cooling: VecDeque<((u32, u32), std::time::Instant)> = VecDeque::new();
        // Pool 1 is almost empty (1000 tiles freed earlier at 0 ms, only 24 still live).
        for t in 24..POOL_TILES {
            cooling.push_back(((1, t), ms(0)));
        }
        // Pool 0 is mostly full (only 24 tiles freed later at 10 ms, 1000 still live).
        for t in 0..24 {
            cooling.push_back(((0, t), ms(10)));
        }
        // Allocate 24 new tiles once both have cooled: all 24 must pack into fuller pool 0,
        // leaving draining pool 1 untouched.
        for _ in 0..24 {
            let (h, _) = reuse_tile(&mut free, &mut cooling, ms(2010)).expect("a packed tile");
            assert_eq!(h, 0, "must pack into fuller pool 0 instead of re-pinning draining pool 1");
        }
        assert_eq!(free.heap_len(0), 0, "pool 0 is completely packed");
        assert_eq!(free.heap_len(1), (POOL_TILES - 24) as usize, "pool 1 stayed untouched");

        // When pool 1's remaining 24 tiles free and cool, all of pool 1 is free and ready to retire.
        for t in 0..24 {
            cooling.push_back(((1, t), ms(2100)));
        }
        assert_eq!(free_heaps(&free, &cooling, 2, ms(4100)), vec![1]);

        // An all-free heap is skipped while a partially-used heap has space, and `has_empty_heap`
        // reports it so `retire_due` retires it even after `cooling` has been drained.
        free.push((0, 0));
        free.drain_cooling(&mut cooling, ms(4100));
        assert!(cooling.is_empty());
        assert!(free.has_empty_heap());
        assert_eq!(reuse_tile(&mut free, &mut cooling, ms(4100)), Some((0, 0)), "partial pool 0 before all-free pool 1");
        assert_eq!(free.heap_len(1), POOL_TILES as usize, "all-free pool 1 left intact for retirement");
        assert!(retire_due(&cooling, free.has_empty_heap(), ms(0), ms(4100)));
        // When no partial pool remains, the all-free pool is reused instead of creating a new heap.
        assert_eq!(reuse_tile(&mut free, &mut cooling, ms(4100)).map(|(h, _)| h), Some(1));
    }

    /// A GPU-view tile given back and unmapped at a submit is not the video memory of the next
    /// tile asked for: that gets another, and the tile is used again once it cooled. Before, it
    /// was the next tile taken, mapped, cleared and filled on the sparse queue while the title's
    /// work submitted before could still read it.
    #[test]
    fn a_tile_given_back_is_not_used_again_before_it_cooled() {
        unsafe {
            let Some(mut v) = Views::new(113) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 3);
            v.write(0, 3, 59);
            v.s.asks.push(w);
            v.submit(&[]);
            let pool = v.s.mapped[&w].pool;
            v.s.forget_mirrors_in(c, c + TILE);
            assert_eq!(v.s.parked.get(&w), Some(&pool));
            v.submit(&[]);
            // (unmapped at the submit, and quarantined until the GPU is past it)
            settle_quarantined(&mut v.s);
            assert_eq!(v.s.cooling.iter().map(|t| t.0).collect::<Vec<_>>(), vec![pool], "unmapped, cooling");
            v.s.asks.push(w + TILE);
            v.submit(&[]);
            assert_ne!(v.s.mapped[&(w + TILE)].pool, pool, "another tile");
            // (as 2 s later)
            for t in v.s.cooling.iter_mut() {
                t.1 = t.1.checked_sub(COOL).expect("an earlier instant");
            }
            v.s.asks.push(w + 2 * TILE);
            v.submit(&[]);
            assert_eq!(v.s.mapped[&(w + 2 * TILE)].pool, pool, "cooled: used again");
            assert!(v.s.cooling.is_empty());
        }
    }

    /// A submit that frees no tile tries to retire pools once a tile cooled or a heap waits to
    /// go, `COOL` after the last try at the earliest.
    #[test]
    fn a_submit_with_nothing_freed_retires_pools_once_tiles_cooled() {
        let t0 = std::time::Instant::now();
        let ms = |n: u64| t0 + std::time::Duration::from_millis(n);
        // the last tile of a pool freed at 0 ms, by a flush that tried at 0 ms
        let cooling = VecDeque::from([((0, 0), ms(0))]);
        assert!(!retire_due(&cooling, false, ms(0), ms(1999)), "still cooling");
        assert!(retire_due(&cooling, false, ms(0), ms(2000)), "cooled");
        assert!(!retire_due(&cooling, false, ms(1000), ms(2000)), "tried 1 s ago");
        assert!(!retire_due(&VecDeque::new(), false, ms(0), ms(5000)), "nothing freed, nothing retired");
        assert!(retire_due(&VecDeque::new(), true, ms(0), ms(2000)), "a retired heap to drop");
        assert!(!retire_due(&VecDeque::new(), true, ms(500), ms(2000)));
    }

    /// A pool whose last tile a submit freed goes at a submit `COOL` later that frees none.
    /// Before, only a submit that freed tiles retired pools, while those it freed still cooled:
    /// the pool's 64 MB stayed until the title gave back another tile.
    #[test]
    fn a_pool_goes_once_its_last_tile_cooled_with_nothing_parked() {
        unsafe {
            let Some(mut v) = Views::new(113) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 3);
            v.write(0, 3, 61);
            v.s.asks.push(w);
            v.submit(&[]);
            let pool = v.s.mapped[&w].pool;
            v.s.forget_mirrors_in(c, c + TILE);
            v.submit(&[]);
            // (unmapped at the submit, and quarantined until the GPU is past it)
            settle_quarantined(&mut v.s);
            assert_eq!(v.s.cooling.iter().map(|t| t.0).collect::<Vec<_>>(), vec![pool], "unmapped, cooling");
            assert!(v.s.pools[pool.0 as usize].is_some(), "its last tile is cooling");
            v.submit(&[]);
            assert!(v.s.pools[pool.0 as usize].is_some(), "less than COOL later");
            // (as 2 s later)
            for t in v.s.cooling.iter_mut() {
                t.1 = t.1.checked_sub(COOL).expect("an earlier instant");
            }
            v.s.retire_tried = v.s.retire_tried.checked_sub(COOL).expect("an earlier instant");
            assert!(v.s.parked.is_empty());
            v.submit(&[]);
            assert!(v.s.pools[pool.0 as usize].is_none(), "retired with nothing parked");
            assert!(v.s.cooling.is_empty() && !v.s.free.has_pool(pool.0));
            assert_eq!(v.s.retired.len(), 1);
        }
    }

    /// A submit fills tiles from their CPU views up to `fill_pages` committed pages (not tiles):
    /// the rest wait for the next submits in their order, and tiles asked for later wait behind
    /// them. Before, a submit took 4096 tiles at once (256 MiB hashed and uploaded on the title's
    /// thread with the state's lock held).
    #[test]
    fn a_submit_fills_a_bounded_amount_of_committed_memory_in_order() {
        unsafe {
            let Some(mut v) = Views::new(109) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            v.commit(0, 6);
            v.write(0, 6, 53);
            // four CPU tiles with one committed page each
            for k in 6..10 {
                assert!(!VirtualAlloc((c + k * TILE) as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
                *((c + k * TILE) as *mut u64) = k as u64;
            }
            v.s.fill_pages = 2 * PAGES_PER_TILE;
            let shown = |s: &State| (0..10).filter(|&k| s.mapped.contains_key(&(w + k * TILE))).collect::<Vec<_>>();
            v.s.asks.extend((0..6).map(|k| w + k * TILE));
            v.submit(&[]);
            assert_eq!(shown(&v.s), vec![0, 1], "two whole tiles a submit");
            assert_eq!(v.s.mirror_src.len(), 6, "the rest are linked: their CPU tiles' writes are watched");
            // asked for after: behind the first ones
            v.s.asks.extend((6..10).map(|k| w + k * TILE));
            v.submit(&[]);
            assert_eq!(shown(&v.s), vec![0, 1, 2, 3]);
            v.submit(&[]);
            assert_eq!(shown(&v.s), vec![0, 1, 2, 3, 4, 5]);
            v.submit(&[]);
            assert_eq!(shown(&v.s), (0..10).collect::<Vec<_>>(), "four pages: in one submit");
            assert!(v.s.later.is_empty() && v.s.asks.is_empty());
            assert_eq!(v.s.counts.fill_bytes, (6 * PAGES_PER_TILE + 4) as u64 * PAGE as u64);
        }
    }

    /// A copy back the GPU is done with lands before a submit's tiles change: a page that was a
    /// hole when it was recorded, committed and written in part since, does not take what the GPU
    /// had there. Before, the submit cleared the hole first, and the copy put the GPU's bytes into
    /// the page's zero chunks.
    #[test]
    fn copies_back_done_land_before_the_tiles_change() {
        unsafe {
            let Some(mut v) = Views::new(89) else { return crate::gpu::recomp::runner::skip_test("no reserved buffers") };
            let (w, c) = (v.w, v.c);
            assert!(!VirtualAlloc(c as *const c_void, 8 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            v.s.asks.push(w);
            v.submit(&[]);
            assert_eq!(v.s.mapped[&w].holes, 0xFF00);
            // a copy back of the whole tile, done on the GPU
            let mut job = v.s.rb_job(TILE, D3D12_COMMAND_LIST_TYPE_DIRECT).expect("readback job");
            std::ptr::write_bytes(job.ptr, 0xAB, TILE);
            job.copies.push((w, 0, TILE, v.s.mapped[&w].gen));
            let fence = || v.dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            job.fences = Some((fence(), fence()));
            {
                let mut rq = v.s.readbacks.lock().unwrap();
                rq.job_seq += 1;
                job.seq = rq.job_seq;
                rq.rb_jobs.push_back(job);
            }
            RB_PENDING.fetch_add(1, Ordering::Release);
            // a page committed and written in part since
            let p = c + 9 * PAGE;
            assert!(!VirtualAlloc(p as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            *(p as *mut u64) = 0x600D;
            v.submit(&[p]);
            assert!(v.s.readbacks.lock().unwrap().rb_jobs.is_empty(), "the copy landed");
            assert_eq!(*((c + 0x40) as *const u8), 0xAB, "in the pages committed then");
            assert_eq!(v.s.mapped[&w].holes, 0xFF00 & !(1 << 9));
            assert_eq!(*(p as *const u64), 0x600D);
            assert!(std::slice::from_raw_parts((p + 8) as *const u8, PAGE - 8).iter().all(|&b| b == 0), "not in the page committed since");
        }
    }

    struct SendQueue(ID3D12CommandQueue);
    unsafe impl Send for SendQueue {}

    struct IsolatedAsyncState {
        dev: ID3D12Device,
        base: usize,
        gen: u32,
        state: Arc<Mutex<Option<State>>>,
        rb: Arc<SharedReadbacks>,
        rb_submit: Arc<Mutex<RbSubmit>>,
    }

    impl IsolatedAsyncState {
        unsafe fn new() -> Self {
            let (base, mut s) = test_state_with_reserved(1);
            assert!(!VirtualAlloc(base as *const c_void, TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
            s.commit_range(base, TILE);
            s.flush_pending();
            std::ptr::write_bytes(base as *mut u8, 0, TILE);
            let dev = s.dev.clone();
            let gen = s.mapped[&base].gen;
            let rb = Arc::clone(&s.readbacks);
            let rb_submit = Arc::clone(&s.rb_submit);
            assert!(s.rbq.is_some(), "async readback queue required");
            Self { dev, base, gen, state: Arc::new(Mutex::new(Some(s))), rb, rb_submit }
        }

        unsafe fn make_job(&self, byte: u8, len: usize, title_f: ID3D12Fence, done_f: ID3D12Fence, rb_v: u64, done_v: u64) -> RbJob {
            let mut job = take_or_alloc_rb_job(&self.dev, &self.rb, len, D3D12_COMMAND_LIST_TYPE_DIRECT).expect("readback job");
            std::ptr::write_bytes(job.ptr, byte, len);
            job.copies.clear();
            job.copies.push((self.base, 0, len, self.gen));
            job.rb_v = rb_v;
            job.done_v = done_v;
            job.fences = Some((title_f, done_f));
            job
        }
    }

    impl Drop for IsolatedAsyncState {
        fn drop(&mut self) {
            use windows_sys::Win32::System::Memory::MEM_RELEASE;
            let mut old = self.state.lock().unwrap().take();
            if let Some(s) = old.as_mut() {
                unsafe {
                    s.wait_for(s.value);
                }
            }
            drop(old);
            unsafe {
                let _ = VirtualFree(self.base as *mut c_void, 0, MEM_RELEASE);
            }
        }
    }

    /// `settle_readbacks_with` spins on a title-done / copy-pending fence completely outside
    /// `STATE`: another thread acquires `STATE` immediately throughout the GPU copy spin.
    #[test]
    fn settle_readbacks_waits_for_copy_fence_without_holding_state() {
        unsafe {
            let iso = IsolatedAsyncState::new();
            let title_f: ID3D12Fence = iso.dev.CreateFence(1, D3D12_FENCE_FLAG_NONE).expect("title fence");
            let done_f: ID3D12Fence = iso.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("done fence");
            let _unblock = SignalOnDrop(done_f.clone(), 1);
            let mut job = iso.make_job(0xAB, 256, title_f, done_f.clone(), 1, 1);
            {
                let mut rq = iso.rb.lock().unwrap();
                rq.job_seq += 1;
                job.seq = rq.job_seq;
                rq.rb_jobs.push_back(job);
            }
            RB_PENDING.fetch_add(1, Ordering::Release);

            let state_c = Arc::clone(&iso.state);
            let rb_c = Arc::clone(&iso.rb);
            let settled = std::thread::spawn(move || settle_readbacks_with(&state_c, &rb_c, Drain::TitleDone, 200_000, None));

            // Give the settler a moment to enter the fence spin loop.
            std::thread::sleep(std::time::Duration::from_millis(5));
            for _ in 0..10 {
                let g = iso.state.try_lock();
                assert!(g.is_ok(), "STATE was held during GPU copy fence spin");
                drop(g);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            done_f.Signal(1).expect("signal copy done");
            assert!(settled.join().unwrap());
            assert_eq!(*(iso.base as *const u8), 0xAB, "copy landed after done_f signalled");
            assert!(iso.rb.lock().unwrap().rb_jobs.is_empty());
        }
    }

    /// When pending jobs have not finished on the GPU yet, `settle_readbacks_with` returns
    /// promptly without acquiring `STATE` even while another thread holds `STATE`.
    #[test]
    fn settle_readbacks_skips_state_lock_when_no_job_finished() {
        unsafe {
            let iso = IsolatedAsyncState::new();
            let title_f: ID3D12Fence = iso.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("title fence");
            let done_f: ID3D12Fence = iso.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("done fence");
            let imm_f: ID3D12Fence = iso.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("imm fence");
            let mut job = iso.make_job(0xCD, 256, title_f, done_f, 1, 1);
            {
                let mut rq = iso.rb.lock().unwrap();
                rq.job_seq += 1;
                job.seq = rq.job_seq;
                rq.rb_jobs.push_back(job);
                rq.job_seq += 1;
                let seq = rq.job_seq;
                rq.imm_jobs.push_back(ImmJob { seq, fence: imm_f, value: 1, writes: vec![(iso.base + 0x100, 0x99, 8, iso.gen)] });
            }
            RB_PENDING.fetch_add(2, Ordering::Release);

            let held = iso.state.lock().unwrap();
            let state_c = Arc::clone(&iso.state);
            let rb_c = Arc::clone(&iso.rb);
            let (tx, rx) = std::sync::mpsc::channel();
            let t0 = std::time::Instant::now();
            let worker = std::thread::spawn(move || {
                let ok = settle_readbacks_with(&state_c, &rb_c, Drain::TitleDone, 300, Some(5000));
                let _ = tx.send(ok);
            });
            let outcome = rx.recv_timeout(std::time::Duration::from_millis(500));
            let took = t0.elapsed();
            drop(held);
            worker.join().unwrap();

            let ok = outcome.expect("settle blocked on held STATE despite no finished jobs");
            assert!(ok, "unfinished jobs should return true without blocking on STATE");
            assert!(took < std::time::Duration::from_millis(500), "settle waited {took:?} on held STATE despite no finished jobs");
            assert_eq!(iso.rb.lock().unwrap().rb_jobs.len(), 1);
            assert_eq!(iso.rb.lock().unwrap().imm_jobs.len(), 1);
        }
    }

    /// Concurrent settlers racing on overlapping finished `imm_jobs` and `rb_jobs` land every job
    /// in strict ascending `seq` order and never overwrite a newer `seq` with an older one.
    #[test]
    fn concurrent_settle_lands_jobs_in_strict_seq_order() {
        unsafe {
            let iso = IsolatedAsyncState::new();
            let done: ID3D12Fence = iso.dev.CreateFence(1, D3D12_FENCE_FLAG_NONE).expect("done fence");
            for round in 0..16u64 {
                *(iso.base as *mut u64) = 0;
                let v1 = 0x1111_0000_0000_0000 | round;
                let v3 = 0x3333_0000_0000_0000 | round;
                let v5 = 0x5555_0000_0000_0000 | round;
                let mut j2 = iso.make_job(0x22, 8, done.clone(), done.clone(), 1, 1);
                let mut j4 = iso.make_job(0x44, 8, done.clone(), done.clone(), 1, 1);
                {
                    let mut rq = iso.rb.lock().unwrap();
                    rq.job_seq += 1;
                    let s1 = rq.job_seq;
                    rq.imm_jobs.push_back(ImmJob { seq: s1, fence: done.clone(), value: 1, writes: vec![(iso.base, v1, 8, iso.gen)] });
                    rq.job_seq += 1;
                    j2.seq = rq.job_seq;
                    rq.rb_jobs.push_back(j2);
                    rq.job_seq += 1;
                    let s3 = rq.job_seq;
                    rq.imm_jobs.push_back(ImmJob { seq: s3, fence: done.clone(), value: 1, writes: vec![(iso.base, v3, 8, iso.gen)] });
                    rq.job_seq += 1;
                    j4.seq = rq.job_seq;
                    rq.rb_jobs.push_back(j4);
                    rq.job_seq += 1;
                    let s5 = rq.job_seq;
                    rq.imm_jobs.push_back(ImmJob { seq: s5, fence: done.clone(), value: 1, writes: vec![(iso.base, v5, 8, iso.gen)] });
                }
                RB_PENDING.fetch_add(5, Ordering::Release);

                let bar = Arc::new(std::sync::Barrier::new(4));
                let handles: Vec<_> = (0..4)
                    .map(|_| {
                        let s = Arc::clone(&iso.state);
                        let r = Arc::clone(&iso.rb);
                        let b = Arc::clone(&bar);
                        std::thread::spawn(move || {
                            b.wait();
                            assert!(settle_readbacks_with(&s, &r, Drain::TitleDone, 300, None));
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
                assert_eq!(*(iso.base as *const u64), v5, "round {round}: final seq must win");
                let rq = iso.rb.lock().unwrap();
                assert!(rq.rb_jobs.is_empty() && rq.imm_jobs.is_empty() && rq.landing == 0);
            }

            // In-flight landing visibility check: `landing > 0` forces `wait_and_check_ready` to
            // synchronize on `STATE` even after the queues are empty, while the background thread
            // check `has_unpopped_finished_jobs` stays false.
            iso.rb.lock().unwrap().landing = 1;
            assert!(wait_and_check_ready(&iso.rb, Drain::Finished, 0, None) == ReadyCheck::LandNow);
            assert!(!has_unpopped_finished_jobs(&iso.rb));
            iso.rb.lock().unwrap().landing = 0;
        }
    }

    /// `readback_after_execute_with` releases `STATE` before acquiring `RB_SUBMIT` and recording
    /// D3D12 copy commands, and `land_finished_readbacks` returns excess `RbJob`s beyond the 32-job
    /// pool cap so callers drop them outside all locks.
    #[test]
    fn readback_after_execute_records_outside_state_lock_and_drops_excess_outside_locks() {
        unsafe {
            let iso = IsolatedAsyncState::new();
            let title: ID3D12CommandQueue = iso.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
            let mut up: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(256);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            iso.dev
                .CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut up)
                .expect("upload buffer");
            let up = up.expect("upload buffer");
            let mut p: *mut c_void = std::ptr::null_mut();
            up.Map(0, None, Some(&mut p)).expect("Map");
            for i in 0..64u32 {
                *(p as *mut u32).add(i as usize) = 0xC0DE_0000 | i;
            }
            let win_res = iso.state.lock().unwrap().as_ref().unwrap().windows[0].1.clone();
            let alloc: ID3D12CommandAllocator = iso.dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = iso.dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            list.CopyBufferRegion(&win_res, 0, &up, 0, 256);
            list.Close().expect("close");
            {
                let g = iso.state.lock().unwrap();
                let s = g.as_ref().unwrap();
                title.Wait(&s.fence, s.value).expect("wait mapping");
            }
            title.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);

            // Hold `rb_submit` before spawning `readback_after_execute_with`: the worker must finish
            // its `STATE` snapshot (which takes the submitting token, `SubmittingGuard`), release
            // `STATE`, and park on `rb_submit` while the main thread acquires `STATE` cleanly.
            let sub_held = iso.rb_submit.lock().unwrap();
            let state_c = Arc::clone(&iso.state);
            let q_send = SendQueue(title.clone());
            let base = iso.base;
            let worker = std::thread::spawn(move || {
                readback_after_execute_with(&state_c, &q_send.0, &[(base, 256)], &[(base + 0x800, 0xCAFE_BABE_1234_5678, 8)]);
            });
            let t0 = std::time::Instant::now();
            loop {
                // (in the code's order: `STATE`, then the readback queue)
                let state = iso.state.lock().unwrap();
                let snapshot_taken = iso.rb.lock().unwrap().submitting > 0;
                drop(state);
                if snapshot_taken {
                    break;
                }
                assert!(t0.elapsed() < std::time::Duration::from_secs(5), "readback_after_execute_with did not release STATE while rb_submit was held");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            drop(sub_held);
            worker.join().unwrap();

            // Settle until the submitted copy back and immediate both land.
            let t1 = std::time::Instant::now();
            loop {
                if std::ptr::read_volatile(base as *const u32) == 0xC0DE_0000 && std::ptr::read_volatile((base + 0x800) as *const u64) == 0xCAFE_BABE_1234_5678 {
                    break;
                }
                assert!(settle_readbacks_with(&iso.state, &iso.rb, Drain::TitleDone, 2000, None));
                assert!(t1.elapsed() < std::time::Duration::from_secs(5), "async readback did not land");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert_eq!(*((base + 0x800) as *const u64), 0xCAFE_BABE_1234_5678);

            // Part B: excess RbJobs above pool cap (32) are returned to be dropped outside all locks.
            let done: ID3D12Fence = iso.dev.CreateFence(1, D3D12_FENCE_FLAG_NONE).expect("done");
            let mut extra_pool = std::mem::take(&mut iso.rb.lock().unwrap().rb_pool);
            while extra_pool.len() < 32 {
                extra_pool.push(take_or_alloc_rb_job(&iso.dev, &iso.rb, 64 << 10, D3D12_COMMAND_LIST_TYPE_DIRECT).expect("pool job"));
            }
            let mut f1 = iso.make_job(0x77, 64, done.clone(), done.clone(), 1, 1);
            let mut f2 = iso.make_job(0x88, 64, done.clone(), done.clone(), 1, 1);
            {
                let mut rq = iso.rb.lock().unwrap();
                rq.rb_pool.extend(extra_pool);
                assert_eq!(rq.rb_pool.len(), 32);
                rq.job_seq += 1;
                f1.seq = rq.job_seq;
                rq.rb_jobs.push_back(f1);
                rq.job_seq += 1;
                f2.seq = rq.job_seq;
                rq.rb_jobs.push_back(f2);
            }
            RB_PENDING.fetch_add(2, Ordering::Release);

            let excess = {
                let mut g = iso.state.lock().unwrap();
                g.as_mut().unwrap().land_finished_readbacks()
            };
            assert_eq!(iso.rb.lock().unwrap().rb_pool.len(), 32);
            assert_eq!(excess.len(), 2, "excess RbJobs must be returned for drop outside STATE and READBACKS");
            drop(excess);
        }
    }

    /// Contention micro-bench: repeated `STATE` acquisitions stay unblocked while another thread
    /// continuously spins in `settle_readbacks_with` on a title-done / copy-pending fence.
    #[test]
    fn state_lock_latency_stays_unblocked_under_readback_spin_contention() {
        unsafe {
            let iso = IsolatedAsyncState::new();
            let title_f: ID3D12Fence = iso.dev.CreateFence(1, D3D12_FENCE_FLAG_NONE).expect("title fence");
            let done_f: ID3D12Fence = iso.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("done fence");
            let _unblock = SignalOnDrop(done_f.clone(), 1);
            let mut job = iso.make_job(0x5A, 256, title_f, done_f.clone(), 1, 1);
            {
                let mut rq = iso.rb.lock().unwrap();
                rq.job_seq += 1;
                job.seq = rq.job_seq;
                rq.rb_jobs.push_back(job);
            }
            RB_PENDING.fetch_add(1, Ordering::Release);

            let stop = Arc::new(AtomicBool::new(false));
            let state_c = Arc::clone(&iso.state);
            let rb_c = Arc::clone(&iso.rb);
            let stop_c = Arc::clone(&stop);
            let spinner = std::thread::spawn(move || {
                let mut spins = 0u32;
                while !stop_c.load(Ordering::Acquire) {
                    assert!(settle_readbacks_with(&state_c, &rb_c, Drain::TitleDone, 2000, None));
                    spins += 1;
                }
                spins
            });

            std::thread::sleep(std::time::Duration::from_millis(5));
            let mut waits_us = Vec::with_capacity(50);
            for _ in 0..50 {
                let t0 = std::time::Instant::now();
                let g = iso.state.lock().unwrap();
                waits_us.push(t0.elapsed().as_micros() as u64);
                drop(g);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            stop.store(true, Ordering::Release);
            done_f.Signal(1).expect("signal done");
            let spins = spinner.join().unwrap();
            assert!(settle_readbacks_with(&iso.state, &iso.rb, Drain::TitleDone, 2000, None));

            waits_us.sort_unstable();
            let p50 = waits_us[waits_us.len() / 2];
            let p99 = waits_us[(waits_us.len() * 99) / 100];
            let max = *waits_us.last().unwrap();
            eprintln!("sparse lock-split contention bench ({spins} spin calls): STATE lock wait p50={p50} us, p99={p99} us, max={max} us");
            assert!(spins >= 5, "spinner ran too few iterations ({spins})");
            assert!(p99 < 1000, "STATE lock p99={p99} us exceeded 1000 us under readback spin contention");
        }
    }

    /// An async readback plan prepared under `STATE` remains visible to concurrent settling
    /// callers via `ReadbackQueue::submitting` until `submit_async_readback` publishes or cancels
    /// it: even when an unrelated unfinished readback is already queued, a fence observer waits
    /// within its budget instead of returning early with a stale CPU value, returns `false` if its
    /// budget expires before publication, and wakes immediately on plan cancellation.
    #[test]
    fn settle_waits_for_in_flight_plan_publication_and_handles_cancellation() {
        unsafe {
            let iso = IsolatedAsyncState::new();
            let title_q: ID3D12CommandQueue = iso.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
            let dst = iso.base + 0x800;
            std::ptr::write_volatile(dst as *mut u64, 0x1111_2222_3333_4444);

            // Push an unrelated GPU-unfinished job first so `RB_PENDING > 0` already holds.
            let unfin_title: ID3D12Fence = iso.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("unfin title");
            let unfin_done: ID3D12Fence = iso.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("unfin done");
            let mut unfin_job = iso.make_job(0xEE, 64, unfin_title, unfin_done, 1, 1);
            {
                let mut rq = iso.rb.lock().unwrap();
                rq.job_seq += 1;
                unfin_job.seq = rq.job_seq;
                rq.rb_jobs.push_back(unfin_job);
            }
            RB_PENDING.fetch_add(1, Ordering::Release);

            // Submitter prepares an immediate write under `STATE`, drops `STATE`, signals the
            // title fence, waits until that GPU fence completes, then pauses right before
            // publishing into `ReadbackQueue`.
            let (paused_tx, paused_rx) = std::sync::mpsc::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let state_c = Arc::clone(&iso.state);
            let q_send = SendQueue(title_q.clone());
            let submitter = std::thread::spawn(move || {
                let (plan, excess) = {
                    let mut g = state_c.lock().unwrap();
                    let s = g.as_mut().unwrap();
                    s.note_queue(&q_send.0);
                    s.prepare_after_execute(&q_send.0, &[], &[(dst, 0xDEAD_BEEF_CAFE_BABE, 8)])
                };
                drop(excess);
                let plan = plan.expect("immediate write must produce an async readback plan");
                let hook = move |title_f: &ID3D12Fence, v: u64| {
                    let t0 = std::time::Instant::now();
                    while crate::hle::d3d12x::real::fence_completed(title_f) < v {
                        assert!(t0.elapsed() < std::time::Duration::from_secs(5), "title fence did not complete");
                        std::thread::yield_now();
                    }
                    let _ = paused_tx.send(());
                    let _ = release_rx.recv();
                };
                submit_async_readback_inner(&q_send.0, plan, Some(&hook));
            });

            paused_rx.recv_timeout(std::time::Duration::from_secs(5)).expect("submitter did not reach publication pause");
            assert_eq!(iso.rb.lock().unwrap().submitting, 1);
            assert!(iso.state.try_lock().is_ok(), "STATE must already be released during publication pause");

            // Bounded settle that expires while publication is still paused returns `false`
            // instead of claiming success before the immediate is published.
            assert!(!settle_readbacks_with(&iso.state, &iso.rb, Drain::TitleDone, 300, Some(20)));
            assert_eq!(std::ptr::read_volatile(dst as *const u64), 0x1111_2222_3333_4444);

            // A fence observer with sufficient budget waits for publication, then lands the
            // completed immediate before reporting success.
            let (obs_tx, obs_rx) = std::sync::mpsc::channel();
            let obs_state = Arc::clone(&iso.state);
            let obs_rb = Arc::clone(&iso.rb);
            let observer = std::thread::spawn(move || {
                let ok = settle_readbacks_with(&obs_state, &obs_rb, Drain::TitleDone, 300, Some(2000));
                let _ = obs_tx.send(ok);
            });
            assert!(
                obs_rx.recv_timeout(std::time::Duration::from_millis(25)).is_err(),
                "observer returned early before paused plan published"
            );
            release_tx.send(()).expect("release publication");
            let ok = obs_rx.recv_timeout(std::time::Duration::from_secs(2)).expect("observer did not wake after publication");
            assert!(ok, "observer should settle and return true once plan publishes");
            assert_eq!(
                std::ptr::read_volatile(dst as *const u64),
                0xDEAD_BEEF_CAFE_BABE,
                "CPU destination must hold the immediate value when observer returns"
            );
            observer.join().unwrap();
            submitter.join().unwrap();
            assert_eq!(iso.rb.lock().unwrap().submitting, 0);

            // Plan cancellation (e.g. bail on fence allocation failure): dropping `AsyncReadbackPlan`
            // decrements `submitting`, balances `RB_PENDING`, and wakes waiting observers cleanly.
            let (cancelled_plan, excess) = {
                let mut g = iso.state.lock().unwrap();
                g.as_mut().unwrap().prepare_after_execute(&title_q, &[], &[(dst, 0x5555, 8)])
            };
            drop(excess);
            let cancelled_plan = cancelled_plan.expect("plan for cancellation test");
            assert_eq!(iso.rb.lock().unwrap().submitting, 1);
            let (w_tx, w_rx) = std::sync::mpsc::channel();
            let rb_c = Arc::clone(&iso.rb);
            let waiter = std::thread::spawn(move || {
                let r = wait_and_check_ready(&rb_c, Drain::TitleDone, 0, Some(std::time::Instant::now() + std::time::Duration::from_secs(2)));
                let _ = w_tx.send(r);
            });
            std::thread::sleep(std::time::Duration::from_millis(10));
            drop(cancelled_plan);
            let ready = w_rx.recv_timeout(std::time::Duration::from_secs(1)).expect("waiter did not wake on plan drop");
            waiter.join().unwrap();
            assert!(ready == ReadyCheck::NothingReady);
            assert_eq!(iso.rb.lock().unwrap().submitting, 0);
        }
    }
}
