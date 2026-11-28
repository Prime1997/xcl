//! Owner diagnostics adapted from upstream XCL #2697, with heap classes kept separate.
//! These are allocation capacities and tracked CPU ranges, not residency measurements. DXGI
//! usage and process private commit overlap; adding them or subtracting one from the other
//! does not identify driver memory. Untracked allocations remain explicitly unattributed.

#[derive(Clone, Copy)]
struct Segment {
    usage: u64,
    budget: u64,
}

fn segments() -> Option<(Segment, Segment)> {
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D12::ID3D12Device;
    use windows::Win32::Graphics::Dxgi::*;
    let raw = super::real::device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }?;
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;
        let adapter: IDXGIAdapter3 = factory.EnumAdapterByLuid(dev.GetAdapterLuid()).ok()?;
        let query = |group| {
            let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            adapter.QueryVideoMemoryInfo(0, group, &mut info).ok()?;
            Some(Segment { usage: info.CurrentUsage, budget: info.Budget })
        };
        Some((query(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)?, query(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)?))
    }
}

/// Called at the existing activity cadence, with no GPU work, resource creation or waits.
/// Helpers release their own locks before the next owner is queried. Samples are not atomic.
pub fn line(private_commit: u64) -> String {
    let ((texture_local_used, texture_local), (texture_system_used, texture_system)) = super::vram::page_bytes_by_memory();
    let (arenas, arenas_made) = super::vram::arena_bytes();
    let (sparse_pools, sparse_mapped) = super::sparse::pool_bytes();
    let (sparse_cpu, sparse_hashes, sparse_retired) = super::sparse::memory_backing_bytes();
    let coherent = crate::gpu::uma::get().map_or(0, |u| u.size());
    let (textures, texture_logical) = super::title_textures::committed_texture_bytes();
    let (upload_idle, upload_submitted) = super::title_textures::upload_bytes();
    let mb = |bytes: u64| bytes >> 20;
    let dxgi = match segments() {
        Some((local, system)) => format!(
            "DXGI local {} / {} budget, non-local {} / {} budget",
            mb(local.usage), mb(local.budget), mb(system.usage), mb(system.budget)
        ),
        None => "DXGI unknown".into(),
    };
    // Do not treat console texture size as PC allocation size, or heap capacity as resident
    // bytes. Coherent memory is shown once; an imported/driver heap is the same backing.
    format!(
        "memory owners (MiB, capacities; snapshots may differ): process private {}; CPU coherent backing {}, sparse own pages {}, sparse upload hashes {}; DEFAULT target pools {} (unique pages {}), sparse pools {} (mapped {}), sparse retired {}, arenas active capacity {} (made over time {}); L0 target pools {} (unique pages {}); upload idle {}, submitted {}; committed textures {} with {} logical MiB (PC size unknown); {}; driver/PSO, guest allocations outside tracked windows, readbacks and other retired resources not attributed; CPU and GPU figures overlap",
        mb(private_commit), mb(coherent as u64), mb(sparse_cpu as u64), mb(sparse_hashes as u64),
        mb(texture_local as u64), mb(texture_local_used as u64), mb(sparse_pools as u64), mb(sparse_mapped as u64),
        mb(sparse_retired as u64), mb(arenas as u64), mb(arenas_made as u64), mb(texture_system as u64), mb(texture_system_used as u64),
        mb(upload_idle), mb(upload_submitted), textures, mb(texture_logical), dxgi
    )
}
