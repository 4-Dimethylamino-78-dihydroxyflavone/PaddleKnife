//! System memory headroom, for diagnosing GPU "out of memory" on Windows.
//!
//! Under WDDM every GPU allocation is backed by *system commit charge* (RAM +
//! pagefile) so the OS can evict it. A process can therefore be refused GPU
//! memory while the card itself has gigabytes free — all it takes is a machine
//! whose commit limit is nearly exhausted by other applications. ONNX Runtime
//! reports that as `8007000E Not enough memory resources are available`, which
//! reads like a VRAM problem and is not. The failure that motivated Plan 10
//! happened on a 16 GB card with 12 GB free and 6.8 GiB of commit left.

/// Bytes of system commit still available to this process, or `None` where
/// the notion doesn't apply or can't be read.
pub fn available_commit_bytes() -> Option<u64> {
    imp::available_commit_bytes()
}

/// A one-line hint when commit headroom is low enough to explain a GPU
/// allocation failure; `None` when it is not the likely culprit.
pub fn low_commit_hint() -> Option<String> {
    /// Below this, a recognizer batch at the default budget (~4.3 GiB of
    /// commit: 3.6 dedicated + readback staging) is at real risk.
    const LOW_COMMIT_BYTES: u64 = 8 << 30;

    let available = available_commit_bytes()?;
    (available < LOW_COMMIT_BYTES).then(|| {
        format!(
            "only {:.1} GiB of system commit (RAM + pagefile) is available; on Windows every \
             GPU allocation must fit in it, however much VRAM is free — close memory-hungry \
             applications or enlarge the pagefile",
            available as f64 / (1u64 << 30) as f64
        )
    })
}

#[cfg(windows)]
mod imp {
    /// `MEMORYSTATUSEX` from `sysinfoapi.h`.
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        /// "The maximum amount of memory the current process can commit."
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    #[allow(unsafe_code)]
    pub fn available_commit_bytes() -> Option<u64> {
        let mut status = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: `status` is a live, correctly sized and aligned
        // MEMORYSTATUSEX with `length` set as the API requires; the call only
        // writes within it and retains no pointer.
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        (ok != 0).then_some(status.avail_page_file)
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn available_commit_bytes() -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(windows)]
    fn reads_a_plausible_commit_figure_on_windows() {
        let bytes = available_commit_bytes().expect("GlobalMemoryStatusEx succeeds");
        // More than nothing, less than an exbibyte.
        assert!(bytes > 0 && bytes < (1u64 << 60));
    }

    #[test]
    #[cfg(not(windows))]
    fn is_absent_off_windows() {
        assert_eq!(available_commit_bytes(), None);
        assert_eq!(low_commit_hint(), None);
    }
}
