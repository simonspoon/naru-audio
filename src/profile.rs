//! §3.3 hardware profile: RAM, arch and Rosetta translation, which pick the
//! default models and the §3.5 memory budget; and the process RSS that
//! §3.5 measures loads with.

const GIB: u64 = 1024 * 1024 * 1024;

/// The profile's default models (§3.3): both profiles pick the same ones.
const DEFAULT_STT: &str = "parakeet-tdt-0.6b-v2-int8";
const DEFAULT_TTS: &str = "kokoro-v1.0";

#[derive(Debug, Clone)]
pub struct Profile {
    /// `small` or `large`.
    pub name: &'static str,
    pub ram_bytes: u64,
    /// `std::env::consts::ARCH`: what this build is, not the CPU under it.
    pub arch: &'static str,
    /// `sysctl.proc_translated`: an x86_64 build running under Rosetta.
    pub translated: bool,
}

impl Profile {
    pub fn detect() -> Result<Self, String> {
        let ram_bytes = ram_bytes().ok_or("cannot read the machine's RAM size")?;
        Ok(Self::new(ram_bytes, std::env::consts::ARCH, translated()))
    }

    /// `large` is arm64 with at least 24 GB; everything else is `small`.
    pub fn new(ram_bytes: u64, arch: &'static str, translated: bool) -> Self {
        let name = if arch == "aarch64" && ram_bytes >= 24 * GIB {
            "large"
        } else {
            "small"
        };
        Self {
            name,
            ram_bytes,
            arch,
            translated,
        }
    }

    /// §3.5 `max_resident_bytes`: small min(40% RAM, 6 GiB), large 50% RAM.
    pub fn budget_bytes(&self) -> u64 {
        if self.name == "large" {
            self.ram_bytes / 2
        } else {
            (self.ram_bytes / 10 * 4).min(6 * GIB)
        }
    }

    pub fn default_stt(&self) -> &'static str {
        DEFAULT_STT
    }

    pub fn default_tts(&self) -> &'static str {
        DEFAULT_TTS
    }

    /// The `/health` problem for an x86_64 build under Rosetta.
    pub fn problem(&self) -> Option<String> {
        self.translated.then(|| {
            format!(
                "this {} build is running under Rosetta translation; install the arm64 build",
                self.arch
            )
        })
    }
}

#[cfg(target_os = "macos")]
fn sysctl<T: Default>(name: &std::ffi::CStr) -> Option<T> {
    let mut value = T::default();
    let mut len = std::mem::size_of::<T>();
    // SAFETY: `value` is a `T` and `len` is its size.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut value as *mut T).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len == std::mem::size_of::<T>()).then_some(value)
}

#[cfg(target_os = "macos")]
fn ram_bytes() -> Option<u64> {
    sysctl::<u64>(c"hw.memsize")
}

#[cfg(all(unix, not(target_os = "macos")))]
fn ram_bytes() -> Option<u64> {
    // SAFETY: sysconf has no preconditions.
    let (pages, size) = unsafe {
        (
            libc::sysconf(libc::_SC_PHYS_PAGES),
            libc::sysconf(libc::_SC_PAGESIZE),
        )
    };
    (pages > 0 && size > 0).then(|| pages as u64 * size as u64)
}

#[cfg(windows)]
fn ram_bytes() -> Option<u64> {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
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
    // SAFETY: `status` is a MEMORYSTATUSEX with `length` set, as the call requires.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    (ok != 0 && status.total_phys > 0).then_some(status.total_phys)
}

/// The sysctl does not exist on Intel Macs that predate Rosetta 2: not
/// translated.
#[cfg(target_os = "macos")]
fn translated() -> bool {
    sysctl::<libc::c_int>(c"sysctl.proc_translated") == Some(1)
}

#[cfg(not(target_os = "macos"))]
fn translated() -> bool {
    false
}

/// This process's resident set size in bytes (`task_info` on macOS).
#[cfg(target_os = "macos")]
pub fn rss() -> Option<u64> {
    // SAFETY: `mach_task_basic_info` is plain integers, for which all-zero
    // bytes are a valid value.
    let mut info: libc::mach_task_basic_info = unsafe { std::mem::zeroed() };
    let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
    // SAFETY: `info` is a `mach_task_basic_info` and `count` its size in
    // `natural_t`s. `mach_task_self` is deprecated in libc in favour of the
    // mach2 crate, but is the same call.
    #[allow(deprecated)]
    let rc = unsafe {
        libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            (&mut info as *mut libc::mach_task_basic_info).cast(),
            &mut count,
        )
    };
    (rc == libc::KERN_SUCCESS).then_some(info.resident_size)
}

/// This process's resident set size in bytes (`/proc/self/statm`).
#[cfg(all(unix, not(target_os = "macos")))]
pub fn rss() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    // SAFETY: sysconf has no preconditions.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    (size > 0).then(|| pages * size as u64)
}

/// This process's working set in bytes (`K32GetProcessMemoryInfo`).
#[cfg(windows)]
pub fn rss() -> Option<u64> {
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(
            process: isize,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }
    let mut c = ProcessMemoryCounters {
        cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
    };
    // SAFETY: `c` is a PROCESS_MEMORY_COUNTERS with `cb` set, as the call
    // requires; GetCurrentProcess returns a pseudo-handle needing no close.
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    (ok != 0 && c.working_set_size > 0).then_some(c.working_set_size as u64)
}

/// No resident-size probe on other targets.
#[cfg(not(any(unix, windows)))]
pub fn rss() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_and_budget_follow_the_3_3_table() {
        let intel = Profile::new(64 * GIB, "x86_64", false);
        assert_eq!(intel.name, "small");
        assert_eq!(intel.budget_bytes(), 6 * GIB);

        let small_arm = Profile::new(8 * GIB, "aarch64", false);
        assert_eq!(small_arm.name, "small");
        assert_eq!(small_arm.budget_bytes(), 8 * GIB / 10 * 4);

        let large = Profile::new(64 * GIB, "aarch64", false);
        assert_eq!(large.name, "large");
        assert_eq!(large.budget_bytes(), 32 * GIB);
        assert_eq!(Profile::new(24 * GIB, "aarch64", false).name, "large");
    }

    #[test]
    fn rosetta_is_a_problem() {
        assert!(Profile::new(GIB, "x86_64", false).problem().is_none());
        let p = Profile::new(GIB, "x86_64", true).problem().unwrap();
        assert!(p.contains("Rosetta"), "{p}");
    }

    #[test]
    fn detects_this_machine() {
        let p = Profile::detect().unwrap();
        assert!(p.ram_bytes > 0);
        assert_eq!(p.arch, std::env::consts::ARCH);
        assert!(rss().is_some_and(|r| r > 0));
    }
}
