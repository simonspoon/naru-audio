//! In-process MLX through mlx-rs, the native replacement for the [`super::sidecar`]
//! (naru task 2378: build foundation only, no model yet).

use std::path::PathBuf;

use mlx_rs::{Array, Device, ops, with_device};

/// `<dir of this binary>/mlx.metallib`: where MLX looks first, and where a
/// shipped binary must find it (the build-time path exists only on the build
/// machine). MLX resolves symlinks to the binary, so this does too. Check it
/// before the first GPU op: without a metallib, MLX exits the process.
pub fn metallib_path() -> Result<PathBuf, String> {
    let exe = std::env::current_exe()
        .and_then(|p| p.canonicalize())
        .map_err(|e| format!("current_exe: {e}"))?;
    Ok(exe.with_file_name("mlx.metallib"))
}

/// A 2x2 matmul on the GPU; `Ok` holds its result, `[7, 10, 15, 22]`.
pub fn gpu_matmul() -> Result<Vec<f32>, String> {
    with_device(Device::gpu(), || {
        let a = Array::from_slice(&[1f32, 2., 3., 4.], &[2, 2]);
        let b = ops::matmul(&a, &a).map_err(|e| e.to_string())?;
        b.eval().map_err(|e| e.to_string())?;
        Ok(b.as_slice::<f32>().to_vec())
    })
}

/// `mlx check`: the metallib sits next to the binary and a GPU matmul runs.
pub fn self_check() -> Result<String, String> {
    let metallib = metallib_path()?;
    if !metallib.is_file() {
        return Err(format!("{} missing", metallib.display()));
    }
    let got = gpu_matmul()?;
    if got != [7., 10., 15., 22.] {
        return Err(format!("gpu matmul gave {got:?}, want [7, 10, 15, 22]"));
    }
    Ok(format!(
        "metallib {}\ngpu matmul {got:?}",
        metallib.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_matmul_runs() {
        assert_eq!(gpu_matmul().unwrap(), [7., 10., 15., 22.]);
    }
}
