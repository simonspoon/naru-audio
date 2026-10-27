//! The prebuilt sherpa-onnx static libs for Windows reference
//! `__std_find_first_of_trivial_pos_{1,2}`, which only MSVC toolsets 14.44+
//! (VS 2022 17.14) ship in the STL import libs. On older toolsets, point
//! them at the scalar fallbacks in `src/msvc_stl_compat.rs`. `/ALTERNATENAME`
//! applies only to symbols left unresolved, so newer toolsets keep the STL's.
//! `/INCLUDE` pulls the shim into targets that link sherpa without otherwise
//! referencing the lib (the binary, examples).
//!
//! On Apple Silicon, MLX loads `mlx.metallib` from next to the binary first, so
//! link the one mlx-sys builds into `$MLX_RS_METAL_PATH` (.cargo/config.toml)
//! into the profile dir. A symlink, because mlx-sys's build script may not have
//! written it yet: cargo does not order the two. `cp` follows it when installing.
fn main() {
    #[cfg(unix)]
    link_metallib();
    let windows_msvc = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|v| v == "windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").is_ok_and(|v| v == "msvc");
    if windows_msvc {
        for width in [1, 2] {
            println!(
                "cargo:rustc-link-arg=/ALTERNATENAME:__std_find_first_of_trivial_pos_{width}=naru_find_first_of_trivial_pos_{width}"
            );
            println!("cargo:rustc-link-arg-bins=/INCLUDE:naru_find_first_of_trivial_pos_{width}");
            println!(
                "cargo:rustc-link-arg-examples=/INCLUDE:naru_find_first_of_trivial_pos_{width}"
            );
        }
    }
}

#[cfg(unix)]
fn link_metallib() {
    println!("cargo:rerun-if-env-changed=MLX_RS_METAL_PATH");
    let apple_silicon = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|v| v == "macos")
        && std::env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|v| v == "aarch64");
    if apple_silicon && let Some(dir) = std::env::var_os("MLX_RS_METAL_PATH") {
        let metallib = std::path::Path::new(&dir).join("mlx.metallib");
        // OUT_DIR is <profile dir>/build/<pkg>-<hash>/out.
        let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
        let link = out.ancestors().nth(3).unwrap().join("mlx.metallib");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&metallib, &link).expect("link mlx.metallib");
    }
}
