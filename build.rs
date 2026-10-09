//! The prebuilt sherpa-onnx static libs for Windows reference
//! `__std_find_first_of_trivial_pos_{1,2}`, which only MSVC toolsets 14.44+
//! (VS 2022 17.14) ship in the STL import libs. On older toolsets, point
//! them at the scalar fallbacks in `src/msvc_stl_compat.rs`. `/ALTERNATENAME`
//! applies only to symbols left unresolved, so newer toolsets keep the STL's.
//! `/INCLUDE` pulls the shim into targets that link sherpa without otherwise
//! referencing the lib (the binary, examples).
fn main() {
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
