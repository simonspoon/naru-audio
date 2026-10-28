//! §5.3 the MLX sidecar on Apple Silicon: `naru-audio mlx setup|status`
//! (§3.7), the `[mlx] python` key of `config.toml` (§3.1), and the
//! supervised process itself ([`sidecar`]); plus the in-process [`native`]
//! MLX that will replace it.

pub mod breeze;
pub mod qwen3_tts;
pub mod native;
pub mod parakeet;
pub mod sidecar;

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::log::Logger;
use crate::registry::{config_path, read_config};

/// Written into `<home>/mlx/` by `mlx setup`; `python -m naru_audio_mlx`
/// runs from there.
const FILES: &[(&str, &str)] = &[
    ("pyproject.toml", include_str!("../mlx/pyproject.toml")),
    ("uv.lock", include_str!("../mlx/uv.lock")),
    (
        "naru_audio_mlx/__init__.py",
        include_str!("../mlx/naru_audio_mlx/__init__.py"),
    ),
    (
        "naru_audio_mlx/__main__.py",
        include_str!("../mlx/naru_audio_mlx/__main__.py"),
    ),
    (
        "naru_audio_mlx/protocol.py",
        include_str!("../mlx/naru_audio_mlx/protocol.py"),
    ),
];

/// What `mlx status` imports: the sidecar and what it runs on.
const IMPORT_CHECK: &str =
    "import naru_audio_mlx, mlx.core, parakeet_mlx, mlx_whisper, mlx_audio.tts.utils";

/// `<home>/mlx/`: the module, the lock and the venv.
pub fn dir(home: &Path) -> PathBuf {
    home.join("mlx")
}

/// `mlx setup`: writes the embedded environment into `<home>/mlx/`, runs
/// `uv sync --frozen` (venv at `<home>/mlx/.venv`), and records the venv's
/// interpreter in `config.toml`. Running it again brings a changed
/// environment up to date. The interpreter's absolute path.
pub fn setup(home: &Path) -> Result<PathBuf, String> {
    let uv = std::env::var_os("PATH")
        .into_iter()
        .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .map(|dir| dir.join("uv"))
        .find(|uv| uv.is_file())
        .ok_or("`uv` is not on PATH; install it with `brew install uv`, then run `naru-audio mlx setup` again")?;
    let dir = dir(home);
    for (path, text) in FILES {
        let path = dir.join(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        std::fs::write(&path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    let status = Command::new(&uv)
        .args(["sync", "--frozen", "--project"])
        .arg(&dir)
        // The venv goes where the daemon looks for it.
        .env_remove("UV_PROJECT_ENVIRONMENT")
        .status()
        .map_err(|e| format!("cannot run {}: {e}", uv.display()))?;
    if !status.success() {
        return Err(format!("`uv sync --frozen` failed ({status})"));
    }
    // Not canonicalized: the venv's `python` is a symlink, and only the
    // link itself runs inside the venv.
    let python = std::path::absolute(dir.join(".venv").join("bin").join("python"))
        .map_err(|e| format!("cannot resolve the venv's python: {e}"))?;
    write_python(home, &python)?;
    Ok(python)
}

/// Serve startup: brings the sidecar's deployed Python (`naru_audio_mlx/`)
/// up to the embedded copy, so an upgraded daemon does not run stale
/// scripts. Only differing (or missing) files are rewritten, each through a
/// tmp file and a rename. A changed `pyproject.toml` or `uv.lock` needs
/// `uv sync`, which this does not run: it only warns. Without `<home>/mlx/`
/// (`mlx setup` never ran) it does nothing. Failures are logged, not fatal.
pub fn refresh_scripts(home: &Path, log: &Logger) {
    let dir = dir(home);
    if !dir.is_dir() {
        return;
    }
    for (name, text) in FILES {
        let path = dir.join(name);
        if std::fs::read(&path).is_ok_and(|deployed| deployed == text.as_bytes()) {
            continue;
        }
        if !name.starts_with("naru_audio_mlx/") {
            log.line(
                "warn",
                None,
                &format!("mlx_env_stale file={name} run=`naru-audio mlx setup`"),
            );
            continue;
        }
        let tmp = path.with_extension("py.tmp");
        let written = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(&tmp, text))
            .and_then(|()| std::fs::rename(&tmp, &path));
        match written {
            Ok(()) => log.info(None, &format!("mlx_script_refreshed file={name}")),
            Err(e) => log.line(
                "warn",
                None,
                &format!("mlx_script_refresh_failed file={name} error={e}"),
            ),
        }
    }
}

/// `mlx status`: the recorded interpreter, if any, and whether it runs the
/// sidecar (it exists and imports it), or why not.
pub fn status(home: &Path) -> (Option<PathBuf>, Result<(), String>) {
    let python = match python(home) {
        Ok(p) => p,
        Err(e) => return (None, Err(e)),
    };
    if !python.is_file() {
        return (
            Some(python),
            Err("the interpreter does not exist".to_string()),
        );
    }
    let result = match Command::new(&python)
        .args(["-c", IMPORT_CHECK])
        .current_dir(dir(home))
        .output()
    {
        Ok(out) if out.status.success() => Ok(()),
        Ok(out) => Err(String::from_utf8_lossy(&out.stderr)
            .lines()
            .last()
            .unwrap_or("the import failed")
            .to_string()),
        Err(e) => Err(format!("cannot run it: {e}")),
    };
    (Some(python), result)
}

/// `config.toml` `[mlx] python`, which `mlx setup` writes.
pub fn python(home: &Path) -> Result<PathBuf, String> {
    let config = read_config(home)?;
    config
        .get("mlx")
        .and_then(|t| t.get("python"))
        .and_then(|p| p.as_str())
        .map(PathBuf::from)
        .ok_or_else(|| "not set up; run `naru-audio mlx setup`".to_string())
}

/// Whether STT model `name` runs in-process ([`parakeet`]) instead of in
/// the sidecar: any Parakeet model, unless `config.toml` sets
/// `[mlx] native_stt = false` (naru task 2380).
pub fn native_stt(home: &Path, name: &str) -> Result<bool, String> {
    if !name.starts_with("parakeet") {
        return Ok(false);
    }
    let config = read_config(home)?;
    let on = match config.get("mlx").and_then(|t| t.get("native_stt")) {
        None => true,
        Some(v) => v
            .as_bool()
            .ok_or("config.toml: `[mlx] native_stt` is not true or false")?,
    };
    Ok(on)
}

/// Whether TTS model `name` runs in-process ([`breeze`]) instead of in the
/// sidecar: the 8-bit Breeze model (the only checkpoint ported), unless
/// `config.toml` sets `[mlx] native_tts = false` (naru task 2384).
pub fn native_tts(home: &Path, name: &str) -> Result<bool, String> {
    // Tracks `breeze::check_quantization`: other Breeze checkpoints would
    // only fail the native load.
    if !(name.starts_with("breeze") && name.contains("8bit")) {
        return Ok(false);
    }
    let config = read_config(home)?;
    let on = match config.get("mlx").and_then(|t| t.get("native_tts")) {
        None => true,
        Some(v) => v
            .as_bool()
            .ok_or("config.toml: `[mlx] native_tts` is not true or false")?,
    };
    Ok(on)
}

/// Whether TTS model `name` runs in-process ([`qwen3_tts`]) instead of in
/// the sidecar: the two ported Qwen3-TTS Base checkpoints, unless
/// `config.toml` sets `[mlx] native_qwen3_tts = false` (naru task 2389).
pub fn native_qwen3_tts(home: &Path, name: &str) -> Result<bool, String> {
    if !matches!(name, "qwen3-tts-0.6b-base-mlx" | "qwen3-tts-1.7b-base-mlx") {
        return Ok(false);
    }
    let config = read_config(home)?;
    let on = match config.get("mlx").and_then(|t| t.get("native_qwen3_tts")) {
        None => true,
        Some(v) => v
            .as_bool()
            .ok_or("config.toml: `[mlx] native_qwen3_tts` is not true or false")?,
    };
    Ok(on)
}

/// Whether the native Breeze engine caches its depth decoder
/// ([`breeze::Params::depth_cache`]): `[mlx] breeze_depth_cache = true`
/// (naru task 2383; off by default, as it samples differently from the
/// sidecar).
pub fn breeze_depth_cache(home: &Path) -> Result<bool, String> {
    let config = read_config(home)?;
    let on = match config.get("mlx").and_then(|t| t.get("breeze_depth_cache")) {
        None => false,
        Some(v) => v
            .as_bool()
            .ok_or("config.toml: `[mlx] breeze_depth_cache` is not true or false")?,
    };
    Ok(on)
}

/// Sets `[mlx] python`, keeping every other key and table, and their
/// comments and layout. A symlinked `config.toml` stays a symlink: its
/// target is what is replaced.
fn write_python(home: &Path, python: &Path) -> Result<(), String> {
    let path = config_path(home);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    let mut config: toml_edit::DocumentMut = text
        .parse()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mlx = config
        .entry("mlx")
        .or_insert_with(|| toml_edit::Item::Table(toml_edit::Table::new()));
    let Some(mlx) = mlx.as_table_like_mut() else {
        return Err("config.toml: `mlx` is not a table".to_string());
    };
    mlx.insert("python", toml_edit::value(python.display().to_string()));
    let path = match std::fs::canonicalize(&path) {
        Ok(target) => target,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => path,
        Err(e) => return Err(format!("cannot resolve {}: {e}", path.display())),
    };
    let tmp = path.with_extension("toml.tmp");
    std::fs::write(&tmp, config.to_string())
        .and_then(|()| std::fs::rename(&tmp, &path))
        .map_err(|e| format!("cannot write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_stt_is_the_default_and_parakeet_only() {
        let home = tempfile::tempdir().unwrap();
        let pk = "parakeet-tdt-0.6b-v2-mlx";
        let whisper = "whisper-large-v3-turbo-mlx";
        // No config.toml, and no key: native.
        assert_eq!(native_stt(home.path(), pk), Ok(true));
        std::fs::write(config_path(home.path()), "[mlx]\npython = \"/p\"\n").unwrap();
        assert_eq!(native_stt(home.path(), pk), Ok(true));
        std::fs::write(config_path(home.path()), "[mlx]\nnative_stt = true\n").unwrap();
        assert_eq!(native_stt(home.path(), pk), Ok(true));
        // Only Parakeet has a native port.
        assert_eq!(native_stt(home.path(), whisper), Ok(false));
        // Switched off: the sidecar.
        std::fs::write(config_path(home.path()), "[mlx]\nnative_stt = false\n").unwrap();
        assert_eq!(native_stt(home.path(), pk), Ok(false));
        std::fs::write(config_path(home.path()), "[mlx]\nnative_stt = \"yes\"\n").unwrap();
        assert!(native_stt(home.path(), pk).is_err());
    }

    #[test]
    fn native_tts_is_the_default_and_8bit_breeze_only() {
        let home = tempfile::tempdir().unwrap();
        let breeze = "breeze-tts-2-8bit-mlx";
        let qwen = "qwen3-tts-12hz-0.6b-base-mlx";
        // No config.toml, and no key: native.
        assert_eq!(native_tts(home.path(), breeze), Ok(true));
        std::fs::write(config_path(home.path()), "[mlx]\npython = \"/p\"\n").unwrap();
        assert_eq!(native_tts(home.path(), breeze), Ok(true));
        std::fs::write(config_path(home.path()), "[mlx]\nnative_tts = true\n").unwrap();
        assert_eq!(native_tts(home.path(), breeze), Ok(true));
        // Only 8-bit Breeze has a native port.
        assert_eq!(native_tts(home.path(), qwen), Ok(false));
        assert_eq!(native_tts(home.path(), "breeze-tts-2-mlx"), Ok(false));
        assert_eq!(native_tts(home.path(), "breeze-tts-2-4bit-mlx"), Ok(false));
        // Switched off: the sidecar.
        std::fs::write(config_path(home.path()), "[mlx]\nnative_tts = false\n").unwrap();
        assert_eq!(native_tts(home.path(), breeze), Ok(false));
        std::fs::write(config_path(home.path()), "[mlx]\nnative_tts = \"yes\"\n").unwrap();
        assert!(native_tts(home.path(), breeze).is_err());
        // The STT switch is separate.
        std::fs::write(config_path(home.path()), "[mlx]\nnative_stt = false\n").unwrap();
        assert_eq!(native_tts(home.path(), breeze), Ok(true));
    }

    #[test]
    fn native_qwen3_tts_is_the_default_and_base_only() {
        let home = tempfile::tempdir().unwrap();
        let (small, large) = ("qwen3-tts-0.6b-base-mlx", "qwen3-tts-1.7b-base-mlx");
        // No config.toml, and no key: native.
        assert_eq!(native_qwen3_tts(home.path(), small), Ok(true));
        std::fs::write(config_path(home.path()), "[mlx]\nnative_tts = false\n").unwrap();
        assert_eq!(native_qwen3_tts(home.path(), large), Ok(true));
        std::fs::write(config_path(home.path()), "[mlx]\nnative_qwen3_tts = true\n").unwrap();
        assert_eq!(native_qwen3_tts(home.path(), small), Ok(true));
        assert_eq!(native_qwen3_tts(home.path(), large), Ok(true));
        // Only the two Base checkpoints have a native port.
        assert_eq!(native_qwen3_tts(home.path(), "qwen3-tts-0.6b-mlx"), Ok(false));
        assert_eq!(native_qwen3_tts(home.path(), "qwen3-tts-1.7b-voicedesign-mlx"), Ok(false));
        assert_eq!(native_qwen3_tts(home.path(), "breeze-tts-2-8bit-mlx"), Ok(false));
        // Switched off: the sidecar.
        std::fs::write(config_path(home.path()), "[mlx]\nnative_qwen3_tts = false\n").unwrap();
        assert_eq!(native_qwen3_tts(home.path(), small), Ok(false));
        assert_eq!(native_qwen3_tts(home.path(), large), Ok(false));
        std::fs::write(config_path(home.path()), "[mlx]\nnative_qwen3_tts = 1\n").unwrap();
        assert!(native_qwen3_tts(home.path(), small).is_err());
    }

    #[test]
    fn breeze_depth_cache_is_off_by_default() {
        let home = tempfile::tempdir().unwrap();
        assert_eq!(breeze_depth_cache(home.path()), Ok(false));
        std::fs::write(config_path(home.path()), "[mlx]\nnative_tts = true\n").unwrap();
        assert_eq!(breeze_depth_cache(home.path()), Ok(false));
        std::fs::write(
            config_path(home.path()),
            "[mlx]\nbreeze_depth_cache = true\n",
        )
        .unwrap();
        assert_eq!(breeze_depth_cache(home.path()), Ok(true));
        std::fs::write(config_path(home.path()), "[mlx]\nbreeze_depth_cache = 1\n").unwrap();
        assert!(breeze_depth_cache(home.path()).is_err());
    }

    #[test]
    fn setting_python_keeps_the_rest_of_config_toml() {
        let home = tempfile::tempdir().unwrap();
        assert!(python(home.path()).is_err());
        std::fs::write(
            config_path(home.path()),
            "[memory]\nmax_resident = \"8GiB\"\n\n[mlx]\nother = 1\n",
        )
        .unwrap();
        write_python(home.path(), Path::new("/venv/bin/python")).unwrap();
        write_python(home.path(), Path::new("/venv/bin/python")).unwrap();
        assert_eq!(
            python(home.path()).unwrap(),
            PathBuf::from("/venv/bin/python")
        );
        let config = read_config(home.path()).unwrap();
        assert_eq!(config["memory"]["max_resident"].as_str(), Some("8GiB"));
        assert_eq!(config["mlx"]["other"].as_integer(), Some(1));
    }

    #[test]
    fn refreshing_rewrites_only_stale_scripts_and_warns_on_a_stale_env() {
        let home = tempfile::tempdir().unwrap();
        let log_path = home.path().join("log");
        let log = Logger::file(&log_path, u64::MAX).unwrap();
        let dir = dir(home.path());
        std::fs::create_dir_all(dir.join("naru_audio_mlx")).unwrap();
        let embedded = |name: &str| FILES.iter().find(|(n, _)| *n == name).unwrap().1;
        std::fs::write(dir.join("pyproject.toml"), "stale").unwrap();
        std::fs::write(dir.join("uv.lock"), embedded("uv.lock")).unwrap();
        let init = dir.join("naru_audio_mlx/__init__.py");
        std::fs::write(&init, embedded("naru_audio_mlx/__init__.py")).unwrap();
        let old = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1);
        std::fs::File::options()
            .write(true)
            .open(&init)
            .unwrap()
            .set_modified(old)
            .unwrap();
        let main = dir.join("naru_audio_mlx/__main__.py");
        std::fs::write(&main, "print('stale')\n").unwrap();

        refresh_scripts(home.path(), &log);

        assert_eq!(
            std::fs::read(&main).unwrap(),
            embedded("naru_audio_mlx/__main__.py").as_bytes()
        );
        // Missing counts as stale.
        assert_eq!(
            std::fs::read_to_string(dir.join("naru_audio_mlx/protocol.py")).unwrap(),
            embedded("naru_audio_mlx/protocol.py")
        );
        assert_eq!(std::fs::metadata(&init).unwrap().modified().unwrap(), old);
        // The environment files are never rewritten, only warned about.
        assert_eq!(
            std::fs::read_to_string(dir.join("pyproject.toml")).unwrap(),
            "stale"
        );
        let logged = std::fs::read_to_string(&log_path).unwrap();
        assert!(logged.contains("mlx_env_stale file=pyproject.toml"));
        assert!(!logged.contains("file=uv.lock"));
        assert!(!logged.contains("file=naru_audio_mlx/__init__.py"));
    }

    #[test]
    fn refreshing_without_setup_creates_nothing() {
        let home = tempfile::tempdir().unwrap();
        refresh_scripts(home.path(), &Logger::stderr());
        assert!(!dir(home.path()).exists());
    }

    #[test]
    fn setting_python_keeps_comments_order_and_a_symlink() {
        let home = tempfile::tempdir().unwrap();
        let target_dir = tempfile::tempdir().unwrap();
        let target = target_dir.path().join("naru-audio.toml");
        let before = "# my settings\n[mlx]\nother = 1 # kept\n\n[defaults]\nstt = \"x\"\n";
        std::fs::write(&target, before).unwrap();
        std::os::unix::fs::symlink(&target, config_path(home.path())).unwrap();

        write_python(home.path(), Path::new("/venv/bin/python")).unwrap();
        assert!(
            std::fs::symlink_metadata(config_path(home.path()))
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "# my settings\n[mlx]\nother = 1 # kept\npython = \"/venv/bin/python\"\n\n\
             [defaults]\nstt = \"x\"\n"
        );
    }
}
