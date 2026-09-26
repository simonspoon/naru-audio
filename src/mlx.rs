//! §5.3 the MLX sidecar on Apple Silicon: `naru-audio mlx setup|status`
//! (§3.7), the `[mlx] python` key of `config.toml` (§3.1), and the
//! supervised process itself ([`sidecar`]).

pub mod sidecar;

use std::path::{Path, PathBuf};
use std::process::Command;

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
const IMPORT_CHECK: &str = "import naru_audio_mlx, mlx.core, parakeet_mlx, mlx_audio.tts.utils";

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
