//! §3 `/admin`: the static, buildless admin UI, embedded at compile time
//! from `src/admin/` (same `include_bytes!` pattern as the MLX sidecar
//! files, `src/mlx.rs:16-28`). `GET /admin` and `/admin/` serve
//! `index.html`; `GET /admin/{*path}` serves any other asset by exact path
//! match, or a JSON 404 envelope for anything not in the table.

use axum::extract::Path;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};

use crate::error::ApiError;

struct Asset {
    path: &'static str,
    content_type: &'static str,
    bytes: &'static [u8],
}

/// Every file under `src/admin/`, with the path it is served at under
/// `/admin/` and its `Content-Type`. Adding a file here is the only step
/// needed to serve it — there is no build step or directory scan.
const ASSETS: &[Asset] = &[
    Asset {
        path: "admin.css",
        content_type: "text/css; charset=utf-8",
        bytes: include_bytes!("../admin/admin.css"),
    },
    Asset {
        path: "fonts/OFL.txt",
        content_type: "text/plain; charset=utf-8",
        bytes: include_bytes!("../admin/fonts/OFL.txt"),
    },
    Asset {
        path: "fonts/inter-latin-400-normal.woff2",
        content_type: "font/woff2",
        bytes: include_bytes!("../admin/fonts/inter-latin-400-normal.woff2"),
    },
    Asset {
        path: "fonts/inter-latin-500-normal.woff2",
        content_type: "font/woff2",
        bytes: include_bytes!("../admin/fonts/inter-latin-500-normal.woff2"),
    },
    Asset {
        path: "fonts/inter-latin-600-normal.woff2",
        content_type: "font/woff2",
        bytes: include_bytes!("../admin/fonts/inter-latin-600-normal.woff2"),
    },
    Asset {
        path: "fonts/inter-latin-700-normal.woff2",
        content_type: "font/woff2",
        bytes: include_bytes!("../admin/fonts/inter-latin-700-normal.woff2"),
    },
    Asset {
        path: "fonts/orbitron-latin-400-normal.woff2",
        content_type: "font/woff2",
        bytes: include_bytes!("../admin/fonts/orbitron-latin-400-normal.woff2"),
    },
    Asset {
        path: "fonts/orbitron-latin-700-normal.woff2",
        content_type: "font/woff2",
        bytes: include_bytes!("../admin/fonts/orbitron-latin-700-normal.woff2"),
    },
    Asset {
        path: "fonts/share-tech-mono-latin-400-normal.woff2",
        content_type: "font/woff2",
        bytes: include_bytes!("../admin/fonts/share-tech-mono-latin-400-normal.woff2"),
    },
    Asset {
        path: "js/api.mjs",
        content_type: "text/javascript; charset=utf-8",
        bytes: include_bytes!("../admin/js/api.mjs"),
    },
    Asset {
        path: "js/ui.mjs",
        content_type: "text/javascript; charset=utf-8",
        bytes: include_bytes!("../admin/js/ui.mjs"),
    },
    Asset {
        path: "js/app.mjs",
        content_type: "text/javascript; charset=utf-8",
        bytes: include_bytes!("../admin/js/app.mjs"),
    },
    Asset {
        path: "js/overview.mjs",
        content_type: "text/javascript; charset=utf-8",
        bytes: include_bytes!("../admin/js/overview.mjs"),
    },
    Asset {
        path: "js/models.mjs",
        content_type: "text/javascript; charset=utf-8",
        bytes: include_bytes!("../admin/js/models.mjs"),
    },
    Asset {
        path: "js/voices.mjs",
        content_type: "text/javascript; charset=utf-8",
        bytes: include_bytes!("../admin/js/voices.mjs"),
    },
    Asset {
        path: "js/clone.mjs",
        content_type: "text/javascript; charset=utf-8",
        bytes: include_bytes!("../admin/js/clone.mjs"),
    },
    Asset {
        path: "js/design.mjs",
        content_type: "text/javascript; charset=utf-8",
        bytes: include_bytes!("../admin/js/design.mjs"),
    },
    Asset {
        path: "js/playground.mjs",
        content_type: "text/javascript; charset=utf-8",
        bytes: include_bytes!("../admin/js/playground.mjs"),
    },
    Asset {
        path: "js/pcm-worklet.mjs",
        content_type: "text/javascript; charset=utf-8",
        bytes: include_bytes!("../admin/js/pcm-worklet.mjs"),
    },
    Asset {
        path: "js/samples.mjs",
        content_type: "text/javascript; charset=utf-8",
        bytes: include_bytes!("../admin/js/samples.mjs"),
    },
    Asset {
        path: "js/studio.mjs",
        content_type: "text/javascript; charset=utf-8",
        bytes: include_bytes!("../admin/js/studio.mjs"),
    },
    Asset {
        path: "js/studio-edit.mjs",
        content_type: "text/javascript; charset=utf-8",
        bytes: include_bytes!("../admin/js/studio-edit.mjs"),
    },
    Asset {
        path: "js/studio-wave.mjs",
        content_type: "text/javascript; charset=utf-8",
        bytes: include_bytes!("../admin/js/studio-wave.mjs"),
    },
    Asset {
        path: "js/studio-clip.mjs",
        content_type: "text/javascript; charset=utf-8",
        bytes: include_bytes!("../admin/js/studio-clip.mjs"),
    },
];

const INDEX: &[u8] = include_bytes!("../admin/index.html");
const INDEX_CONTENT_TYPE: &str = "text/html; charset=utf-8";

/// `GET /admin`, `/admin/`.
pub async fn index() -> Response {
    asset_response(INDEX_CONTENT_TYPE, INDEX)
}

/// `GET /admin/{*path}`: an embedded asset by exact path, or a JSON 404
/// envelope (the same shape every other API error uses) for anything not
/// in [`ASSETS`].
pub async fn asset(Path(path): Path<String>) -> Response {
    match ASSETS.iter().find(|a| a.path == path) {
        Some(a) => asset_response(a.content_type, a.bytes),
        None => ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            format!("no admin asset at \"{path}\""),
        )
        .into_response(),
    }
}

/// Shared response headers for `index()` and `asset()`: no caching (this
/// binary, not a CDN, is the only place these files come from, so a stale
/// cache buys nothing and only confuses a `naru-audio` upgrade), no MIME
/// sniffing, and a CSP scoped to what the admin UI actually does — no
/// build step, no third-party origins, audio playback from `blob:` object
/// URLs, the plain WebSocket the streaming STT tab will use, and
/// `style-src 'unsafe-inline'` because every `.mjs` module styles elements
/// with a `style=""` attribute built from app data (dot colors, waveform
/// bars, layout) rather than a stylesheet class per state; none of it is
/// user-controlled text, so it carries none of the injection risk
/// `unsafe-inline` script would (`script-src` stays plain `'self'`).
fn asset_response(content_type: &'static str, bytes: &'static [u8]) -> Response {
    let mut resp = bytes.into_response();
    let headers = resp.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(
            "default-src 'self'; style-src 'self' 'unsafe-inline'; media-src 'self' blob:; connect-src 'self'",
        ),
    );
    resp
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::time::Instant;

    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use crate::log::Logger;
    use crate::manager::{BackendLoader, ModelManager, Settings};
    use crate::profile::Profile;
    use crate::registry::Registry;
    use crate::server::{AppState, PullTracker, router};

    fn app() -> axum::Router {
        let log = Arc::new(Logger::stderr());
        let registry = Arc::new(Registry::open("/nonexistent/naru-audio-home").unwrap());
        let settings = Settings::from_env(Profile::detect().unwrap()).unwrap();
        let models = ModelManager::new(
            registry.clone(),
            log.clone(),
            settings,
            Arc::new(BackendLoader),
        );
        router(Arc::new(AppState {
            port: 7870,
            allow_remote: false,
            started: Instant::now(),
            log,
            registry,
            models: Arc::new(models),
            pulls: Arc::new(PullTracker::new()),
        }))
    }

    #[tokio::test]
    async fn index_serves_html_at_admin_and_admin_slash() {
        for path in ["/admin", "/admin/"] {
            let req = Request::get(path)
                .header(axum::http::header::HOST, "127.0.0.1:7870")
                .body(Body::empty())
                .unwrap();
            let resp = app().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "{path}");
            assert_eq!(
                resp.headers().get(header::CONTENT_TYPE).unwrap(),
                INDEX_CONTENT_TYPE
            );
        }
    }

    #[tokio::test]
    async fn every_asset_is_200_with_its_content_type() {
        for a in ASSETS {
            let req = Request::get(format!("/admin/{}", a.path))
                .header(axum::http::header::HOST, "127.0.0.1:7870")
                .body(Body::empty())
                .unwrap();
            let resp = app().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "{}", a.path);
            assert_eq!(
                resp.headers().get(header::CONTENT_TYPE).unwrap(),
                a.content_type,
                "{}",
                a.path
            );
            let body = resp.into_body().collect().await.unwrap().to_bytes();
            assert_eq!(&body[..], a.bytes, "{}", a.path);
        }
    }

    #[tokio::test]
    async fn unknown_admin_path_is_404_with_envelope() {
        let req = Request::get("/admin/nope")
            .header(axum::http::header::HOST, "127.0.0.1:7870")
            .body(Body::empty())
            .unwrap();
        let resp = app().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "not_found");
    }

    #[tokio::test]
    async fn asset_response_headers_include_csp_and_nosniff() {
        let req = Request::get("/admin")
            .header(axum::http::header::HOST, "127.0.0.1:7870")
            .body(Body::empty())
            .unwrap();
        let resp = app().oneshot(req).await.unwrap();
        assert_eq!(
            resp.headers().get(header::CACHE_CONTROL).unwrap(),
            "no-cache"
        );
        assert_eq!(
            resp.headers().get(header::X_CONTENT_TYPE_OPTIONS).unwrap(),
            "nosniff"
        );
        assert!(
            resp.headers()
                .get(header::CONTENT_SECURITY_POLICY)
                .unwrap()
                .to_str()
                .unwrap()
                .contains("default-src 'self'")
        );
    }

    #[tokio::test]
    async fn csp_connect_src_is_self_only() {
        let req = Request::get("/admin")
            .header(axum::http::header::HOST, "127.0.0.1:7870")
            .body(Body::empty())
            .unwrap();
        let resp = app().oneshot(req).await.unwrap();
        let csp = resp
            .headers()
            .get(header::CONTENT_SECURITY_POLICY)
            .unwrap()
            .to_str()
            .unwrap();
        let connect = csp
            .split(';')
            .map(str::trim)
            .find(|d| d.starts_with("connect-src"))
            .unwrap();
        assert_eq!(connect, "connect-src 'self'");
        assert!(!csp.split_whitespace().any(|t| t == "ws:" || t == "wss:"));
    }

    /// Every `src=`, `href=` and `import ... from` in an HTML/JS/CSS asset
    /// must resolve to another entry in [`ASSETS`] (or `index.html`
    /// itself) — nothing links to a file the router can't serve.
    #[test]
    fn every_reference_resolves() {
        let known: HashSet<&str> = ASSETS.iter().map(|a| a.path).collect();
        let mut all = Vec::new();
        all.push(("index.html", INDEX));
        all.extend(ASSETS.iter().map(|a| (a.path, a.bytes)));
        for (path, bytes) in &all {
            // Fonts and the license text are binary/leaf files: nothing
            // references anything out of them.
            if path.starts_with("fonts/") {
                continue;
            }
            let text = std::str::from_utf8(bytes).unwrap_or_else(|_| panic!("{path} not utf8"));
            for reference in references(text) {
                if reference.starts_with("http") || reference.starts_with('#') {
                    continue;
                }
                let resolved = resolve(path, reference);
                assert!(
                    known.contains(resolved.as_str()) || resolved == "index.html",
                    "{path} references {reference:?} (resolved {resolved:?}), which is not in ASSETS"
                );
            }
        }
    }

    /// Crude but sufficient extraction of `src="..."`, `href="..."` and
    /// `from '...'`/`import '...'` references from HTML/CSS/JS text.
    fn references(text: &str) -> Vec<&str> {
        let mut out = Vec::new();
        for pat in ["src=\"", "href=\"", "from '", "import '", "url('"] {
            let mut rest = text;
            while let Some(i) = rest.find(pat) {
                let after = &rest[i + pat.len()..];
                let end_char = pat.chars().last().unwrap();
                let close = if end_char == '\'' { '\'' } else { '"' };
                if let Some(j) = after.find(close) {
                    out.push(&after[..j]);
                    rest = &after[j..];
                } else {
                    break;
                }
            }
        }
        out
    }

    fn resolve(from: &str, reference: &str) -> String {
        // `index.html` references assets by absolute `/admin/...` path (so
        // they resolve the same whether the page was reached as `/admin`
        // or `/admin/` — see the `same_origin_get_is_allowed`-style
        // trailing-slash gotcha this fixed); strip that prefix to compare
        // against `ASSETS`' own relative paths. Everything else (a `.mjs`
        // importing a sibling module) is relative to its own path.
        if let Some(rest) = reference.strip_prefix("/admin/") {
            return normalize(rest);
        }
        if let Some(dir) = from.rfind('/') {
            let base = &from[..dir];
            normalize(&format!("{base}/{reference}"))
        } else {
            normalize(reference)
        }
    }

    fn normalize(path: &str) -> String {
        let mut parts: Vec<&str> = Vec::new();
        for seg in path.split('/') {
            match seg {
                "." | "" => {}
                ".." => {
                    parts.pop();
                }
                seg => parts.push(seg),
            }
        }
        parts.join("/")
    }

    #[test]
    fn every_asset_has_a_non_empty_body() {
        assert!(!INDEX.is_empty());
        for a in ASSETS {
            assert!(!a.bytes.is_empty(), "{} is empty", a.path);
        }
    }

    #[test]
    fn asset_paths_are_unique() {
        let mut seen = HashSet::new();
        for a in ASSETS {
            assert!(seen.insert(a.path), "duplicate asset path {}", a.path);
        }
    }
}
