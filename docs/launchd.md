# Running `naru-audio serve` under launchd

macOS only, opt-in (naru task 2043). A hand-run `naru-audio` cannot be restarted or kept
alive; the `com.naru.audio` user agent can. Nothing loads it automatically.

- `scripts/launchd.sh install` renders `scripts/launchd/com.naru.audio.plist.in` to
  `~/Library/LaunchAgents/com.naru.audio.plist`, lints it, then `launchctl bootout`
  (ignored if absent), `bootstrap gui/<uid>` and `kickstart`. Re-running is idempotent.
- It runs `naru-audio serve --allow-remote --listen 0.0.0.0:7870` (the manual run's flags),
  `RunAtLoad` + `KeepAlive`, `ProcessType` Interactive. Binary: `$PREFIX/bin/naru-audio`, else
  the first of `~/.cargo/bin`, `~/.local/bin`, `/opt/homebrew/bin`.
- Logs: `~/.naru-audio/serve.log` and `serve.err.log`.
- Env: launchd starts bare, so the plist sets `HOME`, `RUST_LOG=naru_audio=info` and a `PATH`
  with `~/.naru-audio/mlx/.venv/bin` first (the MLX sidecar), then cargo, local, Homebrew and system dirs.
- `install --print` shows the plist; `--dry-run` shows the plan. `uninstall` boots the agent out
  and removes the plist.
- **Security:** `--allow-remote` on `0.0.0.0` is unauthenticated, exactly like the manual run.

## Brew coexistence

The brew formula (`homebrew.mxcl.naru-audio`) is a different label but binds the same port 7870.
Use brew **or** launchd on a machine, never both. `install` refuses while
`brew services list` shows naru-audio started; run `brew services stop naru-audio` first.
Other platforms and the formula are unchanged.

## One manual cutover

1. Quit the hand-run `naru-audio` (two servers on 7870 collide).
2. `scripts/launchd.sh install`.
3. Check `curl -s http://127.0.0.1:7870/health` and `tail ~/.naru-audio/serve.log`.

Rollback: `scripts/launchd.sh uninstall`, then start it by hand again.

Gate: `scripts/launchd-check.sh` (stub `launchctl` and `brew`, never touches the real agent).
Loop smoke test passed.
