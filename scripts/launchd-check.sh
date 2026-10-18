#!/usr/bin/env bash
# launchd gate (naru task 2043, docs/launchd.md): scripts/launchd.sh under a throwaway
# HOME and NARU_LAUNCHD_DIR with stub `launchctl` and `brew` — never loads the real
# agent. Plist renders and lints (label, RunAtLoad, KeepAlive, binary, serve flags,
# log paths); install = bootout + bootstrap + kickstart, idempotent; a started brew
# service refuses the install; uninstall = bootout + plist gone; a hostile HOME is
# XML-escaped; --print/--dry-run write nothing; bad usage exits 2.
set -euo pipefail
[ "$(uname -s)" = Darwin ] || { echo "skip: launchd-check is macOS only"; exit 0; }
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
ok() { echo "ok: $*"; }
export HOME="$TMP/home" PREFIX="$TMP/prefix" NARU_LAUNCHD_DIR="$TMP/agents" NARU_LAUNCHCTL="$TMP/launchctl" NARU_BREW="$TMP/brew"
mkdir -p "$HOME"
UID_=$(id -u)
cat >"$NARU_LAUNCHCTL" <<STUB
#!/bin/sh
echo "\$*" >>"$TMP/launchctl.log"
STUB
cat >"$NARU_BREW" <<STUB
#!/bin/sh
cat "$TMP/brew.out" 2>/dev/null || true
STUB
chmod +x "$NARU_LAUNCHCTL" "$NARU_BREW"
PLIST="$NARU_LAUNCHD_DIR/com.naru.audio.plist"
log() { cat "$TMP/launchctl.log" 2>/dev/null || true; }

out=$(scripts/launchd.sh install --print)
echo "$out" | plutil -lint - >/dev/null || fail "printed plist does not lint"
[ ! -e "$PLIST" ] && [ -z "$(log)" ] || fail "--print touched something"
for want in com.naru.audio RunAtLoad KeepAlive Interactive "$PREFIX/bin/naru-audio" "<string>serve</string>" \
  "<string>--allow-remote</string>" "<string>--listen</string>" "<string>0.0.0.0:7870</string>" \
  "$HOME/.naru-audio/serve.log" "$HOME/.naru-audio/serve.err.log" "$HOME/.naru-audio/mlx/.venv/bin" "naru_audio=info"; do
  echo "$out" | grep -qF "$want" || fail "plist lacks $want"
done
ok "plist renders and lints"

scripts/launchd.sh install --dry-run >/dev/null
[ ! -e "$PLIST" ] && [ -z "$(log)" ] || fail "--dry-run changed something"
scripts/launchd.sh uninstall --dry-run >/dev/null
[ -z "$(log)" ] || fail "uninstall --dry-run changed something"
ok "--dry-run writes nothing"

scripts/launchd.sh install >/dev/null 2>&1
plutil -lint "$PLIST" >/dev/null || fail "installed plist does not lint"
[ -d "$HOME/.naru-audio" ] || fail "no ~/.naru-audio"
[ "$(log)" = "bootout gui/$UID_/com.naru.audio
bootstrap gui/$UID_ $PLIST
kickstart gui/$UID_/com.naru.audio" ] || fail "install launchctl calls: $(log)"
ok "install bootouts, bootstraps and kickstarts"

scripts/launchd.sh install >/dev/null 2>&1
[ "$(log | grep -c '^bootstrap')" = 2 ] || fail "re-install did not bootstrap again"
ok "install is idempotent"

: >"$TMP/launchctl.log"
printf 'Name        Status  User\nnaru-audio  started simonspoon\n' >"$TMP/brew.out"
rm -f "$PLIST"
if scripts/launchd.sh install >/dev/null 2>&1; then fail "install ran over a started brew service"; fi
[ ! -e "$PLIST" ] && [ -z "$(log)" ] || fail "refused install still changed something"
printf 'Name        Status  User\nnaru-audio  none\n' >"$TMP/brew.out"
scripts/launchd.sh install >/dev/null 2>&1 || fail "stopped brew service blocked install"
ok "a started brew service refuses install; a stopped one does not"

: >"$TMP/launchctl.log"
scripts/launchd.sh uninstall >/dev/null
[ ! -e "$PLIST" ] || fail "uninstall left the plist"
[ "$(log)" = "bootout gui/$UID_/com.naru.audio" ] || fail "uninstall calls: $(log)"
ok "uninstall boots out and removes the plist"

BAD="$TMP/a & b <c>"
mkdir -p "$BAD"
HOME="$BAD" scripts/launchd.sh install --print | plutil -lint - >/dev/null || fail "hostile HOME breaks the plist"
HOME="$BAD" scripts/launchd.sh install --print | grep -qF 'a &amp; b &lt;c&gt;' || fail "hostile HOME not escaped"
ok "hostile HOME is XML-escaped"

rc=0; scripts/launchd.sh bogus 2>/dev/null || rc=$?
[ "$rc" = 2 ] || fail "bad verb exit $rc, want 2"
rc=0; scripts/launchd.sh install --nope 2>/dev/null || rc=$?
[ "$rc" = 2 ] || fail "bad flag exit $rc, want 2"
ok "bad usage exits 2"
echo "all launchd checks passed"
