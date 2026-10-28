#!/usr/bin/env bash
# Run `naru-audio serve` as a launchd user agent (naru task 2043, docs/launchd.md).
#
#   scripts/launchd.sh install [--dry-run|--print]
#   scripts/launchd.sh uninstall [--dry-run]
#
# install renders scripts/launchd/com.naru.audio.plist.in to
# ~/Library/LaunchAgents/com.naru.audio.plist (runs `naru-audio serve --allow-remote
# --listen 0.0.0.0:7870`; logs ~/.naru-audio/serve.log + serve.err.log), then bootstraps
# and kickstarts it. Opt-in: nothing loads it automatically. Quit any hand-run
# `naru-audio` first (port 7870). Refuses while the brew service is started.
# Binary: $PREFIX/bin/naru-audio, else the first of ~/.cargo/bin, ~/.local/bin,
# /opt/homebrew/bin that has one (default ~/.cargo/bin).
# --print writes the plist to stdout and touches nothing; --dry-run prints the plan.
# Env seams: NARU_LAUNCHD_DIR (plist dir), NARU_LAUNCHCTL, NARU_BREW (binaries).
# Exit 2 off macOS or on bad usage.
set -euo pipefail

shopt -u patsub_replacement 2>/dev/null || true  # a literal & in a substituted value

[ "$(uname -s)" = Darwin ] || { echo "launchd.sh: macOS only" >&2; exit 2; }
verb="${1:-}"; mode="${2:-}"
case "$verb:$mode" in
  install:|install:--dry-run|install:--print|uninstall:|uninstall:--dry-run) ;;
  *) echo "usage: launchd.sh install [--dry-run|--print] | uninstall [--dry-run]" >&2; exit 2 ;;
esac

here="$(cd "$(dirname "$0")" && pwd)"
LABEL=com.naru.audio
DIR="${NARU_LAUNCHD_DIR:-$HOME/Library/LaunchAgents}"
PLIST="$DIR/$LABEL.plist"
LAUNCHCTL="${NARU_LAUNCHCTL:-launchctl}"
BREW="${NARU_BREW:-brew}"
DOMAIN="gui/$(id -u)"

if [ -n "${PREFIX:-}" ]; then
  BIN="$PREFIX/bin/naru-audio"
else
  BIN="$HOME/.cargo/bin/naru-audio"
  for d in "$HOME/.cargo/bin" "$HOME/.local/bin" /opt/homebrew/bin; do
    [ -x "$d/naru-audio" ] && { BIN="$d/naru-audio"; break; }
  done
fi

xml() { printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'; }

render() {
  local path="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
  local out; out="$(cat "$here/launchd/$LABEL.plist.in")"
  out="${out//__BIN__/$(xml "$BIN")}"
  out="${out//__HOME__/$(xml "$HOME")}"
  out="${out//__PATH__/$(xml "$path")}"
  printf '%s\n' "$out"
}

if [ "$mode" = --print ]; then render; exit 0; fi

if [ "$verb" = install ]; then
  if [ "$mode" = --dry-run ]; then
    echo "would write $PLIST (bin $BIN), bootstrap $DOMAIN, kickstart $DOMAIN/$LABEL"; exit 0
  fi
  # The brew formula also binds 7870: never run both.
  if command -v "$BREW" >/dev/null 2>&1 \
     && "$BREW" services list 2>/dev/null | grep -E '^naru-audio[[:space:]]+started' >/dev/null; then
    echo "launchd.sh: the brew service naru-audio is started; run 'brew services stop naru-audio' first (both bind 7870)" >&2
    exit 1
  fi
  mkdir -p "$DIR" "$HOME/.naru-audio"
  render >"$PLIST.tmp"
  plutil -lint "$PLIST.tmp" >/dev/null || { rm -f "$PLIST.tmp"; echo "launchd.sh: rendered plist is invalid" >&2; exit 1; }
  mv -f "$PLIST.tmp" "$PLIST"
  # Idempotent: drop a loaded copy first (absent is fine), then load the new one.
  "$LAUNCHCTL" bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
  "$LAUNCHCTL" bootstrap "$DOMAIN" "$PLIST"
  "$LAUNCHCTL" kickstart "$DOMAIN/$LABEL"
  echo "ok: installed $PLIST; naru-audio now runs under launchd (logs ~/.naru-audio/serve.log)"
  echo "note: a hand-run 'naru-audio' on port 7870 must be quit first or the two collide" >&2
else
  if [ "$mode" = --dry-run ]; then echo "would bootout $DOMAIN/$LABEL and remove $PLIST"; exit 0; fi
  "$LAUNCHCTL" bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
  rm -f "$PLIST"
  echo "ok: uninstalled $LABEL"
fi
