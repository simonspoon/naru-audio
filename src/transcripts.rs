//! Ambient transcripts on disk (naru task 2672): every final ambient segment,
//! from any client or the server's own listener, appended as one line of JSON
//! to `<home>/ambient-transcripts/<local date>.jsonl`. Kept indefinitely:
//! nothing here deletes. One `write` per line on an append-only file, so a
//! crash loses at most the line being written; a reader skips a torn line.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

pub const DIR: &str = "ambient-transcripts";

const DAY_MS: i64 = 86_400_000;

/// The end of year 9999 UTC: later times are clamped to it.
const MAX_MS: i64 = 253_402_300_800_000;

/// Appends `record` stamped with `now` (`at` local with offset, `at_utc`,
/// `unix_ms`) to the day's file, creating it (and the dir) private to the user.
pub fn append(home: &Path, now: SystemTime, mut record: Value) -> io::Result<()> {
    let ms = unix_ms(now);
    let (local, date) = local_time(ms);
    record["at"] = json!(local);
    record["at_utc"] = json!(humantime::format_rfc3339_millis(now).to_string());
    record["unix_ms"] = json!(ms);
    let dir = home.join(DIR);
    fs::create_dir_all(&dir)?;
    let mut line = serde_json::to_vec(&record).map_err(io::Error::other)?;
    line.push(b'\n');
    let mut opts = OpenOptions::new();
    opts.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        opts.mode(0o600);
        let _ = fs::set_permissions(&dir, fs::Permissions::from_mode(0o700));
    }
    let path = dir.join(format!("{date}.jsonl"));
    // A line torn by a crash gets its own line break, so this one stays whole.
    if ends_torn(&path)? {
        line.insert(0, b'\n');
    }
    let mut file = opts.open(path)?;
    file.write_all(&line)
}

fn ends_torn(path: &Path) -> io::Result<bool> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e),
    };
    if f.metadata()?.len() == 0 {
        return Ok(false);
    }
    f.seek(SeekFrom::End(-1))?;
    let mut last = [0u8];
    f.read_exact(&mut last)?;
    Ok(last[0] != b'\n')
}

/// Which segments a read returns: one instance's (`None` the legacy segments
/// stored with no instance), or every one.
pub enum Instance<'a> {
    Only(Option<&'a str>),
    All,
}

/// The segments with `from_ms <= unix_ms < to_ms` for `instance`, oldest first.
pub fn read(home: &Path, from_ms: i64, to_ms: i64, instance: Instance) -> io::Result<Vec<Value>> {
    let dir = home.join(DIR);
    let mut files: Vec<PathBuf> = match fs::read_dir(&dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
            .collect(),
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    // Files are named by local date; a day either side covers any offset.
    let (_, first) = local_time(from_ms.clamp(0, MAX_MS).saturating_sub(DAY_MS));
    let (_, last) = local_time((to_ms.saturating_add(DAY_MS)).clamp(0, MAX_MS - 1));
    files.retain(|p| {
        p.file_stem()
            .and_then(|s| s.to_str())
            .is_some_and(|d| d >= first.as_str() && d <= last.as_str())
    });
    files.sort();
    let mut out = Vec::new();
    for path in files {
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e),
        };
        // Line by line, so a line torn mid-character spoils only itself.
        for line in bytes.split(|&b| b == b'\n') {
            let Ok(v) = serde_json::from_slice::<Value>(line) else {
                continue;
            };
            let Some(ms) = v["unix_ms"].as_i64() else {
                continue;
            };
            let wanted = match instance {
                Instance::All => true,
                Instance::Only(id) => v["instance"].as_str() == id,
            };
            if wanted && ms >= from_ms && ms < to_ms {
                out.push(v);
            }
        }
    }
    out.sort_by_key(|v| v["unix_ms"].as_i64());
    Ok(out)
}

/// Unix milliseconds from `YYYY-MM-DD[THH:MM[:SS]]` (local time; a space may
/// replace the `T`), the same followed by `Z` or `±HH:MM`, or plain digits
/// (already unix milliseconds).
pub fn parse_time(s: &str) -> Option<i64> {
    let s = s.trim();
    if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) {
        return s.parse::<i64>().ok().map(|ms| ms.min(MAX_MS));
    }
    let (body, offset_min) = if let Some(body) = s.strip_suffix('Z') {
        (body, Some(0))
    } else if s.is_ascii() && s.len() > 16 && matches!(s.as_bytes()[s.len() - 6], b'+' | b'-') {
        let (body, off) = s.split_at(s.len() - 6);
        let sign = if off.starts_with('-') { -1 } else { 1 };
        let b = off.as_bytes();
        if b[3] != b':' || ![1, 2, 4, 5].iter().all(|&i| b[i].is_ascii_digit()) {
            return None;
        }
        let h: i64 = off[1..3].parse().ok()?;
        let m: i64 = off[4..6].parse().ok()?;
        if h > 23 || m > 59 {
            return None;
        }
        (body, Some(sign * (h * 60 + m)))
    } else {
        (s, None)
    };
    let (date, clock) = match body.split_once(['T', ' ']) {
        Some((d, c)) => (d, c),
        None => (body, "00:00"),
    };
    let mut d = date.split('-').map(|p| p.parse::<i64>().ok());
    let (y, mo, da) = (d.next()??, d.next()??, d.next()??);
    // Seconds may carry a fraction, as the stored `at` does.
    let (clock, frac) = clock.split_once('.').unwrap_or((clock, ""));
    let frac_ms = match frac {
        "" => 0,
        f if f.len() <= 3 && f.bytes().all(|b| b.is_ascii_digit()) => {
            format!("{f:0<3}").parse::<i64>().ok()?
        }
        _ => return None,
    };
    let mut c = clock.split(':').map(|p| p.parse::<i64>().ok());
    let (h, mi) = (c.next()??, c.next()??);
    let sec = c.next().unwrap_or(Some(0))?;
    if d.next().is_some()
        || c.next().is_some()
        || !(1970..=9999).contains(&y)
        || !(1..=12).contains(&mo)
        || !(1..=31).contains(&da)
        || !(0..24).contains(&h)
        || !(0..60).contains(&mi)
        || !(0..61).contains(&sec)
    {
        return None;
    }
    let naive = (days_from_civil(y, mo, da) * 86_400 + h * 3600 + mi * 60 + sec) * 1000 + frac_ms;
    match offset_min {
        Some(off) => Some(naive - off * 60_000),
        // Local: the offset in force at that moment (a second pass settles
        // a guess made across a DST change).
        None => {
            let guess = naive - local_offset_s(naive) * 1000;
            Some(naive - local_offset_s(guess) * 1000)
        }
    }
}

fn unix_ms(t: SystemTime) -> i64 {
    t.duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis() as i64
}

/// `ms` as local RFC 3339 with its offset, and its local `YYYY-MM-DD`.
fn local_time(ms: i64) -> (String, String) {
    let off = local_offset_s(ms);
    let local_s = ms.div_euclid(1000) + off;
    let (y, mo, d) = civil_from_days(local_s.div_euclid(86_400));
    let sod = local_s.rem_euclid(86_400);
    let sign = if off < 0 { '-' } else { '+' };
    let date = format!("{y:04}-{mo:02}-{d:02}");
    let at = format!(
        "{date}T{:02}:{:02}:{:02}.{:03}{sign}{:02}:{:02}",
        sod / 3600,
        sod % 3600 / 60,
        sod % 60,
        ms.rem_euclid(1000),
        off.abs() / 3600,
        off.abs() % 3600 / 60,
    );
    (at, date)
}

/// The local UTC offset in seconds at unix time `ms` (0 where unknown).
#[cfg(unix)]
fn local_offset_s(ms: i64) -> i64 {
    let t = ms.div_euclid(1000) as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: both pointers are valid for the call; localtime_r is reentrant.
    if unsafe { libc::localtime_r(&t, &mut tm) }.is_null() {
        return 0;
    }
    tm.tm_gmtoff as i64
}

#[cfg(not(unix))]
fn local_offset_s(_ms: i64) -> i64 {
    0
}

/// Days since 1970-01-01 of a proleptic Gregorian date (Howard Hinnant).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (
        if m <= 2 {
            yoe + era * 400 + 1
        } else {
            yoe + era * 400
        },
        m,
        d,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(ms: i64) -> SystemTime {
        UNIX_EPOCH + Duration::from_millis(ms as u64)
    }

    #[test]
    fn appends_and_reads_a_range_by_instance() {
        let home = tempfile::tempdir().unwrap();
        let t0 = parse_time("2026-11-09T15:00:00Z").unwrap();
        append(home.path(), at(t0), json!({"text": "a", "instance": "i1"})).unwrap();
        append(
            home.path(),
            at(t0 + 60_000),
            json!({"text": "b", "instance": "i2"}),
        )
        .unwrap();
        append(
            home.path(),
            at(t0 + 120_000),
            json!({"text": "c", "instance": null}),
        )
        .unwrap();
        append(
            home.path(),
            at(t0 + DAY_MS),
            json!({"text": "d", "instance": "i1"}),
        )
        .unwrap();

        let texts = |inst| {
            read(home.path(), t0, t0 + 2 * DAY_MS, inst)
                .unwrap()
                .iter()
                .map(|v| v["text"].as_str().unwrap().to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(texts(Instance::Only(Some("i1"))), ["a", "d"]);
        assert_eq!(texts(Instance::Only(Some("i2"))), ["b"]);
        assert_eq!(texts(Instance::Only(None)), ["c"]);
        assert_eq!(texts(Instance::All), ["a", "b", "c", "d"]);
        // `to` is exclusive.
        let one = read(home.path(), t0, t0 + 60_000, Instance::All).unwrap();
        assert_eq!(one.len(), 1);
        let v = &one[0];
        assert_eq!(v["unix_ms"], t0);
        assert_eq!(v["at_utc"], "2026-11-09T15:00:00.000Z");
        assert_eq!(parse_time(v["at"].as_str().unwrap()), Some(t0));
        // One file per local date, at least two days here.
        assert!(fs::read_dir(home.path().join(DIR)).unwrap().count() >= 2);
    }

    #[test]
    fn a_torn_line_is_skipped() {
        let home = tempfile::tempdir().unwrap();
        let t0 = parse_time("2026-11-09T15:00:00Z").unwrap();
        append(home.path(), at(t0), json!({"text": "a", "instance": null})).unwrap();
        let file = fs::read_dir(home.path().join(DIR))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let mut f = OpenOptions::new().append(true).open(&file).unwrap();
        f.write_all(b"{\"text\":\"torn").unwrap();
        drop(f);
        append(
            home.path(),
            at(t0 + 1),
            json!({"text": "b", "instance": null}),
        )
        .unwrap();
        let got = read(home.path(), t0, t0 + 10, Instance::All).unwrap();
        let texts: Vec<_> = got.iter().map(|v| v["text"].as_str().unwrap()).collect();
        assert_eq!(texts, ["a", "b"]);
    }

    #[test]
    fn a_line_torn_mid_character_spoils_only_itself() {
        let home = tempfile::tempdir().unwrap();
        let t0 = parse_time("2026-11-09T15:00:00Z").unwrap();
        append(home.path(), at(t0), json!({"text": "a"})).unwrap();
        let file = fs::read_dir(home.path().join(DIR))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let mut f = OpenOptions::new().append(true).open(&file).unwrap();
        f.write_all(&"{\"text\":\"日本".as_bytes()[..12]).unwrap();
        drop(f);
        append(home.path(), at(t0 + 1), json!({"text": "b"})).unwrap();
        let got = read(home.path(), t0, t0 + 10, Instance::All).unwrap();
        assert_eq!(got.len(), 2);
        // Huge bounds neither panic nor drop everything.
        let all = read(home.path(), 0, i64::MAX, Instance::All).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn files_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let home = tempfile::tempdir().unwrap();
        append(home.path(), SystemTime::now(), json!({"text": "a"})).unwrap();
        let file = fs::read_dir(home.path().join(DIR))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        assert_eq!(
            fs::metadata(file).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[test]
    fn parses_times() {
        assert_eq!(parse_time("1700000000000"), Some(1_700_000_000_000));
        assert_eq!(parse_time("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_time("1970-01-01T01:00:00+01:00"), Some(0));
        assert_eq!(
            parse_time("2026-11-09 10:00Z"),
            parse_time("2026-11-09T10:00:00Z")
        );
        assert_eq!(
            parse_time("2026-11-09T10:00:00-05:00"),
            parse_time("2026-11-09T15:00:00Z")
        );
        // Local time round-trips through the stored `at`.
        let ms = parse_time("2026-11-09T10:00").unwrap();
        assert!(local_time(ms).0.starts_with("2026-11-09T10:00:00.000"));
        assert_eq!(parse_time("2026-13-01T00:00Z"), None);
        assert_eq!(parse_time("1970-01-01T00:00:01.5Z"), Some(1500));
        assert_eq!(parse_time("yesterday"), None);
        assert_eq!(parse_time("9223372036854775807"), Some(MAX_MS));
        assert_eq!(parse_time("99999999999999999-01-01T00:00Z"), None);
        assert_eq!(parse_time("2026-11-09T25:00Z"), None);
        assert_eq!(parse_time("2026-11-09T10:00:00+-5:00"), None);
        assert_eq!(parse_time("2026-11-09T10:00:00±05:00"), None);
    }
}
