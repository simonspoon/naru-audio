//! mlx_audio.resample for downsampling (naru task 2369, bit-exact): scipy `resample_poly(x, up, down,
//! window=fir, padtype="edge")` with the kaiser_best FIR from
//! `_polyphase_filter` (`firwin(2*64*max_rate+1, rolloff/max_rate,
//! window=("kaiser", beta))`). Computed in f64 like scipy, then cast to f32.
//! Python's `resample_audio_chunks` keeps whole-buffer output exactly, so
//! this does the whole buffer.

// cephes's tables, digit for digit.
#![allow(clippy::excessive_precision)]

const I0A: [f64; 30] = [
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1,
];
const I0B: [f64; 25] = [
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1,
];

fn chbevl(x: f64, p: &[f64]) -> f64 {
    let (mut b0, mut b1, mut b2) = (p[0], 0.0, 0.0);
    for &v in &p[1..] {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + v;
    }
    0.5 * (b0 - b2)
}

/// cephes i0 (scipy.special.i0).
fn i0(x: f64) -> f64 {
    let x = x.abs();
    if x <= 8.0 {
        x.exp() * chbevl(x / 2.0 - 2.0, &I0A)
    } else {
        x.exp() * chbevl(32.0 / x - 2.0, &I0B) / x.sqrt()
    }
}

/// numpy's pairwise summation (float64 `np.sum` of a contiguous array).
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0; // numpy starts from -0.0, the same for any nonzero sum
        for &v in a {
            r += v;
        }
        r
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// scipy `firwin(numtaps, cutoff, window=("kaiser", beta))`, lowpass, scaled.
fn firwin_kaiser(numtaps: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let alpha = 0.5 * (numtaps as f64 - 1.0);
    let sinc = |x: f64| {
        let y = std::f64::consts::PI * if x == 0.0 { 1.0e-20 } else { x };
        y.sin() / y
    };
    let walpha = (numtaps as f64 - 1.0) / 2.0;
    let ib = i0(beta);
    let mut h: Vec<f64> = (0..numtaps)
        .map(|n| {
            let m = n as f64 - alpha;
            // h = 0; h += right * sinc(right * m); h -= 0 * sinc(0 * m)
            let mut v = 0.0 + cutoff * sinc(cutoff * m);
            v -= 0.0 * sinc(0.0 * m);
            let r = (n as f64 - walpha) / walpha;
            v * (i0(beta * (1.0 - r.powf(2.0)).sqrt()) / ib)
        })
        .collect();
    // scale_frequency 0: c = cos(0) = 1, s = sum(h * c)
    let s = pairwise_sum(&h.iter().map(|v| v * 1.0).collect::<Vec<_>>());
    for v in h.iter_mut() {
        *v /= s;
    }
    h
}

fn output_len(len_h: usize, in_len: usize, up: usize, down: usize) -> usize {
    ((in_len - 1) * up + len_h - 1) / down + 1
}

/// scipy `upfirdn(h, x, up, down, mode="edge")` (`_apply_impl`).
fn upfirdn_edge(h: &[f64], x: &[f64], up: usize, down: usize) -> Vec<f64> {
    let len_h = h.len();
    let h_padlen = len_h + (up - len_h % up) % up;
    let mut h_full = vec![0.0; h_padlen];
    h_full[..len_h].copy_from_slice(h);
    // reshape(-1, up).T[:, ::-1].ravel()
    let rows = h_padlen / up;
    let mut htf = Vec::with_capacity(h_padlen);
    for ph in 0..up {
        for r in (0..rows).rev() {
            htf.push(h_full[r * up + ph]);
        }
    }
    let hpp = rows as i64;
    let len_x = x.len() as i64;
    let len_out = output_len(len_h, x.len(), up, down);
    let mut out = vec![0.0f64; len_out];
    let xval = |i: i64| -> f64 {
        if i < 0 {
            x[0]
        } else if i >= len_x {
            x[(len_x - 1) as usize]
        } else {
            x[i as usize]
        }
    };
    let (mut x_idx, mut y_idx, mut t) = (0i64, 0usize, 0i64);
    while x_idx < len_x && y_idx < len_out {
        let h0 = (t * hpp) as usize;
        for (h_idx, xc) in (h0..).zip((x_idx - hpp + 1)..=x_idx) {
            out[y_idx] += xval(xc) * htf[h_idx];
        }
        y_idx += 1;
        t += down as i64;
        x_idx += t / up as i64;
        t %= up as i64;
    }
    while y_idx < len_out {
        let h0 = (t * hpp) as usize;
        for (h_idx, xc) in (h0..).zip((x_idx - hpp + 1)..=x_idx) {
            out[y_idx] += xval(xc) * htf[h_idx];
        }
        y_idx += 1;
        t += down as i64;
        x_idx += t / up as i64;
        t %= up as i64;
    }
    out
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// `x` (not empty) from `orig` Hz down to `target` Hz.
pub fn resample_poly(x: &[f32], orig: usize, target: usize) -> Vec<f32> {
    let g = gcd(orig, target);
    let (up, down) = (target / g, orig / g);
    let max_rate = up.max(down);
    let (num_zeros, rolloff, beta) = (64usize, 0.9475937167399596f64, 14.769656459379492f64);
    let fir = firwin_kaiser(
        2 * num_zeros * max_rate + 1,
        rolloff / max_rate as f64,
        beta,
    );
    let n_in = x.len();
    let n_out = (n_in * up) / down + usize::from(!(n_in * up).is_multiple_of(down));
    let half_len = (fir.len() - 1) / 2;
    let mut h: Vec<f64> = fir.iter().map(|v| v * up as f64).collect();
    let n_pre_pad = down - half_len % down;
    let mut n_post_pad = 0;
    let n_pre_remove = (half_len + n_pre_pad) / down;
    while output_len(h.len() + n_pre_pad + n_post_pad, n_in, up, down) < n_out + n_pre_remove {
        n_post_pad += 1;
    }
    let mut hp = vec![0.0; n_pre_pad];
    hp.append(&mut h);
    hp.extend(std::iter::repeat_n(0.0, n_post_pad));
    let xd: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let y = upfirdn_edge(&hp, &xd, up, down);
    y[n_pre_remove..n_pre_remove + n_out]
        .iter()
        .map(|&v| v as f32)
        .collect()
}
