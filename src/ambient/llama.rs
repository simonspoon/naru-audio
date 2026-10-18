//! llama.cpp inference for the Intern-Decision-4B gate (cargo feature `llama`).
//!
//! The model loads once, on a dedicated worker thread that owns the backend, model and context, so
//! the async server never blocks on a decode and nothing here needs to be `Send`. A request is a
//! chunk of text; the answer is the raw logits of A, B and C at the readout marker.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Mutex;
use std::sync::mpsc;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::LlamaToken;

use super::intern::{CONTEXT_TOKENS, MAX_CHUNK_CHARACTERS, Readout, render};

type Reply = mpsc::SyncSender<Result<[f32; 3], String>>;

/// A loaded model behind a worker thread.
pub struct LlamaGate {
    tx: Mutex<mpsc::Sender<(String, Reply)>>,
}

impl LlamaGate {
    /// Loads the GGUF at `path`. A missing file or a load failure is an `Err`, never a panic.
    pub fn load(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Err(format!("model missing: {}", path.display()));
        }
        let path = path.to_path_buf();
        let (tx, rx) = mpsc::channel::<(String, Reply)>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), String>>(1);
        std::thread::Builder::new()
            .name("naru-ambient-gate".into())
            .spawn(move || match Worker::open(&path) {
                Ok(mut worker) => {
                    let _ = ready_tx.send(Ok(()));
                    while let Ok((text, reply)) = rx.recv() {
                        let _ = reply.send(worker.run(&text));
                    }
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            })
            .map_err(|e| format!("cannot start the gate thread: {e}"))?;
        ready_rx
            .recv()
            .map_err(|_| "gate thread died while loading".to_string())??;
        Ok(Self { tx: Mutex::new(tx) })
    }
}

impl Readout for LlamaGate {
    fn logits(&self, text: &str) -> Result<[f32; 3], String> {
        let (reply, answer) = mpsc::sync_channel(1);
        self.tx
            .lock()
            .map_err(|_| "gate lock poisoned".to_string())?
            .send((text.to_string(), reply))
            .map_err(|_| "gate thread is gone".to_string())?;
        answer
            .recv()
            .map_err(|_| "gate thread died mid-request".to_string())?
    }
}

struct Worker {
    // Field order is drop order: the context must go before the model, the model before the backend.
    context: llama_cpp_2::context::LlamaContext<'static>,
    model: Box<LlamaModel>,
    _backend: LlamaBackend,
    marker: LlamaToken,
    symbols: [LlamaToken; 3],
}

impl Worker {
    fn open(path: &Path) -> Result<Self, String> {
        let mut backend = LlamaBackend::init().map_err(|e| format!("llama backend: {e}"))?;
        backend.void_logs();
        let params = LlamaModelParams::default().with_n_gpu_layers(999);
        let model = Box::new(
            LlamaModel::load_from_file(&backend, path, &params)
                .map_err(|e| format!("model load failed: {e}"))?,
        );
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().min(8) as i32)
            .unwrap_or(4);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CONTEXT_TOKENS as u32))
            .with_n_batch(CONTEXT_TOKENS as u32)
            .with_n_ubatch(1024)
            .with_n_threads(threads)
            .with_n_threads_batch(threads);
        // SAFETY: the context borrows the boxed model, whose heap address is stable, and `Worker`
        // drops the context first (field order), so the borrow never outlives the model.
        let model_ref: &'static LlamaModel = unsafe { &*(model.as_ref() as *const LlamaModel) };
        let context = model_ref
            .new_context(&backend, ctx_params)
            .map_err(|e| format!("context init failed: {e}"))?;
        let vocab = model.vocab();
        let single = |text: &str, special: bool| -> Result<LlamaToken, String> {
            let t = vocab.tokenize(text.as_bytes(), false, special);
            match t.as_slice() {
                [one] => Ok(*one),
                _ => Err(format!("tokenizer: {text:?} is not one token")),
            }
        };
        let marker = single("<decision>", true)?;
        let symbols = [
            single("A", false)?,
            single("B", false)?,
            single("C", false)?,
        ];
        Ok(Self {
            context,
            model,
            _backend: backend,
            marker,
            symbols,
        })
    }

    fn run(&mut self, text: &str) -> Result<[f32; 3], String> {
        let vocab = self.model.vocab();
        // The context holds 4096 tokens; a chunk of many-token characters can overflow the character
        // cap, so trim the user text until it fits.
        let mut characters = MAX_CHUNK_CHARACTERS;
        let mut ids = vocab.tokenize(render(text, characters).as_bytes(), false, true);
        while ids.len() > CONTEXT_TOKENS && characters > 0 {
            characters = characters.saturating_sub(ids.len() - CONTEXT_TOKENS);
            ids = vocab.tokenize(render(text, characters).as_bytes(), false, true);
        }
        // The readout marker is the last one: a literal marker in the transcript (tokenized with
        // parse_special) must not move it.
        let at = ids
            .iter()
            .rposition(|t| *t == self.marker)
            .filter(|&i| i > 0 && ids.len() <= CONTEXT_TOKENS)
            .ok_or_else(|| "tokenizer: no readout marker".to_string())?;
        let mut batch = LlamaBatch::new(ids.len(), 1);
        for (i, id) in ids.iter().enumerate() {
            batch
                .add(*id, i as i32, &[0], i == at - 1)
                .map_err(|e| format!("batch: {e}"))?;
        }
        self.context.clear_kv_cache();
        self.context
            .decode(&mut batch)
            .map_err(|e| format!("decode failed: {e}"))?;
        let logits = self.context.get_logits_ith((at - 1) as i32);
        let mut out = [0f32; 3];
        for (o, s) in out.iter_mut().zip(self.symbols) {
            *o = *logits
                .get(s.0 as usize)
                .ok_or_else(|| "symbol outside the logits".to_string())?;
        }
        Ok(out)
    }
}
