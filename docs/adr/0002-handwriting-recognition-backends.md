# ADR-0002: Handwriting Recognition Backends

**Status:** Proposed
**Date:** 2026-08-01
**Deciders:** Nicholas-EW
**Branch:** `claude/handwriting-surya`

## Context

Measured on a real notebook-photo capture (15 pages, Adobe Scan), the
current PP-OCRv5 pipeline fails handwriting three distinct ways:

1. **Detection misses** — ink_recall mean 0.42, near 0.00 on
   display-size letters (detector trained on body-text line heights).
   box_precision stays 0.99: it never hallucinates, it omits.
2. **Recognition is print-trained** — near-misses on neat print-style
   handwriting (r/v, t/+, l/1 confusions), fragments on messy pages.
3. **Layout** — wavy baselines defeat `band_sort`'s
   `|y_center − band_mean| ≤ 0.5 × median_height` join predicate
   (band_sort.rs:104–132), scrambling word order.

Fixes 1 and 3 are pure-Rust pipeline work, activated by new CLI
arguments. Fix 2 needs a handwriting-capable recognition model, which
forces runtime/language questions this ADR settles.

### Constraints

- **License:** PaddleKnife is AGPL-3.0-or-later. AGPL forbids adding
  restrictions, so restricted-license model weights can never be
  bundled — only fetched on demand with the user accepting the model's
  own terms (the existing DESCRIPTORS pattern).
- **GPU fleet:** the default build is DirectML (any DX12 GPU on
  Windows); CUDA is a build-time opt-in feature. A backend that is
  CUDA-only on Windows abandons the default user.
- **Dependency pins:** `ort` is locked at `=2.0.0-rc.12` (oar-ocr
  0.6.3 / ort rc.13 do not compile together; pre-release semver never
  unifies, so any second ort consumer must pin rc.12 exactly).
  `tokenizers 0.22.2` is already in the tree via oar-ocr-core.
- **Process disciplines:** at most one live ORT engine set at a time
  (RunCtx drops before rebuild); pdfium bindings are process-global and
  not thread-safe.
- **Ethos:** ADR-0001 ported away from a Python runtime. Reintroducing
  a mandatory Python dependency needs strong justification.

### Research findings (2026-08-01, all claims verified against primary sources)

**Surya changed shape.** As of v0.20.0 (2026-05-27, current v0.22.1),
"Surya OCR 2" runs layout + OCR + table-rec through a **single ~686M-param
Qwen3.5-style VLM** (18 of 24 decoder layers are Gated-DeltaNet linear
attention; only 6 use conventional KV-cache attention). It is served
**only** through a spawned OpenAI-compatible server: vllm in Docker
(NVIDIA) or llama.cpp `llama-server` with official GGUF weights
(`datalab-to/surya-ocr-2-gguf`, ~1.27 GB + 205 MB mmproj). The v1
line-crop recognition API is gone — v2 recognizes full pages or
layout-block crops. Only text-line *detection* remains in-process
(torch EfficientViT segformer, ~77 MB fp16, cuda/mps/cpu).
[github.com/datalab-to/surya · README, surya/settings.py, surya/recognition/__init__.py]

**No usable Surya ONNX exists anywhere.** No official export; community
issue #397 closed unimplemented; the single HF export
(Prince-1/Surya-Ocr-2-Onnx) is missing the vision/embedding components
its own genai_config declares. An ONNX route would require the separate
onnxruntime-genai runtime (C API, no ort-crate unification).
`WillowVoiceAI/surya-ocr-2` is an unofficial **byte-identical mirror**
of `datalab-to/surya-ocr-2` (same LFS sha256) — reference the official
repo only.

**Surya weights are not open.** Modified AI Pubs OpenRAIL-M: free only
for research/personal use or organizations under **$5M revenue AND $5M
funding**; a no-compete clause with no revenue floor; share-alike that
extends to model **output**; a remote-restriction clause. Code is
Apache-2.0; weights must be download-on-demand, never bundled.
[huggingface.co/datalab-to/surya-ocr-2 · LICENSE, Attachment A]

**Surya's handwriting ability is undocumented.** One "Handwritten
Notes" demo image; no handwriting benchmark. Worst olmOCR-bench
category: OldScan 41.8%. Throughput without NVIDIA/vllm: 0.108 pages/s
(~9 s/page) on Apple-Silicon llama.cpp — expect no better on Windows
CPU/Vulkan.

**surya-rs is not viable.** Detection-only (recognition literally
commented out in `src/bin/main.rs`; roadmap unchecked), dormant (last
human commit 2024-07-28, crates.io frozen at 0.4.0), pinned to Jan-2024
CC-BY-NC weights incompatible with Surya 2, candle 0.6.0 backend with
no cuda feature exposed and **no DirectML backend exists in candle at
all** (issue #344 open since 2023), plus a mandatory native OpenCV +
libclang build dependency on Windows.

**INF-MLLM is not viable.** Beaten by Qwen2.5-VL-7B on every published
doc/OCR benchmark (DocVQA 92.2 vs 95.7, OCRBench 772 vs 864); no
GGUF/ONNX/vllm path; `trust_remote_code` PyTorch pinned to
transformers 4.40.2 / torch 2.2.1, CUDA-only demo, ~16 GB fp16; the
GitHub repo has **no code license**. INF Tech's own current parsers
(Infinity-Parser2 family) are Qwen-architecture — the vendor moved on.
**Qwen2.5-VL-7B-Instruct** is the VLM benchmark leader with a clean
story: Apache-2.0, official GGUF + mmproj (`ggml-org` repo), ~6 GB at
Q4, llama.cpp Vulkan backend = any DX12 GPU. (Avoid the 3B: Qwen
Research License, non-commercial.)

**The in-process enc-dec path is proven.** pykeio/ort's own
phi-3-vision example implements exactly the autoregressive
`past_key_values.{i}.key/.value` → `present.{i}.key/.value` KV-cache
loop a HF-Optimum `decoder_model_merged.onnx` needs. Ready-made TrOCR
ONNX artifacts exist (Xenova: encoder + merged decoder, fp32/fp16/
quantized down to ~40 MB). rten also ships a Rust TrOCR example —
two independent precedents. Caveats: ort's DirectML EP wants static
shapes, requires memory-pattern off + sequential execution, and has an
unresolved accuracy report on a merged decoder — so the **decoder runs
on the CPU EP by default** (CUDA where built), while the detector keeps
its DirectML path. DirectML itself is in maintenance mode (opset 20
ceiling) — a medium-term fleet question beyond this ADR.

**Python sidecar reality.** torch-directml is dead (last release
2024-09, pins torch 2.4.1; Surya needs ≥2.7) — a Surya Python sidecar
on Windows is CPU-torch for detection plus the same llama-server
management we could do ourselves. Surya's CLI does emit clean
`results.json` (polygon/bbox/text/confidence) and `--keep_server`
amortizes model load; uv-managed venv is the sane bootstrap
(Anki/ComfyUI precedent) if this path is ever needed.

**Our seam already exists.** `pub trait OcrBackend: Send { fn ocr(&mut self,
img: &DynamicImage) -> Result<Vec<OcrBlock>> }` + `OcrEngine::with_backend`
(ocr.rs:328–330, 381–383). oar-ocr-core 0.6.2 ships a standalone
`TextDetectionPredictor`, so a det/rec split reuses the existing
PP-OCRv5 detector for recognition-only backends. `download_file()` is
URL-agnostic; only the DESCRIPTORS registry hardwires the
det+rec+dict triple and needs an artifact-list generalization.

## Decision

Build a **pluggable recognition-backend architecture** behind explicit
CLI arguments, in three phases on this branch:

**Phase 1 — Rust-only pipeline fixes (args for A and B):**

```
--preprocess {auto|clahe|off}       contrast normalization before detection
--det-retry-scales <s1,s2,…>        multi-scale detection rescue (e.g. 0.5,0.25)
                                    triggered when a page yields few boxes but
                                    much unexplained ink (ink-coverage check
                                    ported from eval_box_accuracy.py)
```

plus the localized `band_sort` wavy-baseline fix (expose the 0.5
multiplier; switch band joining to local-neighbor chaining).

**Phase 2 — in-process handwriting recognizer via ort (the "model
argument … via onnx"):**

```
--rec-backend {paddle|trocr}        default: paddle (today's pipeline)
--rec-model <alias|path>            registry alias or local ONNX bundle dir
```

TrOCR-class encoder + merged-decoder ONNX under the existing ort
rc.12, HF tokenizer via the already-present `tokenizers` crate,
line crops supplied by the existing PP-OCRv5 detector through the
det/rec split. Decoder on CPU EP by default, CUDA when built with it;
detector unchanged on DirectML. DESCRIPTORS generalizes from
(det, rec, dict) to `artifacts: &[Artifact]` so a bundle can be
`encoder.onnx + decoder_model_merged.onnx + tokenizer.json`.
Scope note: TrOCR-handwritten is English-only (IAM-trained) — correct
for the motivating captures, not a multilingual answer.

**Phase 3 — server-class VLM backend (one client, many models):**

```
--rec-backend vlm
--rec-model {surya-2|qwen2.5-vl-7b|<gguf-path>}
--vlm-endpoint <url>                attach to an already-running server
```

A single OpenAI-compatible HTTP client (ureq, already a dependency)
that PaddleKnife points at a managed `llama-server.exe` (llama.cpp
prebuilt Windows binaries; Vulkan covers the DX12 fleet, CUDA/HIP
where present). The same client serves **Surya-2 GGUF** (its only real
integration path — no Python required; prompt/parse logic ported from
Surya's Apache-2.0 code) and **Qwen2.5-VL GGUF**, and is the natural
substrate for the translation roadmap's LLM-endpoint phase. Surya
weights download-on-demand behind an explicit RAIL-license acceptance
prompt; never bundled.

## Options Considered

### Option 1: TrOCR-class enc-dec via ort, in-process ✅ (Phase 2)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium — KV-cache decode loop (proven pattern), greedy first |
| Cost | ~40–350 MB models, zero new heavyweight deps |
| GPU | Encoder DML-capable; decoder CPU-EP default, CUDA opt-in |
| License | Model weights MIT (Microsoft) — confirm at descriptor time |
| Fit | Perfect: same runtime, same detector, same seam |

**Pros:** in-process; tiny; offline; slots into DESCRIPTORS; two Rust precedents.
**Cons:** English handwriting only; decoder not DML-safe (CPU decode); quality ceiling below VLMs on messy pages.

### Option 2: Surya 2 via llama-server GGUF, HTTP backend ✅ (Phase 3)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium-high — server lifecycle mgmt + prompt/parse port |
| Cost | ~1.5 GB download; ~9 s/page without strong GPU |
| GPU | llama.cpp Vulkan/CUDA/HIP — vendor-neutral, covers DX12 fleet |
| License | Code Apache-2.0; **weights RAIL-restricted** — gate + never bundle |
| Fit | Good as *server tier*; no Python needed |

**Pros:** honors the Surya preference via its only real path; client doubles for Qwen-VL and future translation endpoint.
**Cons:** handwriting quality unproven (no benchmark exists); heavy; license gates commercial users; Windows officially undocumented upstream.

### Option 3: Full surya-ocr Python sidecar ❌ (fallback only)

Faithful to upstream behavior, but on Windows it degenerates to
CPU-torch detection + the same llama-server we'd manage anyway, plus a
Python env (uv venv) PaddleKnife must bootstrap. Dominated by
Option 2 except for fidelity. Revisit only if the prompt/parse port
proves unfaithful.

### Option 4: surya-rs (candle) ❌ rejected

No recognition (commented out), dormant since 2024, obsolete
non-commercial weights, no CUDA feature, candle has no DirectML at
all, mandatory OpenCV native dep. Fails on every dimension.

### Option 5: INF-MLLM ❌ rejected

Loses to Qwen2.5-VL-7B on every published doc benchmark; no deployable
non-CUDA path; unlicensed code; 16 GB fp16. Its own vendor now builds
on Qwen. Qwen2.5-VL-7B replaces it inside Option 2's client.

### Runtime eliminations (language-reliance analysis)

| Runtime | Verdict | Why |
|---|---|---|
| **ort 2.0.0-rc.12** | ✅ in-process backbone | Already pinned; KV-cache decode proven; DML (det) + CPU/CUDA (dec) |
| **llama.cpp (C++)** | ✅ server tier | Official Surya-2 path; Vulkan = any DX12 GPU; prebuilt Windows exes |
| candle | ❌ | No DirectML/DX12 ever (issue #344); Windows CUDA builds fragile |
| tract | ❌ | No DirectML; CUDA new; merged-decoder If-op support unverified |
| tch-rs / libtorch | ❌ | 188 MB (CPU) – 2.45 GB (CUDA) DLL payload; CUDA-only GPU |
| Python/torch | ❌ as dependency | torch-directml dead; Surya itself moved recognition out of torch |

## Trade-off Analysis

The pivotal trade is **in-process fidelity vs server-class quality**.
Phase 2 gives an offline, tiny, license-clean handwriting recognizer
that will fix the *legible* print-style pages (the r/v, t/+ class of
error) but not the messiest cursive. Phase 3 buys the VLM ceiling at
the price of a managed server process, gigabyte downloads, and — for
Surya specifically — a restricted weights license and unbenchmarked
handwriting claims. Doing both behind one `--rec-backend` argument
means neither blocks the other, and the A/B comparison on the user's
actual notebook scans (via `eval_box_accuracy.py` + text diffing)
decides which tier becomes the recommended default for handwriting.

The pivotal *non*-trade: nothing requires Python. Surya's own
architecture change (recognition → llama-server) means the "Python
sidecar" question dissolved into "manage a C++ server binary", which
Rust does comfortably over HTTP with ureq.

## Consequences

**Easier:** adding future recognizers (one trait + registry entry);
translation roadmap Phase-2 (LLM endpoint) inherits the VLM client;
per-page backend routing (low-confidence → handwriting tier) becomes a
config concern, not an architecture change.

**Harder:** DESCRIPTORS registry migration touches every existing
script entry; llama-server lifecycle (spawn, health-check, port pick,
teardown) is new operational surface; the one-live-engine discipline
now spans multi-session bundles (encoder + decoder count as one slot).

**Revisit when:** DirectML's maintenance-mode status forces an EP
strategy change; Surya publishes a handwriting benchmark or relicenses;
onnxruntime-genai grows a Rust binding (would reopen in-process VLM);
Infinity-Parser2 (qwen3_5-arch doc parser) publishes weights + license
worth evaluating in the VLM client.

## Action Items

0. [ ] Evaluate `yassa9/qwen600` (user-proposed minimal CUDA Qwen
       runner) as a possible CUDA-specific in-process alternative to
       llama-server for the Phase-3 VLM tier: verify multimodal/vision
       support (mmproj?), qwen3_5 hybrid-attention compatibility
       (Surya-2 needs Gated-DeltaNet linear-attention layers, not just
       KV-cache attention), weight format, license, maintenance, and
       Windows build story. Also re-check whether a newer multimodal
       Qwen (Qwen3-VL family) with llama.cpp support supersedes
       Qwen2.5-VL-7B as the VLM alias. Fold verdicts into Option 2's
       runtime table before Phase 3 starts.
1. [ ] Phase 1: `--preprocess`, `--det-retry-scales`, ink-coverage
       trigger, band_sort baseline fix + tests on the 15-page notebook-photo capture
2. [ ] Registry: `Artifact`-list generalization of DESCRIPTORS with
       per-artifact URLs/licenses; license-acceptance gate for
       restricted weights
3. [ ] Det/rec split via oar-ocr `TextDetectionPredictor`; route
       existing pipeline through it unchanged
4. [ ] Phase 2: TrOCR backend (encoder + merged decoder, greedy
       decode, `tokenizers`), `--rec-backend trocr`,
       `--rec-model base-handwritten|large-handwritten|<path>`;
       confirm MIT license on chosen ONNX artifacts at descriptor time
5. [ ] Phase 3: llama-server manager + OpenAI-compatible client;
       Surya-2 prompt/parse port (from Apache-2.0 code); Qwen2.5-VL
       model alias; `--vlm-endpoint` attach mode
6. [ ] Benchmark all tiers on the 15-page notebook-photo capture (ink_recall +
       transcription diff); pick the recommended handwriting default
7. [ ] README + `--help` documentation of backend/argument matrix
