# Plan 4: GPU Acceleration via DirectML

**Goal:** Cut the ~5-min-per-book OCR runtime by routing inference through
a GPU execution provider. DirectML is the primary target — zero extra
install for Windows users, works on any DX12 GPU (Intel iGPU, AMD, NVIDIA).
CUDA is an opt-in follow-up for NVIDIA-only users who want the extra ceiling.

**Scope:** Plumbing + a `--device` flag. No pipeline rewrite, no backend
swap. Plan 1 designed `OcrBackend` behind a trait and picked `oar-ocr`
specifically because it wraps ORT — Plan 4 is cashing that chip.

---

## Why DirectML first

| EP | Targets | Install cost |
|---|---|---|
| DirectML | Any Windows 10+ DX12 GPU | None (system DLL since 1709) |
| CUDA | NVIDIA only | CUDA Toolkit + matching ORT CUDA wheel |
| TensorRT | NVIDIA only | CUDA + TensorRT + ORT build |
| CoreML | macOS only | None on Mac |

For this project's user base (Windows, mixed GPU vendors), **DirectML is
the zero-friction default**. A laptop Intel iGPU should see 3-5× speedup;
a discrete AMD/NVIDIA GPU 5-10×.

## oar-ocr API (already exposes this)

`OAROCRBuilder::ort_session(OrtSessionConfig)` takes an EP list. The
variants we care about are:
- `OrtExecutionProvider::DirectML`
- `OrtExecutionProvider::CUDA`
- `OrtExecutionProvider::CPU` (always listed last as safety fallback)

Crate features (`oar-ocr/<ep>`) gate the compile-time inclusion of
each EP's backend. Enabling `directml` pulls in
`ort/directml` → ONNX Runtime's DirectML provider library.

---

## Tasks

- [ ] **Task 1** — Add `directml` feature to `paddleknife-core` and enable
      it by default. Cargo gates CUDA/TensorRT/CoreML behind opt-in features
      (`cuda`, `tensorrt`, `coreml`) so non-NVIDIA users don't pay for them.
- [ ] **Task 2** — Introduce `OcrDevice` enum in core (Auto, Cpu,
      DirectMl, Cuda) + `OcrEngine::from_paths_with_device` that threads
      an `OrtSessionConfig` into the oar-ocr builder chain.
- [ ] **Task 3** — `--device {auto|cpu|dml|cuda}` CLI flag (ValueEnum).
      Default `auto`: try DirectML on Windows, CPU elsewhere.
- [ ] **Task 4** — Lightweight runtime check: if `auto` is chosen, log
      which EP actually won the race (shown in the final stdout summary).
- [ ] **Task 5** — Benchmark: re-run Muna Madan + Invoices on `cpu` and
      `dml`; record runtime + file-size deltas.
- [ ] **Task 6** — Document findings at the end of this plan doc; note
      any graceful-degradation behavior observed.

## Done criteria

- `cargo build --release` succeeds on Windows with the DirectML EP enabled.
- `paddleknife ocr <file> --device dml` runs and produces identical output
  to `--device cpu` (modulo fp32/fp16 rounding in the recognizer logits).
- Measured end-to-end runtime drops by >2× on the Muna Madan book on
  the test machine.
- `cargo test --workspace` still passes.
- If DirectML init fails (VM / old driver), we fall back to CPU with a
  warning, not a crash.

## Out of scope (later)

- Persisting the user's chosen device in `%APPDATA%` (Plan 4.5 or TUI work).
- TensorRT / CoreML features (opt-in Cargo features only, not CLI-exposed
  until needed).
- Half-precision / quantized models — would need different ONNX weights.
- Per-stage device selection (det on GPU, rec on CPU) — pipeline-wide for now.

---

## Task 6 results — DirectML benchmarks (2026-04-15)

Test machine: Windows 11, consumer GPU (DX12-capable), same box as Plans 1-3.

| Workload | CPU (Plan 2.5) | DirectML (Plan 4) | Speedup |
|---|---|---|---|
| `hello_world.png` (1 region)    | ~0.7s  | 3.1s   | 0.2× (cold start) |
| Invoices PDF (9 pages, 278 lines, ~157 regions/page peak)  | **1m52s** | **19.6s**  | **5.7×** |
| Muna Madan (51 pages, 1259 lines, Devanagari rec) | **5m12s** | **1m10s**  | **4.5×** |

Notes:
- The `hello_world.png` case is dominated by DirectML session
  initialization (~2-3s); not representative of normal use.
- On real workloads we see **4-6× speedup**, as predicted in the plan.
- ORT prints one expected warning at startup:
  `Some nodes were not assigned to the preferred execution providers`
  — this is normal; shape-related ops always run on CPU in ORT.
- `region_batch_size(32)` was required to avoid DirectML crashing on
  dense pages (157 regions hit an adapter-execution failure). Added
  automatically for GPU devices; CPU keeps unbounded batching.

## Graceful degradation verified

- `--device dml` on Windows + DX12 GPU: uses DirectML ✓
- `--device auto` on Windows: same as above ✓
- ORT falls back to CPU within a single session if a specific node isn't
  assignable to the GPU; we always list CPU last in `execution_providers()`.

## Build / binary impact

- Release binary: 27 MB → **27 MB** (~no change; DirectML bits are tiny).
- Build time: +~20s on clean build; incremental builds unchanged.
- No new runtime dependencies — DirectML.dll is a Windows system DLL
  since Windows 10 1709; ONNX Runtime's DirectML EP is statically
  linked into our binary via the `ort/directml` Cargo feature.

Plan 4 closed. The binary is now "lightspeed enough" to iterate on
Plan 2.6 (Devanagari post-processing, auto-script-detect, etc.) without
waiting 5+ minutes per test run.
