# Plan 4.5: Batch-Size Tuning + CUDA as Opt-In

**Goal:** Squeeze more throughput out of the GPU path for users with
extra VRAM, and make CUDA an officially-supported opt-in build for
NVIDIA owners. DirectML stays the default.

**Scope:** Two CLI knobs (batch sizes), a benchmark sweep, and
verification of the CUDA build. No backend rewrite.

---

## Background

Plan 4 capped `region_batch_size` at 32 because DirectML crashed on a
dense scanned page (157 regions). 32 is oar-ocr's "medium VRAM"
recommendation and was the safe default. But machines with discrete
GPUs (4 GB+ VRAM) likely tolerate much larger batches with measurable
gains in wall-clock time.

`image_batch_size` is the corresponding knob for the *detector* — how
many full pages get sent through det per call. For PDF batch processing
this matters more than for one-document-at-a-time CLI usage.

CUDA support was wired in Plan 4 as an opt-in Cargo feature
(`paddleknife-core/cuda`) but has never been actually built or bench-
marked because the test machine doesn't have CUDA Toolkit installed.

## Tasks

- [ ] **Task 1** — Add `--region-batch-size N` and `--image-batch-size N`
      CLI flags. Both default to "auto" (current behavior); explicit
      values override the auto-cap.
- [ ] **Task 2** — Benchmark sweep on the invoices PDF with
      `--device dml`: batch 32 (baseline), 64, 128, 256. Record runtime
      and any failures.
- [ ] **Task 3** — If a larger value is consistently faster on this
      machine without errors, bump the default cap.
- [ ] **Task 4** — Try `cargo build --features cuda` on the test machine.
      If it builds, benchmark `--device cuda` against `--device dml`.
      If it doesn't (no CUDA Toolkit), document the build requirement
      so future NVIDIA-owning users know what's needed.
- [ ] **Task 5** — Document the findings in this plan file.

## Done criteria

- Both CLI knobs are present and validated by clap.
- A documented sweep of batch sizes vs runtime exists in this file.
- The default `region_batch_size` for GPU is updated to the largest
  value that completes reliably on the test machine.
- CUDA build status documented (works / requires X / not tested).
- `cargo test --workspace` + clippy `-D warnings` clean.

## Out of scope (Plan 4.6+)

- Tuning ORT thread counts (intra/inter ops) — minor on GPU paths.
- TensorRT integration (NVIDIA-only, FP16 quantization, hardest path).
- Auto-detect VRAM and pick the batch size dynamically.
- ONNX model quantization (would need different model weights from
  Paddle's release).

---

## Task 5 results — empirical bench sweep

Test machine: Windows 11, consumer DX12 GPU + CUDA-capable NVIDIA card.
Workloads: 9-page invoices PDF (dense — page 1 has 157 regions),
51-page Muna Madan (sparse — typically 20-40 regions/page).

### DirectML region-batch sweep (invoices)

| `--region-batch-size` | Runtime | Notes |
|---|---|---|
| 32 (Plan 4 default)   | 19.3s | Conservative |
| **64 (new default)**  | **14.3s** | **35% faster** |
| 128                   | 15.8s | Diminishing returns; cache pressure |
| 256                   | FAIL  | `adapter execution failed` on 157-region page |

### DirectML region-batch sweep (Muna Madan, --script devanagari)

| Batch | Runtime |
|---|---|
| 32  | 1m16.8s |
| **64** | **1m11.6s** | (7% faster — sparser pages, smaller win) |
| 128 | 1m11.9s | (tied with 64) |

Conclusion: **64 is universally equal or faster**, never worse. New default.

> **Correction (Plan 10, 2026-09-21).** "Universally" was too strong: both
> workloads here have *narrow* lines. A batch is padded to its widest crop, so
> its memory is `count × width`, and 64 crops sharing a batch with one 50:1 line
> needs ≥ 9 GiB. The `256 → FAIL` row above was that mechanism, not "DirectML
> crashing at extreme batch sizes". 64 remains the default count cap, but only
> together with the padded-area budget — see
> `2026-09-21-plan-10-resilient-batching.md`.

### CUDA build + runtime

`cargo build --release --no-default-features --features cuda` succeeded
on the test machine — ort 2.x bundles the CUDA execution provider, no
separate CUDA Toolkit needed at build time. Runtime requires the user
have CUDA runtime DLLs (cudart, cudnn, cublas) on PATH or in the system.

CUDA runtime bench (invoices, batch=64): **14.6s** — basically tied
with DirectML (14.3s). Larger batches (128/256/512) didn't help; CUDA
*tolerates* batch=256 (DML crashes there) but doesn't get faster.

CUDA on Muna Madan (batch=64): **1m51s**, *slower* than DirectML's
1m11s. CUDA's per-kernel-launch overhead hurts on the smaller PP-OCRv5
recognizer when many short batches are dispatched.

### Recommendations table for users

| User profile | Build | Flags |
|---|---|---|
| Default Windows | DirectML (default) | nothing — auto picks DML |
| NVIDIA, very dense pages | `--features cuda` | `--device cuda --region-batch-size 256` |
| Old Windows / VM / no GPU | DirectML (default) | `--device cpu` |
| Discrete AMD GPU | DirectML (default) | nothing |
| Mac | rebuild needed | `--features coreml`; `--device auto` (picks CoreML) |

CUDA is not the speed-king on this size of model. Its value is
reliability at extreme batch sizes (when DirectML crashes) and
NVIDIA-specific pinned-memory paths if you start chaining many docs.

### CLI surface added

```
--region-batch-size <N>   Override recognizer batch (default: 64 on GPU)
--image-batch-size  <N>   Override detector batch (default: oar-ocr's)
```

Plan 4.5 closed. Move on to Plan 3 (Ratatui TUI).
