# Multi-script recognizers — Cyrillic, Latin-extended, Korean, CJK

**Date:** 2026-06-12
**Status:** Implemented & verified (Plan 6) — commits on
`claude/thirsty-cannon-0a793c`

## Results (verified end-to-end with synthetic samples)

All six requested scripts auto-detect and recognize correctly from
Windows-font-rendered samples:

| Script | Auto-routed to | Recognized text |
|---|---|---|
| Chinese | LatinCjk (fast path) | 中华人民共和国成立了 |
| Japanese | LatinCjk (fast path) | 日本語のテキスト認識 |
| Czech | Latin (diacritic upgrade) | Příliš žluťoučký kůň úpěl |
| Russian | Eslav (race) | Привет мир интересный |
| Ukrainian | Cyrillic (race) | Українська мова цікава |
| Korean | Korean (race, downloaded) | 한국어 텍스트 인식 테스트 |

Russian multi-line → searchable PDF: `ink_recall 0.971`, `box_precision
1.000`; Cyrillic text round-trips out of the PDF exactly. 102 workspace
tests + 13 auto-detect decision-fn unit tests pass; clippy clean.

Two bugs found by verification (both fixed): transliterated-Cyrillic
read as confident Latin homoglyphs (mid-word-capital guard added), and
the race early-exit (0.85) firing on wrong-script confident-garbage
before the right model ran (raised to 0.95, below correct-script scores
0.97–1.00). Decision logic is pure and unit-tested.


**Builds on:** the glyphless text layer (Plan 5) — box drawing is already
fully script-agnostic, so this plan adds only *recognition* coverage and
*routing*, not layout.

## Goal

Read and produce correct searchable PDFs for Russian, Ukrainian, Czech,
Korean, Japanese, and Chinese, in addition to the existing Latin/CJK and
Devanagari. Box drawing, CJK/Thai join, and RTL ordering already work for
all of these — verified by the Plan 5 alignment harness.

## What gates each script (verified)

The bundled `ppocrv5_dict.txt` (18,383 entries) covers: 15,565 CJK
ideographs, 180 kana, basic Latin. So:

- **Chinese, Japanese:** already recognized by the bundled model. Needs
  verification, not new models.
- **Czech:** bundled dict is missing `ď ť ů ž` → needs the `latin` model.
- **Russian/Ukrainian:** 11 Cyrillic chars bundled → needs `eslav`.
- **Korean:** 2 Hangul bundled → needs `korean`.

PaddlePaddle publishes official pre-converted ONNX (Apache-2.0) for all
three, and the detector is shared across every script.

## Design

### 1. Script registry (replaces the 2-variant enum)

`model.rs`: `RecognizerScript` grows from 2 hardcoded arms to a small
enum (`LatinCjk`, `Devanagari`, `Latin`, `Cyrillic`, `Eslav`, `Korean`)
backed by a static descriptor table. Each descriptor carries:

```
struct ScriptDescriptor {
    id: RecognizerScript,
    cli_value: &str,        // --script value
    rec_filename: &str,     // models/<rec>.onnx
    dict_filename: &str,    // models/<dict>.txt
    hf_onnx_repo: &str,     // PaddlePaddle/<model>_onnx  (download)
    dict_url: &str,         // PaddleOCR raw dict .txt     (download)
    postproc: PostProc,     // None | Nepali
}
```

`rec_filename()` / `dict_filename()` keep their signatures (read from the
table) so existing callers and tests are unaffected. Detector path stays
shared. This is the only structural change; everything downstream of
`ModelPaths` is already script-blind.

### 2. Post-processing policy (per descriptor)

The destructive Nepali cleanup becomes one `PostProc::Nepali` policy bound
to Devanagari only. Every new script is `PostProc::None`. PDF output
already skips postproc entirely (Plan 5 fix), so this only affects
txt/json/md — where each new script gets clean passthrough.

### 3. Download-on-demand

New `paddleknife models pull <script>` subcommand + auto-fetch when a
recognizer is selected and its files are absent. Downloads:

- rec ONNX from `https://huggingface.co/PaddlePaddle/<repo>/resolve/main/inference.onnx`
- dict `.txt` from the PaddleOCR GitHub raw path

into `models/`, with a progress bar (reuse `indicatif`) and atomic
write (temp + rename). HTTP via `ureq` (blocking, rustls — small, no
async runtime). Checksums/sizes logged; a failed download leaves no
partial file. `models pull --all` fetches everything.

### 4. Full auto-detection (layered)

`--script auto` (default) becomes a layered decision, generalizing the
existing LatinCjk-vs-Devanagari confidence race:

1. **First pass** with the broad `LatinCjk` model on page 1 → text + avg
   confidence.
2. **High confidence (≥0.7):** the base model read it well.
   - Codepoint-classify the text (`script.rs::detect_script`).
   - Mostly ASCII-Latin / CJK / Japanese → keep `LatinCjk` (fast path,
     no extra passes — the common case).
   - Latin with significant Extended-Latin (diacritics over a threshold)
     → switch to `Latin` for full coverage (catches Czech/Polish/…).
3. **Low confidence (<0.7):** the base model failed → a script it can't
   emit. Run a confidence race across the specialist models that are
   present on disk (`Devanagari`, `Eslav`, `Cyrillic`, `Korean`, `Latin`),
   auto-downloading any that are missing, and pick the highest average.

The decision function is pure (`text + confidence + which models exist →
chosen script`) so it is unit-tested without touching ORT or disk.

**Honest limit:** Latin-script languages that the base model reads at
*moderate* confidence (not clearly failing) may route to `LatinCjk`
instead of `Latin`; the Extended-Latin-ratio test in step 2 mitigates
this, and `--script czech`/`latin` always forces it. Cyrillic/Korean/
Devanagari are robustly auto-detected because the base model clearly
fails on them.

### 5. CLI

`ScriptArg` gains `russian`, `ukrainian`, `czech`, `latin`, `korean`,
`chinese`, `japanese` values (several aliasing onto the same descriptor —
`russian`/`ukrainian` → `Eslav`, `czech` → `Latin`, `chinese`/`japanese`
→ `LatinCjk`). `auto` stays the default.

## Testing

- **Unit:** descriptor table resolution; auto-detect decision function
  (table-driven cases per script); download-URL construction; postproc
  policy mapping.
- **Integration:** pull one real model (`latin` or `korean`), confirm it
  loads through oar-ocr and runs on a synthetic strip without error.
- **Verification boundary (explicit):** I can prove the *machinery* —
  routing, download, load, box accuracy via `eval_box_accuracy.py`. I
  cannot fully prove *recognition accuracy* on languages with no test
  document on hand (Korean/Russian/Czech/Chinese/Japanese); that needs
  real sample PDFs from you. Chinese/Japanese I can verify works today if
  you supply (or I synthesize) a sample.

## Out of scope

- Two-stage detector+script-ID auto (rejected in favor of the layered
  confidence approach).
- Per-script postproc beyond passthrough (can come later if a language
  needs cleanup).
- Mixed-script *per-line* routing within one page (still whole-document).
- Server (vs mobile) rec models.

## Implementation order

1. Registry refactor (`RecognizerScript` table) + postproc policy + tests.
2. CLI `ScriptArg` values + wiring; verify existing Devanagari/Latin paths
   still pass.
3. Download module + `models pull` subcommand (test on one real model).
4. Layered auto-detect + decision-function unit tests.
5. Verify: CJK sample end-to-end; box-accuracy on whatever samples exist;
   docs + memory update.
