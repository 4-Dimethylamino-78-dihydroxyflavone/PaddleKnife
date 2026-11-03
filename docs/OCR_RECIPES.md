# OCR recipes — optimised arguments

Every number here was measured on the reference corpus in `tests/` with the
Rust CLI on Windows + DirectML (PP-OCRv6, oar-ocr 0.8.1). Nothing is quoted
from a vendor benchmark.

---

## The universal command

```bash
paddleknife ocr INPUT --device auto --quality medium --script auto -f pdf --detect-page-orientation
```

That is the best-quality setting for essentially everything, and on a GPU it
costs nothing over the alternatives. Every flag in it is already the default
except `--detect-page-orientation` and `-f pdf`, so the short form is:

```bash
paddleknife ocr INPUT -f pdf --detect-page-orientation
```

For plain text instead of a searchable PDF, drop to `-f txt`.

### Why these values

| Flag | Value | Reason |
|---|---|---|
| `--device` | `auto` | Picks DirectML on Windows / CoreML on macOS, falls back to CPU. PP-OCRv6 medium runs fine on DirectML — verified, no execution-provider failure. |
| `--quality` | `medium` | **On GPU all three tiers take the same wall time** (see below), so there is no reason to trade accuracy away. Only drop tiers on a CPU-only machine. |
| `--script` | `auto` | Samples the first page with text (of the first three) and routes; keeps latin-cjk when all three are blank. Pin it only when you already know the language — it saves one sample pass. |
| `--detect-page-orientation` | on | One tiny classifier pass per page, and it genuinely is 4-class (0/90/180/270). Rescues fully sideways or upside-down scans. Needs the classifier model (see below). **Safe only since Plan 10** — see the warning below. |

**The flag needs its classifier.** `--detect-page-orientation` runs the
PP-LCNet document-orientation model (`pp-lcnet_x1_0_doc_ori.onnx`, about
6.5 MB), a separate file in the models folder. The first run that uses the
flag downloads it. `paddleknife models list` shows whether it is present.
On a machine that will run offline, fetch it ahead of time with
`paddleknife models pull --orientation`. `--detect-line-orientation` works
the same way with its own classifier.

> **Builds from before 2026-09-30 must not use `--detect-page-orientation`
> on screenshots or Devanagari pages.** Those builds predate Plan 10. The
> classifier is trained on scanned documents and misjudges other material as
> upside-down — with scores as high as its correct verdicts, so nothing flagged
> it. Measured with the old binary: 3 of 9 upright chat screenshots garbled
> (confidence 0.97 → 0.65), and 3 of 4 upright *Muna Madan* pages reduced from
> ~150 text blocks to **0 or 1**. Since Plan 10 every verdict is checked against
> recognition confidence: a page that comes back poorly read is re-read flipped
> 180° and the better result kept. On the same pages the new binary matches the
> best achievable result on every one, including genuinely rotated copies. Cost:
> one extra pass, only on pages that read poorly the first time.

### Tier timings are flat on GPU

Same 4894×6991 receipt, DirectML, wall clock including model load:

| `--quality` | Weights | Time | Confidence |
|---|---|---|---|
| `tiny` | 6.0 MiB | 4 s | 0.96 |
| `small` | 29.6 MiB | 4 s | 0.99 |
| `medium` | 132.2 MiB | 4 s | 0.98 |

15-page handwriting PDF: `small` 18 s, `medium` 19 s.

These models are small enough that GPU execution is dominated by image
transfer and per-region batching, not by the network. The ~2.4× CPU
detection penalty for `medium` (measured separately at 1.42 s vs PP-OCRv5
mobile's 0.58 s per page) simply does not appear on DirectML.

**So: use `medium` unless you are on CPU only.** On CPU, `small` is the sweet
spot and `tiny` is genuinely faster than the old PP-OCRv5 mobile detector
while scoring better.

---

## Per-document recipes

### Receipts and invoices

```bash
paddleknife ocr receipt.jpg --script english -f txt
```

Pinning `--script english` skips the auto-detect sample pass. No other
tuning needed — faded thermal paper and EFTPOS dot-matrix both read cleanly
at default thresholds.

### Books and long scans

```bash
paddleknife ocr book.pdf -f pdf --detect-page-orientation
```

`-f pdf` writes the original raster plus an invisible text layer.
`--pdf-image auto` (the default) copies the original scan bytes through
verbatim when a page is a plain single-image scan, so output stays roughly
input-sized rather than ballooning.

For a folder of page scans that form one book:

```bash
paddleknife ocr ./pages --merge -f pdf --detect-page-orientation
```

### Badly scanned / photographed pages

```bash
paddleknife ocr scan.jpg -f pdf --detect-page-orientation --detect-line-orientation
```

`--detect-line-orientation` is the only case where the extra ~15% latency is
worth it: it corrects **upside-down** lines. See the caveat below — it does
*not* fix sideways text.

### Rotated / multi-column material

```bash
paddleknife ocr insert.jpg --script english -f txt
```

No special flag. Sideways text is handled by quad ordering in the OCR
adapter, and `band_sort` groups by reading axis — horizontal text bands by y
and orders by x, rotated text bands by x and orders by y. Horizontal lines
are emitted first, rotated ones after.

### Redacted / typewriter-era documents

```bash
paddleknife ocr foia.pdf --script english -f txt
```

Defaults are fine. Expect character-level errors on 1950s carbon copy —
that is the source, not the model.

### Handwriting

```bash
paddleknife ocr notes.pdf --script english --quality medium -f txt
```

Use `medium` specifically. On real pencil-on-notebook samples it beat
`small` on exactly the words that matter:

| | `small` | `medium` |
|---|---|---|
| confidence | 0.89 | **0.93** |
| sample words | "ylars", "veminder", "appealance", "ifs" | **"years", "reminder", "appearance", "it's"** |

PP-OCR is a printed-text CTC recogniser, so treat handwriting output as a
searchable draft rather than a transcript.

### Nepali / Devanagari (and Cyrillic, Korean)

```bash
paddleknife ocr book.pdf --script nepali -f pdf
```

PP-OCRv6's dictionary contains **no** Devanagari, Cyrillic or Hangul, so
these scripts keep their PP-OCRv5 recogniser at every `--quality` value.
`--quality` still selects the v6 *detector*, which is script-agnostic — you
get better detection with unchanged recognition. Aliases: `nepali`, `hindi`,
`sanskrit`, `marathi`, `cyrillic`, `russian`, `korean`, `czech`, `polish`,
`vietnamese`.

---

## Measured results on the reference corpus

All at `--quality medium --device auto`.

| Document | Type | Result |
|---|---|---|
| Receipt photo (JPEG) | faded thermal receipt | 19 lines, 0.98 |
| Receipt photo (JPEG) | EFTPOS receipt | 27 lines, **1.00** |
| Poster photo (JPEG) | stylised poster | 11 lines, 0.94 |
| Leaflet scan (JPEG) | rotated multi-column insert | 54 lines, 0.99 |
| FOIA release (PDF) | 1956 redacted FOIA scan | 216 lines / 6 pages, 0.94 |
| Handwritten notes (PDF) | pencil handwriting | 130 lines / 15 pages, 0.93 |
| *Muna Madan* scan (PDF) | Nepali book | 1242 lines / 51 pages, 99.4% Devanagari |

---

## First run — fetching weights

```bash
paddleknife models list
paddleknife models pull --all --quality medium
```

`models list` prints tiers and scripts separately, since the detector is
shared across scripts. Weights download on demand during a normal `ocr` run
too, so `models pull` is only needed to pre-warm, or for a machine that
will run offline. The orientation classifiers behind
`--detect-page-orientation` and `--detect-line-orientation` also download
the first time their flag is used; `models pull --orientation` fetches them
ahead of time, and `--all` includes them.

To keep the download small, pull one script and one tier:

```bash
paddleknife models pull english --quality small
```

---

## Memory — "Not enough memory resources" on a GPU with VRAM to spare

You should not need any of this: PaddleKnife budgets recognizer memory and
retries pages that run out. This section is for understanding what it did.

**What the recognizer costs.** A recognition batch is zero-padded to its widest
line, so its memory is `crops × width`, not `crops`. The CTC head alone is
`crops × (width / 8) × vocabulary × 4 bytes`, held about three times over, and
PP-OCRv6's vocabulary is 18,710. One 50:1 line (a chat header listing every
member of a group; a long table row) used to drag a whole 47-line page up to a
2,480 px tensor: ≥ 9 GiB. Batches are now bounded by padded area as well as by
count, so the same page peaks at 3.6 GiB and narrow-line pages batch exactly as
before.

Measured whole-process GPU memory (detector + recognizer + weights) on that
document, DirectML, PP-OCRv6 medium. About 0.85 GiB of it is a fixed floor, so
it does not halve with the batch:

| `--region-batch-size` | peak GPU memory |
|---|---|
| 64 (default) | 3.6 GiB |
| 32 | 2.2 GiB |
| 16 | 1.5 GiB — a sensible start for a 4 GB card |
| 8 | 1.2 GiB |

**What happens when a page still runs out.** It is retried at half the batch,
then at 8, then on CPU — for that page only; the next page starts from the top
again. After two such pages the run stays on the size that worked and prints
the `--region-batch-size` that would have skipped the failed attempts. CPU is
~2.4× slower and is never adopted for a whole run. A PDF page that fails even
there is written without text and listed at the end; an image file that fails
is listed as a failed file.

**The limit is often not VRAM.** On Windows every GPU allocation must also be
backed by *system commit* (RAM + pagefile). The failure that prompted all this
happened on a 16 GB card with 12 GB free — and 6.8 GiB of commit left, because
everything else on the machine had 93 % of it. ONNX Runtime reports that as
`8007000E Not enough memory resources are available`, which reads like a VRAM
problem and is not. PaddleKnife now says so when it sees it. Check with:

```powershell
Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit'
```

If the two numbers are close, close applications or enlarge the pagefile; no
GPU setting will help.

**A fidelity note.** Padding perturbs a CTC recognizer slightly — mostly dropped
spaces (`+61 491 570 006` → `+61491570006`). On the 9-page screenshot document,
text similarity to the unbatched CPU reading was 0.992 at the default batch and
0.9985 at `--region-batch-size 8`, for the same wall time. One document is not
a law, but where exact strings matter it is worth a try.

---

## What does *not* help

**`--region-ref-width 0`.** It turns the padded-area budget off and restores
count-only batching — the configuration that runs out of memory on pages mixing
short and very long lines. It exists for benchmarking and for reproducing that
failure; the retry ladder will rescue the run, slowly.

**`--detect-line-orientation` on sideways text.** The model behind it,
`pp-lcnet_x1_0_textline_ori.onnx`, takes 80×160 input and emits **2 classes**
— it distinguishes upright from upside-down and nothing else. It cannot
detect 90° rotation. On a rotated multi-column insert it slightly *reduced*
recovered text. Use it for flipped scans, not sideways ones.
(`--detect-page-orientation` genuinely is 4-class; that one does handle 90°.)

**Downgrading `--quality` for speed on a GPU.** All three tiers measured the
same wall time. You are giving up accuracy for nothing.

**Raising the detector's input cap.** oar-ocr's default is
`limit_side_len=960, limit_type=Max`, which downscales large scans. It is
overridable via `OAROCRBuilder::text_detection_config()`, but a sweep from
960 → 3200 on a 4894×6991 receipt found the **same 21 regions at every
setting** — only confidence moved (0.858 → 0.898). No evidence it is costing
recall on this corpus, so it is not exposed as a flag. Revisit if a page
turns up where it demonstrably helps.

**`--script latin` for Western European text.** PP-OCRv6's dictionary
covers German, French, Spanish, Italian, Portuguese, Dutch, Polish, Czech,
Slovak, Hungarian, Romanian, Turkish and the Nordic set. `--script auto`
keeps PP-OCRv6 for lightly accented text, but moves a page that is heavy in
accented letters (over 6 % of its Latin letters: typical of Czech, Polish,
Slovak or Hungarian) to the v5 `latin` model. Pin `--script latin-cjk` to
keep PP-OCRv6 for those languages. Pin `--script latin` for
**Vietnamese** (PP-OCRv6 is missing 46 tonal characters) or when you need
typographic punctuation `‐ ‑ ‒ „ ‹ › ‼ №` and Spanish `¡`, which the v5
Latin dictionary has and v6 dropped.

---

## Evaluation tooling

`dev/` holds the harnesses used to produce the numbers above. Their Python
dependencies are in `dev/requirements.txt`; `dev/README.md` has the details.

```bash
python dev/compare_ocr.py page.png --preset v5-mobile --preset v6-medium
python dev/degrade_bench.py book.pdf --start-page 25
python dev/eval_box_accuracy.py book_paddleknife_OCR.pdf --pages 3-49
```

- `ppocr_onnx.py` — reference DBNet + CTC pipeline with every threshold exposed
- `ppocr_viz.py` — reimplementation of PaddleX's two-panel comparison renderer
- `compare_ocr.py` — side-by-side model comparison, writes visualisations
- `degrade_bench.py` — blur / low-res / JPEG / noise / fade / skew robustness sweep
- `eval_box_accuracy.py` — scores any searchable PDF's text layer against the
  printed ink (ink recall, box precision), whichever tool produced it
