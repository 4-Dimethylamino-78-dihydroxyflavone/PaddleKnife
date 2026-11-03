//! Failure-tolerant page OCR (Plan 10).
//!
//! [`OcrEngine`] runs a page once and reports what happened.
//! [`ResilientEngine`] wraps it with the two recoveries a long unattended
//! run needs:
//!
//! 1. **A batch-size ladder.** Running out of GPU memory is an *expected*
//!    failure under dynamic tensor shapes, and the remedy is mechanical:
//!    retry with a smaller recognizer batch. The ladder is
//!    `base → base/2 → 8 → CPU`. The padded-area budget scales with the batch
//!    size, so each rung really does use proportionally less memory.
//! 2. **An orientation check.** The page-orientation classifier finds the
//!    text axis but frequently picks the wrong direction on anything that
//!    isn't a scanned document — with scores as high as its correct
//!    verdicts. Recognition confidence is the reliable judge: read the wrong
//!    way up, Devanagari book pages score 0.48–0.52 against 0.85–0.91 the
//!    right way, chat screenshots 0.62–0.70 against ~0.97. So a poorly-read
//!    page is re-read flipped 180°, and the flipped read is kept only if it
//!    reads the page's characters more confidently *and* recovers most of
//!    them.
//!
//! Both cost nothing on a healthy page.

use crate::{
    error::{OcrError, Result},
    model::ModelPaths,
    ocr::{
        OarOcrBackend, OcrBlock, OcrDevice, OcrEngine, OcrTuning, PageOcr, PageOptions,
        PageOrientation, ReadQuality,
    },
};
use image::DynamicImage;

/// The smallest GPU batch tried before giving up on the GPU for a page.
/// With the default area budget this rung peaks near 0.5 GiB.
pub const FLOOR_REGION_BATCH: usize = 8;

/// Every page starts from the top of the ladder ("per-page reset"), so one
/// freak page in a 1000-page book costs one retry and nothing more. But a
/// *systematically* heavy document — or a small GPU — would otherwise pay a
/// failed attempt plus a session rebuild on every single page. After this
/// many pages that a smaller GPU batch rescued, the run stays on the rung
/// that worked. A page counts once however many reads it took; a page only
/// CPU could rescue does not count, because it says nothing about which
/// batch the GPU can manage.
pub const STICKY_AFTER_DEGRADED_PAGES: usize = 2;

/// Pages whose mean recognition confidence falls below this are re-read
/// flipped 180° when page-orientation detection is on. Measured, read the
/// right way up: Devanagari book pages 0.85–0.91, chat screenshots ~0.97;
/// upside-down: 0.48–0.52 and 0.62–0.70. Genuinely poor scans that land
/// below the bar cost one extra pass and keep their original result.
pub const ORIENTATION_RECHECK_BELOW: f32 = 0.80;

/// The flipped read's per-character confidence must beat the original's by
/// this much to replace it, so noise between two equally bad reads never
/// flips a page.
pub const ORIENTATION_SWITCH_MARGIN: f32 = 0.05;

/// ...and it must recover at least this fraction of the original's
/// recognised characters. Two confident fragments have not read the page
/// better than forty lines at slightly lower confidence, and taking them
/// would strip the page's text layer.
pub const ORIENTATION_MIN_TEXT_RATIO: f32 = 0.5;

/// One way of attempting a page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rung {
    pub device: OcrDevice,
    /// Recognizer batch for this attempt; `None` = as the engine was built.
    pub region_batch: Option<usize>,
}

/// The attempts to make for one page, in order.
///
/// On an accelerator: the configured batch, then half of it, then
/// [`FLOOR_REGION_BATCH`], then CPU as the last resort — CPU is ~2.4× slower,
/// which is never worth it for a whole document but always worth it for the
/// one page that would otherwise be lost. Rungs that would not be strictly
/// smaller than the one before are skipped (`16 → 8 → CPU`, `8 → 4 → CPU`).
///
/// On CPU there is nothing to fall back to: a single rung.
pub fn batch_ladder(device: OcrDevice, base_batch: Option<usize>) -> Vec<Rung> {
    let Some(base) = base_batch.filter(|_| device.is_accelerated()) else {
        return vec![Rung {
            device,
            region_batch: base_batch,
        }];
    };

    let mut batches = vec![base.max(1)];
    for candidate in [base / 2, FLOOR_REGION_BATCH] {
        if candidate >= 1 && batches.last().is_some_and(|&last| candidate < last) {
            batches.push(candidate);
        }
    }

    batches
        .into_iter()
        .map(|b| Rung {
            device,
            region_batch: Some(b),
        })
        .chain(std::iter::once(Rung {
            device: OcrDevice::Cpu,
            region_batch: None,
        }))
        .collect()
}

/// What the recoveries did over a run — for the end-of-run summary.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResilienceReport {
    /// Pages that failed at least once and succeeded on a lower rung. A page
    /// counts once, however many of its reads had to step down.
    pub retried_pages: usize,
    /// Of those, pages that had to fall back to CPU.
    pub cpu_pages: usize,
    /// Of the retried pages, those whose every failure read as the runtime
    /// running out of memory ([`OcrError::is_out_of_memory`]). It only picks
    /// the summary's wording; every engine failure is retried alike.
    pub out_of_memory_pages: usize,
    /// Pages whose 180°-flipped read replaced the classifier's orientation.
    pub reoriented_pages: usize,
    /// Set once the run stopped resetting per page and stayed on this batch.
    pub sticky_batch: Option<usize>,
}

impl ResilienceReport {
    pub fn is_quiet(&self) -> bool {
        *self == Self::default()
    }

    /// What the retried pages ran into, for the end-of-run summary:
    /// `"exceeded GPU memory"` when every failure behind a retry read as an
    /// allocation failure, `"hit GPU errors"` otherwise. The test is a guess
    /// from message text, so the specific claim is made only when it holds
    /// for all of them.
    pub fn retry_cause(&self) -> &'static str {
        if self.out_of_memory_pages == self.retried_pages {
            "exceeded GPU memory"
        } else {
            "hit GPU errors"
        }
    }

    /// Add `other`'s counts to this report, for a run that went through
    /// several engines. The sticky batch keeps the smaller of the two: it is
    /// advice for the next run, and the smaller batch is the one that held
    /// up.
    pub fn absorb(&mut self, other: &Self) {
        // Destructured so a field added later can't compile unmerged.
        let Self {
            retried_pages,
            cpu_pages,
            out_of_memory_pages,
            reoriented_pages,
            sticky_batch,
        } = *other;
        self.retried_pages += retried_pages;
        self.cpu_pages += cpu_pages;
        self.out_of_memory_pages += out_of_memory_pages;
        self.reoriented_pages += reoriented_pages;
        self.sticky_batch = match (self.sticky_batch, sticky_batch) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
    }
}

/// Builds an engine for a device. Injected so tests can stand in a fake.
pub type EngineFactory = Box<dyn FnMut(OcrDevice, OcrTuning) -> Result<OcrEngine> + Send>;

/// An [`OcrEngine`] that retries failed pages down a [`batch_ladder`] and
/// double-checks page orientation. See the module docs.
pub struct ResilientEngine {
    factory: EngineFactory,
    primary: OcrDevice,
    tuning: OcrTuning,
    ladder: Vec<Rung>,
    /// The live session and the device it was built for. At most one ever
    /// exists: two sessions would hold GPU memory simultaneously.
    engine: Option<(OcrDevice, OcrEngine)>,
    /// Ladder index every page starts from. 0 until the run turns sticky.
    floor: usize,
    /// Pages a smaller GPU batch rescued; see [`STICKY_AFTER_DEGRADED_PAGES`].
    degraded_pages: usize,
    low_commit_reported: bool,
    report: ResilienceReport,
}

impl ResilientEngine {
    /// Build for the given models. The primary engine is created eagerly so
    /// a missing model or a broken GPU runtime surfaces here, not on page 1.
    pub fn from_paths(paths: &ModelPaths, device: OcrDevice, tuning: OcrTuning) -> Result<Self> {
        let paths = paths.clone();
        Self::with_factory(
            Box::new(move |device, tuning| OcrEngine::from_paths_tuned(&paths, device, tuning)),
            device,
            tuning,
        )
    }

    /// Build with a caller-supplied engine factory.
    pub fn with_factory(
        mut factory: EngineFactory,
        device: OcrDevice,
        tuning: OcrTuning,
    ) -> Result<Self> {
        let engine = factory(device, tuning)?;
        Ok(Self {
            factory,
            primary: device,
            tuning,
            ladder: batch_ladder(
                device,
                OarOcrBackend::effective_region_batch(device, &tuning),
            ),
            engine: Some((device, engine)),
            floor: 0,
            degraded_pages: 0,
            low_commit_reported: false,
            report: ResilienceReport::default(),
        })
    }

    pub fn report(&self) -> &ResilienceReport {
        &self.report
    }

    /// Forget what earlier reads taught the recoveries — the report, the
    /// degraded-page count and the sticky floor — while keeping the loaded
    /// session. For an engine that served a sampling pass (script
    /// auto-detect) and now serves a run that reads the same pages again:
    /// a page counts once, so its sample read must not count toward the
    /// summary or toward turning the run sticky. The run then starts
    /// exactly as a freshly built engine would, minus the session init.
    pub fn reset_recovery_state(&mut self) {
        self.floor = 0;
        self.degraded_pages = 0;
        self.report = ResilienceReport::default();
    }

    /// OCR one page, recovering from engine failures and mis-orientation.
    pub fn ocr(&mut self, img: &DynamicImage) -> Result<Vec<OcrBlock>> {
        Ok(self.ocr_page(img)?.blocks)
    }

    /// As [`Self::ocr`], also returning the page orientation that was used.
    pub fn ocr_page(&mut self, img: &DynamicImage) -> Result<PageOcr> {
        let mut trace = PageTrace::default();
        let page = self.read_page(img, &mut trace);
        self.settle(&trace);
        page
    }

    /// The first read, then the orientation re-check when it is called for.
    fn read_page(&mut self, img: &DynamicImage, trace: &mut PageTrace) -> Result<PageOcr> {
        let (first, first_idx) = self.climb(img, PageOrientation::Auto, self.floor, trace)?;
        if !self.tuning.detect_page_orientation {
            return Ok(first);
        }
        // A page that yielded nothing counts as confidence 0, not as "fine":
        // text read upside-down can come back entirely empty, and that must
        // not be mistaken for a blank page. A page that really is blank
        // costs one cheap extra pass (detection only) and stays blank.
        if first.mean_confidence().unwrap_or(0.0) >= ORIENTATION_RECHECK_BELOW {
            return Ok(first);
        }

        // Every misjudgement we have measured is an exact 180° flip (upright
        // called 180°, 90° called 270°), so the opposite of whatever was
        // used is the one alternative worth a second pass.
        let flipped_turns = (quarter_turns(first.page_angle_deg) + 2) % 4;
        // Start on the rung the first read succeeded on, not the floor:
        // anything above it has just failed on this very page, and the
        // flipped read recognises the same lines at the same widths.
        let orientation = PageOrientation::Forced(flipped_turns);
        let flipped = match self.climb(img, orientation, first_idx, trace) {
            Ok((page, _)) => page,
            Err(e) => {
                tracing::warn!(error = %e, "orientation re-check failed; keeping first read");
                return Ok(first);
            }
        };
        let before = first.read_quality();
        let after = flipped.read_quality();
        if flipped_read_wins(before, after) {
            let used_deg = u16::from(flipped_turns) * 90;
            tracing::info!(
                classifier_deg = ?first.page_angle_deg,
                used_deg,
                conf_before = before.char_confidence(),
                conf_after = after.char_confidence(),
                chars_before = before.graphemes,
                chars_after = after.graphemes,
                "page orientation corrected by recognition confidence"
            );
            self.report.reoriented_pages += 1;
            Ok(flipped)
        } else {
            Ok(first)
        }
    }

    /// Try the page on each rung from ladder index `start` down. On success,
    /// the page and the index of the rung that read it.
    fn climb(
        &mut self,
        img: &DynamicImage,
        orientation: PageOrientation,
        start: usize,
        trace: &mut PageTrace,
    ) -> Result<(PageOcr, usize)> {
        let mut idx = start;
        loop {
            let rung = self.ladder[idx];
            let err = match self.attempt(rung, img, orientation) {
                Ok(page) => {
                    trace.deepest_ok = trace.deepest_ok.max(Some(idx));
                    if rung.device == self.primary {
                        trace.deepest_gpu_ok = trace.deepest_gpu_ok.max(Some(idx));
                    }
                    return Ok((page, idx));
                }
                Err(e) => e,
            };

            let next = self
                .ladder
                .get(idx + 1)
                .copied()
                .filter(|_| is_retryable(&err));
            // Read commit headroom *before* freeing the failed session: its
            // cached buffers are part of the squeeze the allocation failed
            // in, and freeing them first makes the machine look roomier than
            // it was when it mattered. Only for an allocation failure: low
            // commit explains nothing else, and naming it for a device error
            // would send people to the wrong fix (and spend the one-shot
            // hint before a real out-of-memory could use it).
            let commit_hint = match next {
                Some(_) if !self.low_commit_reported && err.is_out_of_memory() => {
                    crate::sysmem::low_commit_hint()
                }
                _ => None,
            };

            // A failed run leaves its session holding whatever it allocated
            // on the way to failing: ONNX Runtime's DirectML allocator caches
            // freed buffers per session and never returns them. Retrying in
            // the same session would start several GiB in the hole, so drop
            // it and let the next rung rebuild (~1–2 s).
            self.engine = None;

            let Some(next) = next else {
                return Err(err);
            };
            trace.retries += 1;
            if err.is_out_of_memory() {
                trace.oom_retries += 1;
            }
            tracing::warn!(
                error = %err,
                failed = %describe(rung),
                retrying = %describe(next),
                "page failed; retrying on a lower rung"
            );
            // Say so once when the machine, not the GPU, is what's full: the
            // ORT error reads like a VRAM problem and sends people looking
            // in the wrong place.
            if let Some(hint) = commit_hint {
                self.low_commit_reported = true;
                tracing::warn!("likely cause: {hint}");
            }
            idx += 1;
        }
    }

    fn attempt(
        &mut self,
        rung: Rung,
        img: &DynamicImage,
        orientation: PageOrientation,
    ) -> Result<PageOcr> {
        let engine = match &mut self.engine {
            Some((device, engine)) if *device == rung.device => engine,
            slot => {
                // Free the outgoing session *before* building its successor.
                *slot = None;
                let tuning = if rung.device == self.primary {
                    self.tuning
                } else {
                    // Fallback device: its own defaults, not batch sizes
                    // that were chosen for the GPU.
                    OcrTuning {
                        region_batch_size: None,
                        region_ref_width: None,
                        ..self.tuning
                    }
                };
                let engine = (self.factory)(rung.device, tuning)?;
                &mut slot.insert((rung.device, engine)).1
            }
        };
        engine.ocr_page(
            img,
            &PageOptions {
                region_batch_size: rung.region_batch,
                orientation,
            },
        )
    }

    /// Book-keeping once a page is done, however many reads it took.
    fn settle(&mut self, trace: &PageTrace) {
        // CPU is for one page only. Dropping its session now returns its RAM
        // and makes the next page rebuild the GPU engine.
        if self
            .engine
            .as_ref()
            .is_some_and(|(device, _)| *device != self.primary)
        {
            self.engine = None;
        }

        let Some(deepest) = trace.deepest_ok else {
            return; // a lost page is not a recovered page
        };
        if deepest <= self.floor {
            return; // every read worked first time — nothing was degraded
        }
        self.report.retried_pages += 1;
        if self.ladder[deepest].device != self.primary {
            self.report.cpu_pages += 1;
        }
        if trace.oom_retries == trace.retries {
            self.report.out_of_memory_pages += 1;
        }

        // Only a smaller GPU batch that *worked* says anything about the
        // batch this run can afford. A page only CPU could rescue doesn't
        // count, which also keeps CPU from ever becoming the floor: the last
        // accelerated rung is as low as a whole run is allowed to sink.
        let Some(worked) = trace.deepest_gpu_ok.filter(|&idx| idx > self.floor) else {
            return;
        };
        self.degraded_pages += 1;
        if self.degraded_pages < STICKY_AFTER_DEGRADED_PAGES {
            return;
        }
        self.floor = worked;
        self.report.sticky_batch = self.ladder[worked].region_batch;
        tracing::warn!(
            now = %describe(self.ladder[worked]),
            "{} pages {} at a larger batch; staying on the smaller batch for the rest of the \
             run (pass --region-batch-size to start there)",
            self.degraded_pages,
            self.report.retry_cause()
        );
    }
}

/// What one page's reads went through. Settled once the page is done, so a
/// page counts once however many reads — first try, orientation re-check —
/// it took.
#[derive(Debug, Default)]
struct PageTrace {
    /// Deepest ladder index any read of the page succeeded on.
    deepest_ok: Option<usize>,
    /// The same, counting only rungs on the primary (accelerated) device.
    deepest_gpu_ok: Option<usize>,
    /// Failures that sent a read down a rung...
    retries: usize,
    /// ...and how many of them read as out-of-memory.
    oom_retries: usize,
}

/// Only engine-level failures are worth retrying differently; a missing
/// model or an unreadable image fails the same way on every rung.
///
/// Deliberately broad: whether an engine error was an allocation failure
/// can only be guessed from its message ([`OcrError::is_out_of_memory`]),
/// which is good enough to word a summary, not to give up on a page.
fn is_retryable(err: &OcrError) -> bool {
    matches!(err, OcrError::Engine(_))
}

/// Whether a page's 180°-flipped re-read should replace its first read.
///
/// Both reads are weighed by recognised characters, not blocks: the flipped
/// read must read its characters more confidently, by
/// [`ORIENTATION_SWITCH_MARGIN`], *and* recover at least
/// [`ORIENTATION_MIN_TEXT_RATIO`] of the first read's characters. A per-block
/// mean would let two confident fragments beat forty correctly read lines.
fn flipped_read_wins(first: ReadQuality, flipped: ReadQuality) -> bool {
    flipped.char_confidence() > first.char_confidence() + ORIENTATION_SWITCH_MARGIN
        && flipped.graphemes as f32 >= first.graphemes as f32 * ORIENTATION_MIN_TEXT_RATIO
}

fn describe(rung: Rung) -> String {
    match rung.region_batch {
        Some(b) => format!("{} (region batch {b})", rung.device.label()),
        None => rung.device.label().to_string(),
    }
}

/// Classifier angle → quarter turns; "no verdict" counts as upright.
fn quarter_turns(angle_deg: Option<f32>) -> u8 {
    let turns = (angle_deg.unwrap_or(0.0) / 90.0).round() as i32;
    turns.rem_euclid(4) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ocr::OcrBackend;
    use std::sync::{Arc, Mutex};

    const GPU: OcrDevice = OcrDevice::DirectMl;

    fn gpu(batch: usize) -> Rung {
        Rung {
            device: GPU,
            region_batch: Some(batch),
        }
    }
    const CPU_RUNG: Rung = Rung {
        device: OcrDevice::Cpu,
        region_batch: None,
    };

    #[test]
    fn ladder_halves_then_jumps_to_eight_then_cpu() {
        assert_eq!(
            batch_ladder(GPU, Some(64)),
            vec![gpu(64), gpu(32), gpu(8), CPU_RUNG]
        );
        assert_eq!(
            batch_ladder(GPU, Some(128)),
            vec![gpu(128), gpu(64), gpu(8), CPU_RUNG]
        );
    }

    #[test]
    fn ladder_skips_rungs_that_would_not_shrink() {
        assert_eq!(batch_ladder(GPU, Some(16)), vec![gpu(16), gpu(8), CPU_RUNG]);
        assert_eq!(batch_ladder(GPU, Some(12)), vec![gpu(12), gpu(6), CPU_RUNG]);
        assert_eq!(batch_ladder(GPU, Some(8)), vec![gpu(8), gpu(4), CPU_RUNG]);
        assert_eq!(batch_ladder(GPU, Some(1)), vec![gpu(1), CPU_RUNG]);
    }

    #[test]
    fn cpu_has_nothing_to_fall_back_to() {
        let only = Rung {
            device: OcrDevice::Cpu,
            region_batch: Some(4),
        };
        assert_eq!(batch_ladder(OcrDevice::Cpu, Some(4)), vec![only]);
        assert_eq!(batch_ladder(OcrDevice::Cpu, None), vec![CPU_RUNG]);
    }

    #[test]
    fn quarter_turns_rounds_and_wraps() {
        assert_eq!(quarter_turns(None), 0);
        assert_eq!(quarter_turns(Some(0.0)), 0);
        assert_eq!(quarter_turns(Some(90.0)), 1);
        assert_eq!(quarter_turns(Some(180.0)), 2);
        assert_eq!(quarter_turns(Some(270.0)), 3);
        assert_eq!(quarter_turns(Some(360.0)), 0);
    }

    #[test]
    fn flip_needs_more_confident_characters_and_most_of_the_text() {
        let read = |graphemes: usize, confidence: f32| ReadQuality {
            graphemes,
            score: graphemes as f32 * confidence,
        };
        // Forty 4-letter lines at 0.72 against two fragments at 0.9: the
        // fragments are more confident, and a sliver of the page.
        assert!(!flipped_read_wins(read(160, 0.72), read(8, 0.9)));
        // Genuinely upside-down: low-confidence garbage one way, text the other.
        assert!(flipped_read_wins(read(160, 0.50), read(150, 0.88)));
        // Exactly half the text is still enough.
        assert!(flipped_read_wins(read(100, 0.50), read(50, 0.90)));
        // Within the margin is noise, not evidence.
        assert!(!flipped_read_wins(read(100, 0.55), read(100, 0.58)));
        // An empty first read loses to any confident text; two blank reads
        // never flip.
        assert!(flipped_read_wins(ReadQuality::default(), read(4, 0.9)));
        assert!(!flipped_read_wins(
            ReadQuality::default(),
            ReadQuality::default()
        ));
    }

    #[test]
    fn retry_cause_names_memory_only_when_every_retried_page_ran_out() {
        let report = |retried_pages: usize, out_of_memory_pages: usize| ResilienceReport {
            retried_pages,
            out_of_memory_pages,
            ..ResilienceReport::default()
        };
        assert_eq!(report(3, 3).retry_cause(), "exceeded GPU memory");
        assert_eq!(report(3, 2).retry_cause(), "hit GPU errors");
        assert_eq!(report(1, 0).retry_cause(), "hit GPU errors");
    }

    #[test]
    fn absorb_sums_counts_and_keeps_the_smaller_sticky_batch() {
        let mut total = ResilienceReport {
            retried_pages: 2,
            cpu_pages: 1,
            out_of_memory_pages: 2,
            sticky_batch: Some(32),
            ..ResilienceReport::default()
        };
        total.absorb(&ResilienceReport {
            retried_pages: 3,
            out_of_memory_pages: 1,
            reoriented_pages: 4,
            sticky_batch: Some(8),
            ..ResilienceReport::default()
        });
        assert_eq!(total.retried_pages, 5);
        assert_eq!(total.cpu_pages, 1);
        assert_eq!(total.out_of_memory_pages, 3);
        assert_eq!(total.reoriented_pages, 4);
        assert_eq!(total.sticky_batch, Some(8));
        // The cause wording follows the summed counts: one engine's non-OOM
        // retries mean the run as a whole can't claim "exceeded GPU memory".
        assert_eq!(total.retry_cause(), "hit GPU errors");

        // A quiet engine changes nothing; a sticky batch survives a swap
        // to an engine that never needed one.
        total.absorb(&ResilienceReport::default());
        assert_eq!(total.retried_pages, 5);
        assert_eq!(total.sticky_batch, Some(8));
        let mut fresh = ResilienceReport::default();
        fresh.absorb(&total);
        assert_eq!(fresh, total);
    }

    // ---- a scriptable fake engine ----

    #[derive(Debug, Clone, PartialEq)]
    enum Event {
        Build(OcrDevice),
        Page(OcrDevice, Option<usize>, PageOrientation),
    }

    /// Shared, mutable description of "the hardware and the page".
    struct World {
        events: Vec<Event>,
        /// Largest region batch the GPU survives; larger → engine error.
        gpu_max_batch: usize,
        /// Largest batch a forced-orientation read survives, capped by
        /// `gpu_max_batch` too: a flipped page can be heavier still.
        forced_max_batch: usize,
        cpu_works: bool,
        /// What a failed attempt says.
        engine_error: &'static str,
        /// How the page really is rotated, and what the classifier claims.
        true_turns: u8,
        classifier_turns: u8,
        /// Confidence when read the right way up / any other way.
        conf_right: f32,
        conf_wrong: f32,
        /// Blocks (4 letters each) found the right way up / any other way.
        right_blocks: usize,
        wrong_blocks: usize,
        blank: bool,
    }

    impl World {
        fn new() -> Arc<Mutex<World>> {
            Arc::new(Mutex::new(World {
                events: Vec::new(),
                gpu_max_batch: usize::MAX,
                forced_max_batch: usize::MAX,
                cpu_works: true,
                engine_error: "adapter execution failed",
                true_turns: 0,
                classifier_turns: 0,
                conf_right: 0.97,
                conf_wrong: 0.65,
                right_blocks: 1,
                wrong_blocks: 1,
                blank: false,
            }))
        }
    }

    struct FakeBackend {
        device: OcrDevice,
        world: Arc<Mutex<World>>,
    }

    impl OcrBackend for FakeBackend {
        fn ocr(&mut self, img: &DynamicImage) -> Result<Vec<OcrBlock>> {
            Ok(self.ocr_page(img, &PageOptions::default())?.blocks)
        }

        fn ocr_page(&mut self, _img: &DynamicImage, opts: &PageOptions) -> Result<PageOcr> {
            let mut w = self.world.lock().expect("world lock");
            w.events.push(Event::Page(
                self.device,
                opts.region_batch_size,
                opts.orientation,
            ));
            let gpu_limit = match opts.orientation {
                PageOrientation::Auto => w.gpu_max_batch,
                PageOrientation::Forced(_) => w.gpu_max_batch.min(w.forced_max_batch),
            };
            let ok = match self.device {
                OcrDevice::Cpu => w.cpu_works,
                _ => opts.region_batch_size.unwrap_or(64) <= gpu_limit,
            };
            if !ok {
                return Err(OcrError::Engine(w.engine_error.into()));
            }
            let used = match opts.orientation {
                PageOrientation::Auto => w.classifier_turns,
                PageOrientation::Forced(t) => t,
            };
            let (confidence, count) = if used == w.true_turns {
                (w.conf_right, w.right_blocks)
            } else {
                (w.conf_wrong, w.wrong_blocks)
            };
            let blocks = if w.blank {
                Vec::new()
            } else {
                vec![
                    OcrBlock {
                        text: "text".into(),
                        confidence,
                        ..OcrBlock::default()
                    };
                    count
                ]
            };
            Ok(PageOcr {
                blocks,
                page_angle_deg: Some(f32::from(used) * 90.0),
                unread_regions: 0,
            })
        }
    }

    fn engine(world: &Arc<Mutex<World>>, tuning: OcrTuning) -> ResilientEngine {
        let w = Arc::clone(world);
        ResilientEngine::with_factory(
            Box::new(move |device, _tuning| {
                w.lock()
                    .expect("world lock")
                    .events
                    .push(Event::Build(device));
                Ok(OcrEngine::with_backend(Box::new(FakeBackend {
                    device,
                    world: Arc::clone(&w),
                })))
            }),
            GPU,
            tuning,
        )
        .expect("fake engine builds")
    }

    fn page() -> DynamicImage {
        DynamicImage::new_rgb8(4, 4)
    }

    fn take_events(world: &Arc<Mutex<World>>) -> Vec<Event> {
        std::mem::take(&mut world.lock().expect("world lock").events)
    }

    fn try_gpu(batch: usize) -> Event {
        Event::Page(GPU, Some(batch), PageOrientation::Auto)
    }

    #[test]
    fn healthy_page_is_one_attempt_on_the_prebuilt_engine() {
        let world = World::new();
        let mut e = engine(&world, OcrTuning::default());
        e.ocr_page(&page()).expect("page ok");
        assert_eq!(take_events(&world), vec![Event::Build(GPU), try_gpu(64)]);
        assert!(e.report().is_quiet());
    }

    #[test]
    fn failed_page_retries_at_half_batch_on_a_fresh_session() {
        let world = World::new();
        world.lock().expect("lock").gpu_max_batch = 32;
        let mut e = engine(&world, OcrTuning::default());
        e.ocr_page(&page()).expect("recovers");
        assert_eq!(
            take_events(&world),
            vec![
                Event::Build(GPU),
                try_gpu(64),
                Event::Build(GPU), // failed session dropped, not reused
                try_gpu(32),
            ]
        );
        assert_eq!(e.report().retried_pages, 1);
        assert_eq!(e.report().cpu_pages, 0);
    }

    #[test]
    fn next_page_starts_from_the_top_again() {
        let world = World::new();
        world.lock().expect("lock").gpu_max_batch = 32;
        let mut e = engine(&world, OcrTuning::default());
        e.ocr_page(&page()).expect("page 1 recovers");
        take_events(&world);

        // Page 2 is an easy one: full batch, and no rebuild to get there.
        world.lock().expect("lock").gpu_max_batch = usize::MAX;
        e.ocr_page(&page()).expect("page 2 ok");
        assert_eq!(take_events(&world), vec![try_gpu(64)]);
        assert_eq!(e.report().sticky_batch, None);
    }

    #[test]
    fn run_turns_sticky_after_repeated_degraded_pages() {
        let world = World::new();
        world.lock().expect("lock").gpu_max_batch = 32;
        let mut e = engine(&world, OcrTuning::default());
        for _ in 0..STICKY_AFTER_DEGRADED_PAGES {
            e.ocr_page(&page()).expect("recovers");
        }
        assert_eq!(e.report().sticky_batch, Some(32));
        take_events(&world);

        // From here on, pages go straight to the batch that works.
        e.ocr_page(&page()).expect("page ok");
        assert_eq!(take_events(&world), vec![try_gpu(32)]);
        assert_eq!(e.report().retried_pages, STICKY_AFTER_DEGRADED_PAGES);
    }

    #[test]
    fn jumps_to_eight_when_half_is_still_too_much() {
        let world = World::new();
        world.lock().expect("lock").gpu_max_batch = 8;
        let mut e = engine(&world, OcrTuning::default());
        e.ocr_page(&page()).expect("recovers");
        let tried: Vec<Event> = take_events(&world)
            .into_iter()
            .filter(|ev| matches!(ev, Event::Page(..)))
            .collect();
        assert_eq!(tried, vec![try_gpu(64), try_gpu(32), try_gpu(8)]);
    }

    #[test]
    fn cpu_is_the_last_resort_and_never_sticks() {
        let world = World::new();
        world.lock().expect("lock").gpu_max_batch = 0; // GPU hopeless
        let mut e = engine(&world, OcrTuning::default());

        e.ocr_page(&page()).expect("page 1 saved by CPU");
        let tried: Vec<Event> = take_events(&world)
            .into_iter()
            .filter(|ev| matches!(ev, Event::Page(..)))
            .collect();
        assert_eq!(
            tried,
            vec![
                try_gpu(64),
                try_gpu(32),
                try_gpu(8),
                Event::Page(OcrDevice::Cpu, None, PageOrientation::Auto),
            ]
        );

        e.ocr_page(&page()).expect("page 2 saved by CPU");
        assert_eq!(e.report().cpu_pages, 2);
        // Not sticky: no smaller GPU batch worked, so two CPU rescues say
        // nothing about which batch the GPU can manage.
        assert_eq!(e.report().sticky_batch, None);
        take_events(&world);

        // The GPU recovers (another app released VRAM): page 3 runs there,
        // from the top.
        world.lock().expect("lock").gpu_max_batch = usize::MAX;
        e.ocr_page(&page()).expect("page 3 ok");
        assert_eq!(take_events(&world), vec![Event::Build(GPU), try_gpu(64)]);
    }

    #[test]
    fn cpu_rescue_does_not_count_toward_sticky() {
        let world = World::new();
        world.lock().expect("lock").gpu_max_batch = 32;
        let mut e = engine(&world, OcrTuning::default());
        e.ocr_page(&page()).expect("page 1 rescued at batch 32");

        world.lock().expect("lock").gpu_max_batch = 0;
        e.ocr_page(&page()).expect("page 2 rescued by CPU");
        assert_eq!(e.report().retried_pages, 2);
        assert_eq!(e.report().cpu_pages, 1);
        assert_eq!(
            e.report().sticky_batch,
            None,
            "one GPU rescue is not a pattern"
        );

        world.lock().expect("lock").gpu_max_batch = 32;
        e.ocr_page(&page()).expect("page 3 rescued at batch 32");
        assert_eq!(e.report().sticky_batch, Some(32));
    }

    #[test]
    fn sticky_run_never_sinks_to_cpu() {
        let world = World::new();
        world.lock().expect("lock").gpu_max_batch = 8;
        let mut e = engine(&world, OcrTuning::default());
        for _ in 0..STICKY_AFTER_DEGRADED_PAGES {
            e.ocr_page(&page()).expect("rescued at batch 8");
        }
        assert_eq!(e.report().sticky_batch, Some(8));

        // A page even batch 8 can't manage goes to CPU for itself only...
        world.lock().expect("lock").gpu_max_batch = 0;
        e.ocr_page(&page()).expect("page saved by CPU");
        assert_eq!(e.report().cpu_pages, 1);
        assert_eq!(e.report().sticky_batch, Some(8));
        take_events(&world);

        // ...and the next page is back on the GPU at the sticky batch.
        world.lock().expect("lock").gpu_max_batch = usize::MAX;
        e.ocr_page(&page()).expect("page ok");
        assert_eq!(take_events(&world), vec![Event::Build(GPU), try_gpu(8)]);
    }

    #[test]
    fn summary_blames_memory_only_when_the_failures_said_so() {
        let world = World::new();
        {
            let mut w = world.lock().expect("lock");
            w.gpu_max_batch = 32;
            w.engine_error = "oar-ocr predict: 8007000E Not enough memory resources are \
                              available to complete this operation.";
        }
        let mut e = engine(&world, OcrTuning::default());
        e.ocr_page(&page()).expect("recovers");
        assert_eq!(e.report().out_of_memory_pages, 1);
        assert_eq!(e.report().retry_cause(), "exceeded GPU memory");

        // Another DirectML failure is retried just the same, but the summary
        // stops claiming memory.
        world.lock().expect("lock").engine_error = "adapter execution failed";
        e.ocr_page(&page()).expect("recovers too");
        assert_eq!(e.report().retried_pages, 2);
        assert_eq!(e.report().out_of_memory_pages, 1);
        assert_eq!(e.report().retry_cause(), "hit GPU errors");
    }

    #[test]
    fn page_that_fails_everywhere_returns_the_error() {
        let world = World::new();
        {
            let mut w = world.lock().expect("lock");
            w.gpu_max_batch = 0;
            w.cpu_works = false;
        }
        let mut e = engine(&world, OcrTuning::default());
        let err = e.ocr_page(&page()).expect_err("nothing works");
        assert!(matches!(err, OcrError::Engine(_)));
        assert!(e.report().is_quiet(), "a lost page is not a recovered page");

        // ...and the engine is still usable for the next page.
        {
            let mut w = world.lock().expect("lock");
            w.gpu_max_batch = usize::MAX;
        }
        e.ocr_page(&page()).expect("next page ok");
    }

    #[test]
    fn construction_failure_is_returned_from_the_constructor() {
        let built = Arc::new(Mutex::new(0usize));
        let counter = Arc::clone(&built);
        let result = ResilientEngine::with_factory(
            Box::new(move |_, _| {
                *counter.lock().expect("lock") += 1;
                Err(OcrError::ModelNotFound("rec.onnx".into()))
            }),
            GPU,
            OcrTuning::default(),
        );
        assert!(matches!(result, Err(OcrError::ModelNotFound(_))));
        assert_eq!(*built.lock().expect("lock"), 1);
    }

    #[test]
    fn non_engine_errors_are_not_retried() {
        // The primary session builds; the page fails at batch 64 (an engine
        // error, retried); rebuilding for batch 32 then fails with a missing
        // model — which fails the same way on every rung, so it must end the
        // page there, with no further builds and no CPU attempt.
        let world = World::new();
        world.lock().expect("lock").gpu_max_batch = 32;
        let w = Arc::clone(&world);
        let mut first = true;
        let mut e = ResilientEngine::with_factory(
            Box::new(move |device, _| {
                w.lock()
                    .expect("world lock")
                    .events
                    .push(Event::Build(device));
                if !std::mem::take(&mut first) {
                    return Err(OcrError::ModelNotFound("rec.onnx".into()));
                }
                Ok(OcrEngine::with_backend(Box::new(FakeBackend {
                    device,
                    world: Arc::clone(&w),
                })))
            }),
            GPU,
            OcrTuning::default(),
        )
        .expect("primary builds");
        let err = e.ocr_page(&page()).expect_err("rebuild fails");
        assert!(matches!(err, OcrError::ModelNotFound(_)), "got {err:?}");
        assert_eq!(
            take_events(&world),
            vec![Event::Build(GPU), try_gpu(64), Event::Build(GPU)]
        );
        assert!(e.report().is_quiet());
    }

    #[test]
    fn reset_recovery_state_makes_a_reread_page_count_once() {
        // Auto-detect's sample read of a heavy page, then the run's own read
        // of the same page on the same engine.
        let world = World::new();
        world.lock().expect("lock").gpu_max_batch = 32;
        let mut e = engine(&world, OcrTuning::default());
        e.ocr_page(&page()).expect("sample read recovers");
        assert_eq!(e.report().retried_pages, 1);

        e.reset_recovery_state();
        assert!(e.report().is_quiet());
        take_events(&world);
        e.ocr_page(&page()).expect("run read recovers");
        // Starts from the top as a fresh engine would, reusing the session...
        assert_eq!(
            take_events(&world),
            vec![try_gpu(64), Event::Build(GPU), try_gpu(32)]
        );
        // ...and one heavy page is one retry, not a sticky run.
        assert_eq!(e.report().retried_pages, 1);
        assert_eq!(e.report().sticky_batch, None);
    }

    fn ori_tuning() -> OcrTuning {
        OcrTuning {
            detect_page_orientation: true,
            ..OcrTuning::default()
        }
    }

    #[test]
    fn upside_down_verdict_is_overruled_by_recognition_confidence() {
        // The WhatsApp-screenshot case: page is upright, classifier says 180°.
        let world = World::new();
        world.lock().expect("lock").classifier_turns = 2;
        let mut e = engine(&world, ori_tuning());
        let got = e.ocr_page(&page()).expect("page ok");
        assert_eq!(got.page_angle_deg, Some(0.0));
        assert!((got.mean_confidence().expect("has text") - 0.97).abs() < 1e-6);
        assert_eq!(e.report().reoriented_pages, 1);
        assert_eq!(
            take_events(&world),
            vec![
                Event::Build(GPU),
                try_gpu(64),
                Event::Page(GPU, Some(64), PageOrientation::Forced(0)),
            ]
        );
    }

    #[test]
    fn sideways_page_with_flipped_verdict_is_recovered() {
        // Truly rotated 90°, classifier says 270°: flip of 3 is 1. And the
        // mirror case, whose 270° correction is logged in degrees.
        for (truth, verdict) in [(1, 3), (3, 1)] {
            let world = World::new();
            {
                let mut w = world.lock().expect("lock");
                w.true_turns = truth;
                w.classifier_turns = verdict;
            }
            let mut e = engine(&world, ori_tuning());
            let got = e.ocr_page(&page()).expect("page ok");
            assert_eq!(got.page_angle_deg, Some(f32::from(truth) * 90.0));
            assert_eq!(e.report().reoriented_pages, 1);
        }
    }

    #[test]
    fn sparse_confident_flip_does_not_replace_a_dense_read() {
        // Upright page, read the right way: 40 blocks at 0.72, under the
        // re-check bar. Flipped, the recognizer finds two fragments at 0.9.
        // Per block the fragments look better; per character they are a
        // sliver of the page, and taking them would strip its text layer.
        let world = World::new();
        {
            let mut w = world.lock().expect("lock");
            w.conf_right = 0.72;
            w.right_blocks = 40;
            w.conf_wrong = 0.9;
            w.wrong_blocks = 2;
        }
        let mut e = engine(&world, ori_tuning());
        let got = e.ocr_page(&page()).expect("page ok");
        assert_eq!(got.page_angle_deg, Some(0.0));
        assert_eq!(got.blocks.len(), 40);
        assert_eq!(e.report().reoriented_pages, 0);
        // It was re-checked: the decision kept it, not the trigger.
        assert_eq!(
            take_events(&world),
            vec![
                Event::Build(GPU),
                try_gpu(64),
                Event::Page(GPU, Some(64), PageOrientation::Forced(2)),
            ]
        );
    }

    #[test]
    fn dense_upside_down_page_still_flips() {
        // Truly upside-down, classifier calls it upright: forty lines of
        // garbage at 0.50 one way, forty lines of text at 0.88 the other.
        let world = World::new();
        {
            let mut w = world.lock().expect("lock");
            w.true_turns = 2;
            w.conf_right = 0.88;
            w.conf_wrong = 0.50;
            w.right_blocks = 40;
            w.wrong_blocks = 40;
        }
        let mut e = engine(&world, ori_tuning());
        let got = e.ocr_page(&page()).expect("page ok");
        assert_eq!(got.page_angle_deg, Some(180.0));
        assert_eq!(got.blocks.len(), 40);
        assert!(got
            .blocks
            .iter()
            .all(|b| (b.confidence - 0.88).abs() < 1e-6));
        assert_eq!(e.report().reoriented_pages, 1);
    }

    #[test]
    fn confident_read_is_never_second_guessed() {
        let world = World::new();
        {
            let mut w = world.lock().expect("lock");
            w.true_turns = 2;
            w.classifier_turns = 2; // classifier right: a real upside-down scan
        }
        let mut e = engine(&world, ori_tuning());
        e.ocr_page(&page()).expect("page ok");
        assert_eq!(take_events(&world), vec![Event::Build(GPU), try_gpu(64)]);
        assert_eq!(e.report().reoriented_pages, 0);
    }

    #[test]
    fn genuinely_poor_page_keeps_its_first_read() {
        let world = World::new();
        {
            let mut w = world.lock().expect("lock");
            w.conf_right = 0.55; // bad scan: poor either way up
            w.conf_wrong = 0.52;
        }
        let mut e = engine(&world, ori_tuning());
        let got = e.ocr_page(&page()).expect("page ok");
        assert_eq!(got.page_angle_deg, Some(0.0));
        assert_eq!(e.report().reoriented_pages, 0);
    }

    #[test]
    fn page_that_reads_as_empty_upside_down_is_not_taken_for_blank() {
        // Upside-down text can come back with no blocks at all. That is
        // confidence zero, not a blank page.
        struct EmptyWhenWrong(Arc<Mutex<World>>);
        impl OcrBackend for EmptyWhenWrong {
            fn ocr(&mut self, img: &DynamicImage) -> Result<Vec<OcrBlock>> {
                Ok(self.ocr_page(img, &PageOptions::default())?.blocks)
            }
            fn ocr_page(&mut self, _img: &DynamicImage, opts: &PageOptions) -> Result<PageOcr> {
                let w = self.0.lock().expect("world lock");
                let used = match opts.orientation {
                    PageOrientation::Auto => w.classifier_turns,
                    PageOrientation::Forced(t) => t,
                };
                let blocks = if used == w.true_turns {
                    vec![OcrBlock {
                        text: "text".into(),
                        confidence: 0.9,
                        ..OcrBlock::default()
                    }]
                } else {
                    Vec::new()
                };
                Ok(PageOcr {
                    blocks,
                    page_angle_deg: Some(f32::from(used) * 90.0),
                    unread_regions: 0,
                })
            }
        }

        let world = World::new();
        world.lock().expect("lock").classifier_turns = 2;
        let w = Arc::clone(&world);
        let mut e = ResilientEngine::with_factory(
            Box::new(move |_, _| {
                Ok(OcrEngine::with_backend(Box::new(EmptyWhenWrong(
                    Arc::clone(&w),
                ))))
            }),
            GPU,
            ori_tuning(),
        )
        .expect("builds");
        let got = e.ocr_page(&page()).expect("page ok");
        assert_eq!(got.blocks.len(), 1);
        assert_eq!(got.page_angle_deg, Some(0.0));
        assert_eq!(e.report().reoriented_pages, 1);
    }

    #[test]
    fn blank_page_costs_one_extra_pass_and_stays_blank() {
        let world = World::new();
        {
            let mut w = world.lock().expect("lock");
            w.classifier_turns = 2;
            w.blank = true;
        }
        let mut e = engine(&world, ori_tuning());
        let got = e.ocr_page(&page()).expect("blank ok");
        assert!(got.blocks.is_empty());
        assert_eq!(e.report().reoriented_pages, 0);
        assert_eq!(
            take_events(&world),
            vec![
                Event::Build(GPU),
                try_gpu(64),
                Event::Page(GPU, Some(64), PageOrientation::Forced(0)),
            ]
        );
    }

    #[test]
    fn disabled_detection_never_rechecks() {
        // A wrong verdict and real text, but detection is off: one pass.
        let world = World::new();
        world.lock().expect("lock").classifier_turns = 2;
        let mut e = engine(&world, OcrTuning::default());
        e.ocr_page(&page()).expect("page ok");
        assert_eq!(take_events(&world), vec![Event::Build(GPU), try_gpu(64)]);
    }

    #[test]
    fn orientation_recheck_starts_on_the_rung_that_worked() {
        let world = World::new();
        {
            let mut w = world.lock().expect("lock");
            w.classifier_turns = 2;
            w.gpu_max_batch = 32;
        }
        let mut e = engine(&world, ori_tuning());
        let got = e.ocr_page(&page()).expect("page ok");
        assert_eq!(got.page_angle_deg, Some(0.0));
        // Batch 64 has just failed on this page's lines, and the flipped
        // read has the same lines: it starts at 32 instead of failing again.
        assert_eq!(
            take_events(&world),
            vec![
                Event::Build(GPU),
                try_gpu(64),
                Event::Build(GPU),
                try_gpu(32),
                Event::Page(GPU, Some(32), PageOrientation::Forced(0)),
            ]
        );
        // One page, one retry: not yet a pattern.
        assert_eq!(e.report().retried_pages, 1);
        assert_eq!(e.report().sticky_batch, None);
    }

    #[test]
    fn page_whose_recheck_also_degrades_counts_once() {
        let world = World::new();
        {
            let mut w = world.lock().expect("lock");
            w.classifier_turns = 2;
            w.gpu_max_batch = 32;
            w.forced_max_batch = 8; // the flipped read is heavier still
        }
        let mut e = engine(&world, ori_tuning());
        let got = e.ocr_page(&page()).expect("page 1 ok");
        assert_eq!(got.page_angle_deg, Some(0.0));
        let tried: Vec<Event> = take_events(&world)
            .into_iter()
            .filter(|ev| matches!(ev, Event::Page(..)))
            .collect();
        assert_eq!(
            tried,
            vec![
                try_gpu(64),
                try_gpu(32),
                Event::Page(GPU, Some(32), PageOrientation::Forced(0)),
                Event::Page(GPU, Some(8), PageOrientation::Forced(0)),
            ]
        );
        // Both reads stepped down, but it is one degraded page.
        assert_eq!(e.report().retried_pages, 1);
        assert_eq!(e.report().sticky_batch, None);

        e.ocr_page(&page()).expect("page 2 ok");
        assert_eq!(e.report().retried_pages, 2);
        // Sticky on the batch that carried both reads.
        assert_eq!(e.report().sticky_batch, Some(8));
    }

    #[test]
    fn recheck_of_a_cpu_rescued_page_stays_on_cpu() {
        let world = World::new();
        {
            let mut w = world.lock().expect("lock");
            w.classifier_turns = 2;
            w.gpu_max_batch = 0; // GPU hopeless
        }
        let mut e = engine(&world, ori_tuning());
        let got = e.ocr_page(&page()).expect("page saved by CPU");
        assert_eq!(got.page_angle_deg, Some(0.0));
        // The flipped read reuses the CPU session rather than walking the
        // GPU rungs that have just failed.
        assert_eq!(
            take_events(&world),
            vec![
                Event::Build(GPU),
                try_gpu(64),
                Event::Build(GPU),
                try_gpu(32),
                Event::Build(GPU),
                try_gpu(8),
                Event::Build(OcrDevice::Cpu),
                Event::Page(OcrDevice::Cpu, None, PageOrientation::Auto),
                Event::Page(OcrDevice::Cpu, None, PageOrientation::Forced(0)),
            ]
        );
        assert_eq!(e.report().retried_pages, 1);
        assert_eq!(e.report().cpu_pages, 1);
        assert_eq!(e.report().reoriented_pages, 1);
        assert_eq!(e.report().sticky_batch, None);
    }
}
