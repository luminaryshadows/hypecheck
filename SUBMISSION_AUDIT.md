> Archive of the pre-expansion audit. See the [current submission](SUBMISSION.md) and its [method](research/method.md).

# Submission audit and experiment choice

## Decision
Submit **HypeCheck.ipynb — Signal or Noise?** as the code link, and **WRITEUP.md** as the required approximately one-page write-up. The primary question is whether archived X sentiment adds next-day forecasting value beyond market history. We did not demonstrate that improvement. Do not submit the later volatility/causality framing or headline the random-split accuracy.

## What changed
- Restored the defensible pre-reframing main notebook structure and replaced overlong/contradictory write-up and exploratory claims.
- Added a training-prevalence probability baseline without changing learned-model features, hyperparameter selection or predictions.
- Made slow post-model exploration opt-in and clearly separate from the primary study.
- Saved actual end-to-end notebook outputs. Generator retains outputs only for an identical ordered code-cell sequence; subprocess failures stop execution.
- Removed absolute paths from the DOM test, which previously tested a different working checkout.
- Validated cached sentiment labels, types, finiteness, bounds and probability sum. Blocked allowlisted symlink escapes in the localhost server, with regression tests.
- Replaced the stale README generator; it no longer overwrites the dependency lockfile or restores unsupported claims.

## Verification
An empty working-directory run downloaded and SHA-verified all 11 inputs, recomputed all 1,124 sentiment scores and completed all 12 code cells in **190.53 seconds**. The model-weight download cache could be shared; this is not a fresh machine/environment install. The generated report equals the audit report except for `meta.generated_at`; metrics, predictions, uncertainty and secondary-arm results match exactly. The notebook ran its **19 pipeline tests**. The complete repository suite subsequently passed **23 Python tests**, the frontend self-test and the checkout-relative DOM test (7 primary rows, 4 secondary rows, 5 coins, 193 dates). Evidence: `artifacts/submission-verification.json`.

**Not verified:** a live Google Colab session and its Colab-only pip installation; current dependency-vulnerability status; optional exploratory model search in this final run; a recorded video; registration or external submission. `artifacts/dependency-audit.json` is historical, not a newly run scan.

## Notebook/history comparison
All **12 main-notebook revisions** and the one Memecoins revision were inventoried programmatically in `artifacts/notebook-history.json`. Before this repair every main revision had no saved outputs.

| Revision(s) | What it represents | Submission assessment |
|---|---|---|
| a92c4d7 | Original executable primary study | Sound foundation, sparse presentation |
| 69c7608, ed7d7c2 | Secondary arms, fuller submission structure | Better evidence and presentation, incomplete standalone acquisition |
| 7a85c8b, 0507f74, 741d780 | Team, guide alignment, reveal verification | Presentation/verification refinements |
| d774e4b | Sparse-data diagnostics | Useful, but overstates what collecting more data could do |
| c96fea4, aac629b, 2e861e0 | Richer post features and lexicon; split comparisons | Exploratory; do not select a favourable block as confirmation |
| 4d469c8 | Standalone-fetch improvements, pre-reframing notebook | Best starting point; repaired into the current submission |
| b0d05ad | Hype/volatility opening | Not defensible as written; not reproduced by its own experiment cells |
| 60f8bd7: HypeCheck-Memecoins.ipynb | Simple text model, saved outputs, unseen-market illustration | Secondary demonstration only: random overlap, shared target dates, no date-purged unseen-coin guarantee |

## Other experiments: preserve, do not pool
- `postmodel.py`: useful random/chronological/walk-forward sensitivity; feature/model search is exploratory.
- `pf_model.py`: different trade target; whole-panel median imputation leaks test-distribution information. Not a primary result.
- `meme_model.py`: broader coin universe, repeated outcomes and row-based cuts. Not comparable to the primary test.
- Event/decay scripts: same-day associations, overlapping events and temporary-file dependencies. They cannot establish causal price impact; one decay placebo mismatches coin/BTC random minutes.
- Learned meter: Ridge learns FinTwitBERT teacher scores, so held-out teacher agreement is not independent validation. Full-fit downstream scores can be future-informed. Do not headline them as validated hype.
- Volatility experiment (`ef53ef4`): market-context-only beat hype + context and the hype-shuffle placebo retained performance. The later claim that hype is the source of predictive volatility is unsupported.
- Intensity check (`4b002b5`): failed engagement validation; call the measured quantity sentiment, not validated attention.
- Impact/residual/classification attempts: overlapping exploratory studies, not five independent causal falsifications.

## Official requirement map
Sources: https://highschoolhackathon.org/briefs.html ; linked official Project Briefs PDF; https://highschoolhackathon.org/judges.html ; official submission form linked in SUBMISSION.md.

| Requirement | Evidence/status |
|---|---|
| AI/ML performs the main task | Frozen neural sentiment + trained logistic regression |
| Finance chronological test and simple baselines | Purged dates, always-up, majority, persistence and probability baseline |
| Show actual input/output and at least one failure | Notebook failure cell and replay reveal |
| Notebook outputs visible | All 12 code cells have saved outputs |
| Approximately one-page template write-up | WRITEUP.md; same text embedded in notebook |
| Credits and finance disclaimer | Notebook, README and write-up |
| Accessible code and write-up | Verify anonymous GitHub/raw/Colab URLs at handoff |
| Full team details and registration emails | Team must enter in form; do not publish emails |
| Honest rules confirmation | Team must confirm; commit dates alone cannot prove compliance |
| Video | Optional, maximum 2 minutes; script in SUBMISSION.md, no video recorded |
| Submission | Team must submit form and check confirmation |

Build window: November 14, 2026, 12:00 AM ET through November 15, 11:59 PM ET. Existing commits start November 14 at 14:50 HKT, within that window; this does not independently prove all work began then. Round 1 rubric: technical execution 30, originality 20, impact/track fit 20, testing/honesty 15, communication 15. The strongest axis here is reproducible evaluation; the weaker axes are originality and demonstrated user impact. Do not invent user interviews or prospective success.
