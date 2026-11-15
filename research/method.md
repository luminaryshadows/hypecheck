# HypeCheck method and full results

[Project write-up](../README.md) · [Replay](https://adeviwon.github.io/hypecheck/) · [Notebook](https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb)

## Evaluation

We tested one forecast per coin and UTC day from 2026-04-12 to 2026-10-21. The test has 2,509 forecasts across 193 shared dates.

| Partition | First date | Last date | Forecast rows | Shared dates |
|---|---|---|---:|---:|
| train | 2024-03-01 | 2025-09-29 | 6,171 | 578 |
| validation | 2025-10-01 | 2026-04-10 | 2,496 | 192 |
| test | 2026-04-12 | 2026-10-21 | 2,509 | 193 |

Posts feed the day after their assigned UTC date. We use the later of publication and first archive time when both are present. We use publication time alone where the archive time is missing. An up day closes above its own open. Flat days count as not up. The previous-day rule uses the same open-to-close definition.

The one-day gaps separate the target windows at the split boundaries. Scaling and model fitting use training rows. Validation Brier error selects the setting from three regularization strengths (C = 0.01, 0.1, 1.0). We call up at 50% or higher. The fixed model uses no final-period labels for fitting. The team had inspected those dates in earlier experiments, so results are exploratory.

The study uses a fixed panel of 13 Binance USDT spot pairs with available daily bars. The exchange and coin selection limit generalization. FinTwitBERT scores each post as bullish, bearish or neutral. The social inputs are count, presence, mean bullish-minus-bearish score, disagreement and average bullish and bearish scores. The market inputs use lagged prices and volume. Every learned model includes coin identity. The comparison measures the effect of adding that full bundle of post features.

## Metrics

Accuracy is the fraction of correct calls. Balanced accuracy weights up and not-up days equally. Brier error measures squared probability error and is lower when estimates are closer to the outcomes. The constant training up rate is the probability baseline. The hard decision rules use 0% and 100% probabilities for Brier scoring.

| Method | Accuracy | Balanced accuracy | Brier error |
|---|---:|---:|---:|
| Always up | 46.07% | 50.00% | 0.5393 |
| Previous-day direction | 48.51% | 48.16% | 0.5149 |
| Training majority | 53.93% | 50.00% | 0.4607 |
| Training prevalence | 53.93% | 50.00% | 0.2486 |
| Sentiment + coin | 53.09% | 49.60% | 0.2494 |
| Coin only | 53.93% | 50.00% | 0.2485 |
| Market only | 54.40% | 51.38% | 0.2477 |
| Market + sentiment | 53.61% | 50.59% | 0.2486 |

The posts-and-coin model called not up on 2,370 rows and caught 59 of 1,156 actual up days.

On the 219 rows with observed posts, market history scored 50.68% and adding the posts scored 46.58%. This subset comparison is descriptive. Both models were fit on the original full training set.

### Paired accuracy differences

| Posts + coin minus baseline | Difference | Exploratory 95% interval |
|---|---:|---:|
| Always up | +7.01 pp | [-1.67, +16.02] pp |
| Previous-day direction | +4.58 pp | [-0.04, +8.69] pp |
| Training majority | -0.84 pp | [-1.83, +0.16] pp |
| Coin only | -0.84 pp | [-1.83, +0.16] pp |
| Market only | -1.32 pp | [-3.35, +0.80] pp |

Each accuracy interval includes zero. Intervals resample seven-date blocks and keep coins on the same date together.

Adding the post features increased Brier error by **+0.000908**, with an exploratory 95% interval of **[+0.000238, +0.001589]**. This interval is entirely above zero.

The posts-and-coin model scored 53.09%. Twenty label-shuffled and twenty post-feature-shuffled refits reached maxima of 54.48% and 54.44%. These are checks against spurious accuracy. Four expanding windows reuse the final dates while updating the model as time advances.

### Post types

The first matching keyword group wins in this order: events, promotion, technical, other. A launch post with a rocket emoji goes into events. “Bullish” can trigger the technical group. These labels are approximate. A coin-day can include posts from several groups. Each comparison adds one group’s count and average sentiment to market history and is scored over the full test panel.

| Type | Test coin-days with type | Post-coin matches | Brier change | 95% interval | Adjusted p |
|---|---:|---:|---:|---|---:|
| events | 16 | 16 | +0.000131 | [-0.000179, +0.000582] | 1.000 |
| promotion | 12 | 13 | -0.000029 | [-0.000326, +0.000246] | 1.000 |
| technical | 53 | 57 | +0.000124 | [-0.000052, +0.000309] | 0.586 |
| other | 161 | 185 | +0.000587 | [-0.000003, +0.001205] | 0.396 |

A negative change favors the added features. All four type intervals include zero.

## Archive coverage

Training included 1,464 eligible posts assigned to training dates, all tagged Discord by the archive. Of 232 eligible posts assigned to test dates, 231 were tagged web and 1 were tagged both. 231 lacked first archive time. The 219 test rows with posts account for 8.7% of forecasts. A missing observation means the archive had no matched post. The archive is selected, and ticker tags, English sentiment and truncated or edited text all limit interpretation.

## Reproduce

Open [HypeCheck.ipynb](https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb) and choose **Runtime → Run all**. The default notebook hash-checks cached sentiment and retrains the direction classifiers. The optional recompute switch runs FinTwitBERT again with frozen weights. The site reads recorded forecasts.

From a Python environment with [requirements.txt](../requirements.txt) and IPython for the notebook checks. On macOS use the matching PyPI PyTorch build, without the Linux +cpu suffix:

```bash
python headline_study.py --universe expanded
python build_headline_context.py
python make_deliverables.py
python verify_submission.py
```

The verification command covers the current study and still checks the archived files that ship with the repo. Two legacy checks also require the original local five-coin sentiment cache and report a skip when it is absent.

[Measured report](../docs/headline-report.json) · [Every forecast](../artifacts/headline-predictions.csv) · [Replay excerpts and coverage](../docs/headline-context.json) · [Archived experiments](README.md)

## Sources and credits

- [Pinned X-post archive](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto/tree/c4b17a94a667af3da1c9ed5144f998e3a4f29966) by Stephan Akkerman.
- [Pinned FinTwitBERT checkpoint](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment/tree/da059da3b3bbcb43f9ed1aeb5ae61644010c7e1e) by Stephan Akkerman and Tim Koornstra.
- [Binance daily USDT spot bars](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data), shipped in the [verified 13-coin panel](../data/expanded-memecoin-daily.csv).
- [Veritas Finance brief §4.2.4 and submission format](https://drive.google.com/file/d/15tYD48XRmCgWbk1_lLDtc8lebi6qm-4-/view).
- NumPy, pandas, scikit-learn, PyTorch, Transformers, Hugging Face Hub, PyArrow, Requests and IPython. Hermes Agent and OpenAI assisted with implementation and review.

*This project is a prototype and is not financial advice. The study measured directional and probability error. Trading performance was outside its scope.*
