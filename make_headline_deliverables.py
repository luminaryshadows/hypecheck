"""Generate the write-up, technical results and replay forecasts."""
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
report = json.loads((ROOT / 'docs/headline-report.json').read_text())
context = json.loads((ROOT / 'docs/headline-context.json').read_text())
predictions = pd.read_csv(ROOT / 'artifacts/headline-predictions.csv')
metrics = report['metrics']
period = report['protocol']['splits']['test']
coverage = report['coverage']
post_rows = coverage['test_coin_days_with_posts']
support = context['periods']['test']
training = context['periods']['train']
example = predictions[predictions.coin.eq('DOGE') & predictions.date.str.startswith('2026-04-30')].iloc[0]
browser = 'https://adeviwon.github.io/hypecheck/'
colab = 'https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb'
brief = 'https://drive.google.com/file/d/15tYD48XRmCgWbk1_lLDtc8lebi6qm-4-/view'
dataset = 'https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto'
archive = f"{dataset}/tree/{report['model_source']['dataset_revision']}"
encoder = 'https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment'
binance = 'https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data'

writeup = f"""# HypeCheck — Signal or Noise?

**AI + Finance · Veritas AI International High School Hackathon 2026**

Nathaniel Wong (model), Aaden Yick (data), Alastair Chu (demo and write-up) · Malvern College Hong Kong

[Try the replay]({browser}) · [Run in Colab]({colab}) · [Read the method and full results](research/method.md)

## The problem

A beginner crypto investor sees a bullish post and has to decide how much weight to give it. We tested whether adding archived X posts improved next-day forecasts for {len(report['protocol']['coins'])} Binance-listed memecoins.

## What we built

The replay lets you read yesterday’s posts, compare forecasts made with and without the post data, and reveal what happened. Start with **DOGE, April 30, 2026**. The market model estimated a **{example['Market only']:.2%}** chance of an up day. With two posts added, that fell to **{example['Market + sentiment']:.2%}**. DOGE rose **{example.actual_return:.2%}** from open to close. The two posts covered different time horizons, one describing a breakout and one warning of a later reversal. The model averaged their tone into one daily input.

We adapted [Finance brief §4.2.4]({brief}) from news headlines and the Dow to archived X posts and memecoins. We added the market-history comparison and the replay so a reader can see what post data changes.

## How we tested it

We filtered the public archive to {coverage['eligible_original_posts']:,} original posts with supported ticker tags. A ticker tag can be a passing mention. FinTwitBERT scored the wording, and we trained logistic regression models to predict whether a coin would close above its open on the next UTC day. We fit on earlier dates, chose settings on the following dates, and evaluated on the latest dates.

## Results

**Market history and coin identity got {metrics['Market only']['accuracy']:.2%} right. Adding post counts and sentiment brought that to {metrics['Market + sentiment']['accuracy']:.2%}.** The posts-and-coin model got {metrics['Sentiment + coin']['accuracy']:.2%}, compared with {metrics['Always up']['accuracy']:.2%} for always up, {metrics['Previous-day direction']['accuracy']:.2%} for copying yesterday, and {metrics['Training majority']['accuracy']:.2%} for always picking the most common training outcome.

We observed posts on only **{post_rows} of {period['rows']:,} test rows**. The archive switched from mostly Discord-tagged training data to mostly web-tagged test data. **{support['missing_archive_time']} of {support['posts']} test posts** used publication time because archive timestamps were missing. We had explored these dates before, so we treat the finding as exploratory.

## What we’d do next

Collect a prospective, timestamped sample with more coverage. Compare a post-count model with a count-plus-sentiment model to isolate the value of wording.

*This project is a prototype and is not financial advice.*

## Sources and credits

[Pinned X-post archive]({archive}) · [FinTwitBERT]({encoder}) · [Binance daily bars]({binance}) · [Veritas brief and sample format]({brief}). We used NumPy, pandas, scikit-learn, Transformers and PyTorch, with Hermes Agent and OpenAI coding assistance.
"""
(ROOT / 'README.md').write_text(writeup, encoding='utf-8')

lines = [
    '# HypeCheck method and full results', '',
    '[Project write-up](../README.md) · [Replay](' + browser + ') · [Notebook](' + colab + ')', '',
    '## Evaluation', '',
    f"We tested one forecast per coin and UTC day from {period['start']} to {period['end']}. "
    f"The test has {period['rows']:,} forecasts across {period['dates']} shared dates.", '',
    '| Partition | First date | Last date | Forecast rows | Shared dates |',
    '|---|---|---|---:|---:|',
]
for name, values in report['protocol']['splits'].items():
    lines.append(f"| {name} | {values['start']} | {values['end']} | {values['rows']:,} | {values['dates']} |")
lines += [
    '',
    'Posts feed the day after their assigned UTC date. We use the later of publication and first archive time when both are present. We use publication time alone where the archive time is missing. An up day closes above its own open. Flat days count as not up. The previous-day rule uses the same open-to-close definition.',
    '',
    'The one-day gaps separate the target windows at the split boundaries. Scaling and model fitting use training rows. Validation Brier error selects the setting from three regularization strengths (C = 0.01, 0.1, 1.0). We call up at 50% or higher. The fixed model uses no final-period labels for fitting. The team had inspected those dates in earlier experiments, so results are exploratory.',
    '',
    'The study uses a fixed panel of 13 Binance USDT spot pairs with available daily bars. The exchange and coin selection limit generalization. FinTwitBERT scores each post as bullish, bearish or neutral. The social inputs are count, presence, mean bullish-minus-bearish score, disagreement and average bullish and bearish scores. The market inputs use lagged prices and volume. Every learned model includes coin identity. The comparison measures the effect of adding that full bundle of post features.',
    '',
    '## Metrics', '',
    'Accuracy is the fraction of correct calls. Balanced accuracy weights up and not-up days equally. Brier error measures squared probability error and is lower when estimates are closer to the outcomes. The constant training up rate is the probability baseline. The hard decision rules use 0% and 100% probabilities for Brier scoring.',
    '',
    '| Method | Accuracy | Balanced accuracy | Brier error |',
    '|---|---:|---:|---:|',
]
for name, values in metrics.items():
    lines.append(f"| {name} | {values['accuracy']:.2%} | {values['balanced_accuracy']:.2%} | {values['brier']:.4f} |")

sentiment_up = predictions['Sentiment + coin'] >= .5
actual_up = predictions.actual_up.eq(1)
with_posts = predictions[predictions.post_count > 0]
only_accuracy = ((with_posts['Market only'] >= .5) == with_posts.actual_up).mean()
post_accuracy = ((with_posts['Market + sentiment'] >= .5) == with_posts.actual_up).mean()
lines += [
    '',
    f"The posts-and-coin model called not up on {(~sentiment_up).sum():,} rows and caught {(sentiment_up & actual_up).sum():,} of {actual_up.sum():,} actual up days.",
    '',
    f"On the {len(with_posts)} rows with observed posts, market history scored {only_accuracy:.2%} and adding the posts scored {post_accuracy:.2%}. This subset comparison is descriptive. Both models were fit on the original full training set.",
    '',
    '### Paired accuracy differences', '',
    '| Posts + coin minus baseline | Difference | Exploratory 95% interval |',
    '|---|---:|---:|',
]
paired = report['paired_accuracy_differences']
for name in ('Always up', 'Previous-day direction', 'Training majority', 'Coin only', 'Market only'):
    result = paired[name]
    low, high = result['interval95']
    lines.append(f"| {name} | {100 * result['mean']:+.2f} pp | [{100 * low:+.2f}, {100 * high:+.2f}] pp |")
delta = paired['sentiment_increment_brier']
low, high = delta['interval95']
controls = report['controls']
lines += [
    '',
    'Each accuracy interval includes zero. Intervals resample seven-date blocks and keep coins on the same date together.',
    '',
    f"Adding the post features increased Brier error by **{delta['mean']:+.6f}**, with an exploratory 95% interval of **[{low:+.6f}, {high:+.6f}]**. This interval is entirely above zero.",
    '',
    f"The posts-and-coin model scored {controls['real_accuracy']:.2%}. Twenty label-shuffled and twenty post-feature-shuffled refits reached maxima of {controls['label_permuted_max']:.2%} and {controls['feature_placebo_max']:.2%}. These are checks against spurious accuracy. Four expanding windows reuse the final dates while updating the model as time advances.",
    '',
    '### Post types', '',
    'The first matching keyword group wins in this order: events, promotion, technical, other. A launch post with a rocket emoji goes into events. “Bullish” can trigger the technical group. These labels are approximate. A coin-day can include posts from several groups. Each comparison adds one group’s count and average sentiment to market history and is scored over the full test panel.',
    '',
    '| Type | Test coin-days with type | Post-coin matches | Brier change | 95% interval | Adjusted p |',
    '|---|---:|---:|---:|---|---:|',
]
for kind in report['kinds']:
    result = kind['paired_brier_vs_market']
    low, high = result['interval95']
    lines.append(f"| {kind['kind']} | {kind['test_coin_days_with_kind']} | {kind['test_posts']} | "
                 f"{result['mean']:+.6f} | [{low:+.6f}, {high:+.6f}] | {kind['holm_adjusted_p']:.3f} |")
lines += [
    '',
    'A negative change favors the added features. All four type intervals include zero.',
    '',
    '## Archive coverage', '',
    f"Training included {training['posts']:,} eligible posts assigned to training dates, all tagged Discord by the archive. "
    f"Of {support['posts']} eligible posts assigned to test dates, {support['archive_sources']['web']} were tagged web and "
    f"{support['archive_sources']['both']} were tagged both. {support['missing_archive_time']} lacked first archive time. "
    f"The {post_rows} test rows with posts account for {post_rows / period['rows']:.1%} of forecasts. "
    'A missing observation means the archive had no matched post. The archive is selected, and ticker tags, English sentiment and truncated or edited text all limit interpretation.',
    '',
    '## Reproduce', '',
    f"Open [HypeCheck.ipynb]({colab}) and choose **Runtime → Run all**. The default notebook hash-checks cached sentiment and retrains the direction classifiers. The optional recompute switch runs FinTwitBERT again with frozen weights. The site reads recorded forecasts.",
    '',
    'From a Python environment with [requirements.txt](../requirements.txt) and IPython for the notebook checks. On macOS use the matching PyPI PyTorch build, without the Linux +cpu suffix:', '',
    '```bash', 'python headline_study.py --universe expanded', 'python build_headline_context.py',
    'python make_deliverables.py', 'python verify_submission.py', '```', '',
    'The verification command covers the current study and still checks the archived files that ship with the repo. Two legacy checks also require the original local five-coin sentiment cache and report a skip when it is absent.',
    '',
    '[Measured report](../docs/headline-report.json) · [Every forecast](../artifacts/headline-predictions.csv) · [Replay excerpts and coverage](../docs/headline-context.json) · [Archived experiments](README.md)',
    '',
    '## Sources and credits', '',
    f"- [Pinned X-post archive]({archive}) by Stephan Akkerman.",
    f"- [Pinned FinTwitBERT checkpoint]({encoder}/tree/{report['model_source']['revision']}) by Stephan Akkerman and Tim Koornstra.",
    f"- [Binance daily USDT spot bars]({binance}), shipped in the [verified 13-coin panel](../data/expanded-memecoin-daily.csv).",
    f"- [Veritas Finance brief §4.2.4 and submission format]({brief}).",
    '- NumPy, pandas, scikit-learn, PyTorch, Transformers, Hugging Face Hub, PyArrow, Requests and IPython. Hermes Agent and OpenAI assisted with implementation and review.',
    '',
    '*This project is a prototype and is not financial advice. The study measured directional and probability error. Trading performance was outside its scope.*',
]
(ROOT / 'research/method.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
(ROOT / 'docs/headline-predictions.json').write_text(predictions.to_json(orient='records'), encoding='utf-8')
subprocess.run([sys.executable, str(ROOT / 'build_headline_colab.py')], check=True)
print('Generated the write-up, technical results, replay forecasts and notebook.')
