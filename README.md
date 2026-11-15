# HypeCheck — Signal or Noise?

**AI + Finance · Veritas AI International High School Hackathon 2026**

Nathaniel Wong (model), Aaden Yick (data), Alastair Chu (demo and write-up) · Malvern College Hong Kong

[Try the replay](https://adeviwon.github.io/hypecheck/) · [Run in Colab](https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb) · [Read the method and full results](research/method.md)

## The problem

A beginner crypto investor sees a bullish post and has to decide how much weight to give it. We tested whether adding archived X posts improved next-day forecasts for 13 Binance-listed memecoins.

## What we built

The replay lets you read yesterday’s posts, compare forecasts made with and without the post data, and reveal what happened. Start with **DOGE, April 30, 2026**. The market model estimated a **50.07%** chance of an up day. With two posts added, that fell to **37.77%**. DOGE rose **2.33%** from open to close. The two posts covered different time horizons, one describing a breakout and one warning of a later reversal. The model averaged their tone into one daily input.

We adapted [Finance brief §4.2.4](https://drive.google.com/file/d/15tYD48XRmCgWbk1_lLDtc8lebi6qm-4-/view) from news headlines and the Dow to archived X posts and memecoins. We added the market-history comparison and the replay so a reader can see what post data changes.

## How we tested it

We filtered the public archive to 1,773 original posts with supported ticker tags. A ticker tag can be a passing mention. FinTwitBERT scored the wording, and we trained logistic regression models to predict whether a coin would close above its open on the next UTC day. We fit on earlier dates, chose settings on the following dates, and evaluated on the latest dates.

## Results

**Market history and coin identity got 54.40% right. Adding post counts and sentiment brought that to 53.61%.** The posts-and-coin model got 53.09%, compared with 46.07% for always up, 48.51% for copying yesterday, and 53.93% for always picking the most common training outcome.

We observed posts on only **219 of 2,509 test rows**. The archive switched from mostly Discord-tagged training data to mostly web-tagged test data. **231 of 232 test posts** used publication time because archive timestamps were missing. We had explored these dates before, so we treat the finding as exploratory.

## What we’d do next

Collect a prospective, timestamped sample with more coverage. Compare a post-count model with a count-plus-sentiment model to isolate the value of wording.

*This project is a prototype and is not financial advice.*

## Sources and credits

[Pinned X-post archive](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto/tree/c4b17a94a667af3da1c9ed5144f998e3a4f29966) · [FinTwitBERT](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment) · [Binance daily bars](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data) · [Veritas brief and sample format](https://drive.google.com/file/d/15tYD48XRmCgWbk1_lLDtc8lebi6qm-4-/view). We used NumPy, pandas, scikit-learn, Transformers and PyTorch, with Hermes Agent and OpenAI coding assistance.
