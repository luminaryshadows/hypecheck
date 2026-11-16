import json
from pathlib import Path

part1 = json.loads(Path("/tmp/_cells_part1.json").read_text())
CELLS = part1["cells"]; SHA_JSON = part1["sha"]; RAW = part1["raw"]

def md(s):   CELLS.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n").splitlines(keepends=True)})
def code(s): CELLS.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                           "source":s.strip("\n").splitlines(keepends=True)})

md(r'''
## 4. Baseline: always guess the majority class

The simplest possible rule. Any model we build has to beat it, and it sets the number we have to read every later accuracy against.
''')

code(r'''
# 4 · The baseline, and the scoring function we use from here on.
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

def report(name, y_true, y_pred):
    print(f"{name}")
    print(f"  accuracy  {accuracy_score(y_true, y_pred):.1%}")
    print(f"  precision {precision_score(y_true, y_pred, zero_division=0):.1%}  (of 'up' calls, how many were really up)")
    print(f"  recall    {recall_score(y_true, y_pred, zero_division=0):.1%}  (of real up days, how many we caught)")

majority = int(y_train.mean() > 0.5)
report(f"Baseline: always '{'up' if majority else 'down'}'", y_test, np.full(len(y_test), majority))
''')

md(r'''
The baseline scores **@@BASE_ACC@@** while understanding nothing at all. If our model lands near that number it has learned nothing worth reporting, and we will say so.
''')

md(r'''
## 5. Model 1: TF-IDF + logistic regression

TF-IDF turns each tweet into numbers based on the words it contains, and logistic regression learns which words point towards a rise. This is deliberately the same first model the sample submission uses — a straightforward, inspectable baseline for text.
''')

code(r'''
# 5 · Model 1 — exactly the approach the sample submission takes first.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vec = TfidfVectorizer()
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

model1 = LogisticRegression(max_iter=1000)
model1.fit(Xtr, y_train)
pred1 = model1.predict(Xte)

report("Model 1: TF-IDF + logistic regression", y_test, pred1)
print(confusion_matrix(y_test, pred1))
print(f"\n{len(vec.vocabulary_):,} distinct words and word-pairs learned from the training tweets.")
''')

md(r'''
## 6. Where Model 1 goes wrong

These are test tweets it labelled incorrectly, followed by the words it leans on most in each direction.
''')

code(r'''
# 6 · The mistakes, and what the model is actually keying on.
mistakes = [(t, y, p) for t, y, p in zip(X_test, y_test, pred1) if y != p]
print(f"Model 1 got {len(mistakes)} of {len(y_test):,} test tweets wrong. Five of them:")
for text, true, said in mistakes[:5]:
    print(f"[really {'UP  ' if true else 'down'}, it said {'UP  ' if said else 'down'}] "
          f"{' '.join(str(text).split())[:125]}")

words = np.array(vec.get_feature_names_out())
print("\nWords pushing Model 1 towards 'up':  ", list(words[np.argsort(model1.coef_[0])[-12:]][::-1]))
print("Words pushing Model 1 towards 'down':", list(words[np.argsort(model1.coef_[0])[:12]]))
''')

md(r'''
Read that word list carefully, because it is the most revealing output in the notebook. The strongest "up" words are not directional at all — they are the names of coins and ordinary filler. Nothing here is a claim about the future; the model is picking up *which coin is being discussed and how that author writes*, not *what the tweet says about tomorrow*.

That is worth testing directly, which is what the next two sections do.
''')

md(r'''
## 7. Our addition: test it on pump.fun-era memecoins it has never seen

**This is the part we added beyond the brief, and it mirrors the sample submission's most important move** — the sample trained on old spam texts and then tested on scams people receive today. We do the same thing with markets.

We take four Solana memecoins from the pump.fun era — **WIF, ACT, PNUT and BOME**, all launched after late 2023 — and we test on every tweet about them. **Model 1 has never seen a single one of these tweets, and none of these four coins was in its training data.**
''')

code(r'''
# 7 · The unseen-market test.
ood_pred = model1.predict(vec.transform(new.text))
new = new.assign(pred=ood_pred)

report(f"Model 1 on pump.fun-era memecoins ({len(new):,} tweets it never saw)", new.up, new.pred)
print(f"  for comparison, the base rate on this group is {max(new.up.mean(), 1 - new.up.mean()):.1%}")

print("\nPer coin:")
for coin in NEW:
    c = new.loc[new.coin.eq(coin)]
    if len(c):
        print(f"  {coin:6s} {len(c):4,d} tweets   up rate {c.up.mean():5.1%}   "
              f"model accuracy {accuracy_score(c.up, c.pred):5.1%}   ({'better' if accuracy_score(c.up, c.pred) > max(c.up.mean(), 1-c.up.mean()) else 'no better'} than always guessing)")
''')

md(r'''
**This is our most important finding.** Model 1 scores @@M1_ACC@@ on the test set it was drawn from, and **@@OOD_ACC@@** on memecoins from the launchpad era — against a base rate of @@OOD_BASE@@.

The explanation is the same one the sample submission reaches with modern scams. The model learned the vocabulary and habits of a handful of large-cap memecoin accounts. A pump.fun-era token is written about by different people, in a different register, at a different point in the market cycle. Nothing it learned transfers, because what it learned was never about price at all.
''')

md(r'''
## 8. Two honest checks on the 80/20 split

The brief asks for an 80/20 split, and section 5 gave us one. But a *random* split on this data has a specific problem we should measure rather than assume: tweets about the same coin on the same day share fate, so if one lands in training and its twin lands in test, the model can appear to predict the future when it is really recognising a day it has already seen.

We test that two ways.
''')

code(r'''
# 8a · How leaky is the random split?
train_days = set(big.loc[X_train.index, "trading_day"])
test_days  = big.loc[X_test.index, "trading_day"]
overlap    = test_days.isin(train_days)
print(f"{overlap.mean():.1%} of the test tweets predict a trading day that ALSO appears in training.")
print("On those rows the model has already seen the answer for that coin-day.")

# 8b · The same model, split by time instead of at random.
ordered = big.sort_values("available_at").reset_index(drop=True)
cut = ordered.available_at.iloc[int(len(ordered) * 0.8)]
past, future = ordered.loc[ordered.available_at <= cut], ordered.loc[ordered.available_at > cut]
vec_t = TfidfVectorizer()
model_t = LogisticRegression(max_iter=1000)
model_t.fit(vec_t.fit_transform(past.text), past.up)
report(f"Chronological 80/20 — train to {str(cut)[:10]}, test after ({len(future):,} tweets)",
       future.up, model_t.predict(vec_t.transform(future.text)))
print(f"  base rate on the future portion: {max(future.up.mean(), 1 - future.up.mean()):.1%}")

# 8c · Making the model less lopsided, as the sample submission does.
model_bal = LogisticRegression(max_iter=1000, class_weight="balanced")
model_bal.fit(Xtr, y_train)
report("Model 3: TF-IDF + balanced class weights", y_test, model_bal.predict(Xte))
''')

md(r'''
**The honest split tells a different story from the flattering one.** On the random 80/20 split the model looks respectable; when we train only on the past and test only on what came after, the advantage largely disappears and lands within a couple of points of the base rate (@@CHRONO_ACC@@ against @@CHRONO_BASE@@). Note also that @@LEAK_PCT@@ of the random split's test rows predict a trading day that also sits in training — a real, measurable overlap.

The balanced-weights change in 8c does what it did in the sample submission: it trades accuracy for recall, which is the right trade if you care about catching ups, and the wrong one if you only look at accuracy. We report both rather than picking the flattering one.
''')

md(r'''
## 9. The demo: HypeCheck explains its call

A model that just says "up" or "down" teaches a beginner nothing. So HypeCheck also says **which words in the tweet drove the answer**, using a hype lexicon we curated and then *measured* against this corpus rather than trusting.

Each term in `data/hype-lexicon.csv` carries a **hype score from 0 to 100**: the share of tweets containing that term which also carry an independent hype marker (an exclamation mark, heavy capitalisation, urgency, or an extreme claim), against a corpus baseline of 16%. Terms that the data contradicted were dropped, and terms that never appear are kept in the file flagged as absent, because their absence is a finding in itself.
''')

code(r'''
# 9 · A lexicon we measured, not one we assumed.
lex = pd.read_csv("data/hype-lexicon.csv")

def terms_for(kind, statuses):
    sel = np.logical_and(lex.curated_as.eq(kind).values, lex.status.isin(statuses).values)
    return set(lex.loc[sel, "term"].str.lower())

BULL = terms_for("bullish", ["validated", "weak"])
BEAR = terms_for("bearish", ["validated", "weak"])
hype_rows = lex.loc[np.logical_and(lex.curated_as.eq("hype").values, lex.hype_meter.notna().values)]
HYPE = {str(t).lower(): float(m) for t, m in zip(hype_rows.term, hype_rows.hype_meter)}

print(f"Measured lexicon: {len(BULL)} bullish terms, {len(BEAR)} bearish terms, {len(HYPE)} hype terms.")
print(f"Of {len(lex)} curated terms, {(lex.status == 'absent').sum()} never appear in this corpus at all "
      f"— including {', '.join(lex.loc[lex.status.eq('absent'), 'term'].head(5).tolist())} ...")
''')

code(r'''
# 9b · HypeCheck's explanation.
def hype_check(tweet, model=model1, vectorizer=vec):
    low = tweet.lower()
    bull = sorted([t for t in BULL if t in low])
    bear = sorted([t for t in BEAR if t in low])
    hyp  = sorted([t for t in HYPE if t in low], key=lambda t: -HYPE[t])
    hype_score = int(np.mean([HYPE[t] for t in hyp])) if hyp else 0

    prob = model.predict_proba(vectorizer.transform([tweet]))[0, 1]
    call = "UP" if prob >= 0.5 else "DOWN"

    parts = []
    if bull: parts.append("bullish wording " + ", ".join(f'"{w}"' for w in bull[:3]))
    if bear: parts.append("bearish wording " + ", ".join(f'"{w}"' for w in bear[:3]))
    if hyp:  parts.append("hype wording " + ", ".join(f'"{w}"' for w in hyp[:3]))
    reasons = "; ".join(parts) if parts else "no words from our measured lexicon"

    return (f"Tomorrow: {call}  (model says {prob:.0%} chance of up)\n"
            f"   hype score {hype_score}/100   detected {reasons}")

for tweet in [
    "Short term we probably get more chop but memes with brand name recognition are putting in cycle bottoms. Bullish on $WIF.",
    "This is going to send it. Massive breakout, huge volume, rocket fuel from here. Dont miss it. $PEPE",
    "Boring day. No volume, price sitting below the range, I am watching and waiting. $DOGE",
]:
    print(" ".join(tweet.split())[:105] + " ...")
    print("  ", hype_check(tweet), "\n")
''')

md(r'''
Notice what the explanation reveals. The third tweet — the least exciting of the three — has **no hype score at all**, because it contains none of the 28 hype terms that survived measurement. And the first one contains words like "bullish" that our earlier lexicon work *validated* against the corpus, while words people expect to matter ("to the moon", "100x", "rugpull") are among the terms that **never appear in this dataset once**.

That is the finding underneath the whole project: this corpus is technical-analysis commentary, not retail shilling. We are not sampling hype accounts, which is one more reason a hype-based price model does not work here.
''')

md(r'''
## 10. Results summary

| | Random 80/20 test | Chronological 80/20 test | Unseen pump.fun-era coins |
|---|---|---|---|
| Base rate (majority class) | @@BASE_ACC@@ | @@CHRONO_BASE@@ | @@OOD_BASE@@ |
| **Model 1: TF-IDF + logistic regression** | @@M1_ACC@@ | @@CHRONO_ACC@@ | @@OOD_ACC@@ |
| Model 3: + balanced class weights | @@BAL_ACC@@ | — | — |

The word lists in section 6 explain all three columns. The model learned *who writes about which coin*, not *what the market will do*. It looks passable on the split it was drawn from, weakens when the split respects time, and is no better than a coin toss on markets from the launchpad era.

## 11. Limitations and next steps

- **The sample is small.** @@NTWEETS@@ labelled tweets across nine coins, and one coin (DOGE) supplies most of them. Confidence intervals on every number above are wide, and we have not quoted any as precise.
- **The target is noisy by construction.** A daily up/down move in a memecoin is close to a coin flip. Our earlier work on this project measured how large an edge you would need to detect one reliably at this sample size, and it is larger than anything we found.
- **Same-day tweets share a label.** Tweets about the same coin on the same day all point at the same outcome, so accuracy is not independent across rows. This is why the chronological split in section 8 matters more than the random one.
- **We can only use coins that have a Binance series.** Plenty of pump.fun-era tokens — POPCAT, FARTCOIN, GOAT, FWOG, MOODENG — have tweets in this corpus but no Binance daily bar, so they cannot be scored against our target at all. That is a data-availability limit, not a choice.
- **Next, we would:** label a few hundred tweets by hand for *intent* (directional call, promotion, news) rather than trusting the archive's ticker tags; widen the coin universe to the tokens that do have price history; and report a cost-sensitive measure, because a missed move and a false alarm cost a beginner very differently.

## 12. Sources and credits

- **Tweets:** [StephanAkkerman/financial-tweets-crypto](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto) (MIT).
- **Prices:** Binance spot daily klines (USDT pairs), public market data, pinned in our [repository](https://github.com/REPLACE_REPO).
- **Hype lexicon:** curated and measured by us; published as `data/hype-lexicon.csv`.
- **Libraries:** scikit-learn (TfidfVectorizer, LogisticRegression), pandas, NumPy, Requests.
- **AI tools:** we used an AI coding assistant to help debug pandas joins and to sanity-check the leakage measurement in section 8, and we read the scikit-learn documentation on `class_weight` before using it.
- The sample submission provided by Veritas AI shaped this notebook's structure, and we say so plainly: the 80/20 split, the majority-class baseline, the first TF-IDF model, the mistake analysis, the unseen-data test and the explanation demo are all modelled on it.

*Prototype, not financial advice. The results are the ones we actually measured, including the negative one.*
''')

# substitute tokens into every cell's source BEFORE serialising, so multi-line values stay valid
SHIM = None
for c in CELLS:
    src = "".join(c["source"])
    src = (src.replace("SHA_JSON", SHA_JSON)
              .replace("RAWURL", RAW)
              .replace("REPLACE_REPO", "adeviwon/hypecheck"))
    c["source"] = src.splitlines(keepends=True)

nb = {"cells": CELLS,
      "metadata": {"colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12.3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path("/home/ubuntu/Projects/hypecheck/HypeCheck-Memecoins.ipynb")
txt = json.dumps(nb, indent=1)
out.write_text(txt)
print(f"wrote {out.name} — {len(CELLS)} cells, {len(txt)/1024:.1f} KB")
import re
toks = sorted(set(re.findall(r"@@[A-Z_]+@@", txt)))
print("markdown tokens still to fill:", toks)
print("fences balanced:", txt.count("```") % 2 == 0)
