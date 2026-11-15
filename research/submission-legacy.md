# Archived submission notes - superseded by SUBMISSION.md

> Archive of the five-coin submission draft. See the [current submission](../SUBMISSION.md).

# Submission handoff

## Required links
- Code: https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb
- Write-up: https://github.com/adeviwon/hypecheck/blob/main/WRITEUP.md
- Demo: https://adeviwon.github.io/hypecheck/
- Form: https://airtable.com/appypCyHN77hElIcS/shrXA5VBFdugmoXl6

## One-sentence pitch
Our project helps a beginner crypto investor by taking archived social posts and price history and giving back an honest, baseline-tested comparison of next-day forecasts.

## Before the team submits
- [ ] Confirm registration and eligibility; use each member's registered email.
- [ ] Enter Nathaniel Wong, Aaden Yick and Alastair Chu, each email and school; choose a team name. Do not publish private emails in the public repo.
- [ ] Read the code, methods, limitations and actual results; each member must explain their contribution.
- [ ] Open all submission links logged out; confirm saved notebook outputs are visible.
- [ ] Run the notebook in a fresh Google Colab runtime to verify its Colab-only install step. Local CPU execution is separate evidence.
- [ ] Confirm the rules declaration truthfully. Commit timestamps are evidence, not proof of compliance.
- [ ] Optional: record an unlisted/publicly viewable demo, at most two minutes.
- [ ] One member submits before November 15, 2026, 11:59 PM ET. Verify the confirmation. Latest on-time resubmission is judged.

## Optional two-minute video script
**0:00–0:15 — Problem.** “Memecoin timelines sound confident. We tested whether their sentiment adds information beyond price history—not whether we can make a convincing chart.”

**0:15–1:15 — Demo.** Open the historical replay dashboard. Choose DOGE, 2026-10-10. Compare market-only and market-plus-X probabilities before revealing the outcome. Show the archived post and its availability time. Reveal the actual rise, despite the social model's roughly 33% probability of up. Then show the results table and sparse coverage. Do not imply the dates are live or this is a buy/sell recommendation.

**1:15–1:45 — Method/results.** “We used a pinned pretrained sentiment model, lagged market features, earlier train/validation dates and a purged held-out test. Sentiment scored 53.06% versus 53.68% market-only. Its Brier difference interval crosses zero. We did not demonstrate improvement.”

**1:45–2:00 — Limits/next.** “Only 142 of 965 test coin-days had eligible posts. We'd collect a prospective archive and hand-label post intent. This project is a prototype and is not financial advice.”

Official requirements: https://highschoolhackathon.org/briefs.html and https://highschoolhackathon.org/judges.html. Required code + approximately one-page write-up; video optional, maximum two minutes. No numerical word cap was found in the retrieved brief.
