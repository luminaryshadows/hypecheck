"""Publish short archived post excerpts and coverage for the current replay."""
from collections import Counter
import json
from pathlib import Path

import pandas as pd

import colab_setup
from expanded_inputs import COINS, select_posts
import pipeline as core

ROOT = Path(__file__).resolve().parent


def build_context(posts, predictions, report, scores):
    periods = {}
    for name, bounds in report['protocol']['splits'].items():
        subset = posts[posts.date.between(bounds['start'], bounds['end'])]
        periods[name] = {
            'posts': len(subset),
            'archive_sources': dict(Counter(subset.source.fillna('unknown'))),
            'missing_archive_time': int(subset.first_posted_at.isna().sum()),
        }

    eligible = {}
    excerpts = {}
    bounds = report['protocol']['splits']['test']
    test = posts[posts.date.between(bounds['start'], bounds['end'])]
    for post in test.itertuples():
        post_id = str(post.tweet_id)
        score = scores[post_id]
        text = post.text.strip()
        excerpts[post_id] = {
            'text': text[:240].rstrip() + '…' if len(text) > 240 else text,
            'time': post.available_at.strftime('%Y-%m-%d %H:%M UTC'),
            'time_basis': 'Publication time' if pd.isna(post.first_posted_at) else 'Later of publication and archive time',
            'tone': max(score, key=score.get),
            'score': round(max(score.values()), 4),
        }
        for coin in sorted(set(post.tickers) & set(COINS)):
            key = f'{coin}/{post.date.date()}'
            eligible.setdefault(key, []).append(post_id)

    examples = {}
    for row in predictions.itertuples():
        key = f'{row.coin}/{row.date[:10]}'
        ids = eligible.get(key, [])
        if len(ids) != row.post_count:
            raise ValueError(f'Post count does not match the published forecast for {key}')
        if ids:
            examples[key] = ids[-2:]
    used = {post_id for ids in examples.values() for post_id in ids}
    return {'periods': periods, 'training_up_rate': float(predictions['Training prevalence'].iloc[0]),
            'examples': examples,
            'posts': {key: excerpts[key] for key in sorted(used)}}


if __name__ == '__main__':
    cache_path = 'data/expanded-sentiment.json'
    if core.digest(ROOT / cache_path) != colab_setup.FILE_DIGESTS[cache_path]:
        raise ValueError('Replay needs the pinned sentiment cache')
    raw, _ = core.load_posts()
    scores = json.loads((ROOT / cache_path).read_text())['scores']
    output = build_context(
        select_posts(raw),
        pd.read_csv(ROOT / 'artifacts/headline-predictions.csv'),
        json.loads((ROOT / 'docs/headline-report.json').read_text()),
        scores,
    )
    core.atomic_json(ROOT / 'docs/headline-context.json', output)
