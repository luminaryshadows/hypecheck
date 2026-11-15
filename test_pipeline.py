"""Small runnable gates: .venv/bin/python -m unittest -v test_pipeline.py."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import pipeline as p


class PipelineChecks(unittest.TestCase):
    def test_sentiment_cache_rejects_invalid_probabilities(self):
        p.validate_sentiment_scores({"bullish": 0.2, "bearish": 0.3, "neutral": 0.5})
        bad = [None, {}, {"bullish": True, "bearish": 0, "neutral": 0},
               {"bullish": float("nan"), "bearish": 0, "neutral": 1},
               {"bullish": -0.1, "bearish": 0.1, "neutral": 1},
               {"bullish": 0.8, "bearish": 0.8, "neutral": 0.1}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                p.validate_sentiment_scores(value)

    def test_training_prevalence_baseline_is_train_only(self):
        frame = pd.read_csv(p.ROOT / "data/features.csv", parse_dates=["date"])
        train, _, test = p.split_dates(frame)
        expected = p.metric("Training prevalence", test, np.full(len(test), train.actual_up.mean()))
        report = json.loads((p.ROOT / "docs/report.json").read_text())
        saved = next(row for row in report["metrics"] if row["name"] == "Training prevalence")
        self.assertAlmostEqual(saved["brier"], expected["brier"], places=12)
        self.assertAlmostEqual(saved["accuracy"], expected["accuracy"], places=12)


    def test_market_features_cannot_see_today_or_future(self):
        prices = p.load_prices("DOGE")
        original = p.market_features(prices)
        changed = prices.copy()
        index = 80
        changed.loc[index:, ["open", "high", "low", "close", "volume"]] *= 3
        altered = p.market_features(changed)
        np.testing.assert_allclose(original.loc[index, p.MARKET].to_numpy(float), altered.loc[index, p.MARKET].to_numpy(float), rtol=0, atol=0)
        self.assertFalse(np.allclose(original.loc[index + 1, p.MARKET].to_numpy(float), altered.loc[index + 1, p.MARKET].to_numpy(float)))
        np.testing.assert_allclose(original.actual_return, prices.close / prices.open - 1)

    def test_source_availability_is_conservative(self):
        raw, selected = p.load_posts()
        self.assertFalse(raw.tweet_id.duplicated().any())
        self.assertTrue(selected.tweet_type.eq("tweet").all())
        self.assertTrue((selected.available_at >= selected.created_at).all())
        observed = selected.first_posted_at.notna()
        self.assertTrue((selected.loc[observed, "available_at"] >= selected.loc[observed, "first_posted_at"]).all())
        self.assertTrue((selected.available_at < selected.date).all())
        self.assertTrue((selected.available_at >= selected.date - pd.Timedelta(days=1)).all())
        self.assertFalse(selected.text.duplicated().any())
        self.assertTrue((selected.available_at >= p.START - pd.Timedelta(days=1)).all())

    def test_pinned_dataset_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            (root / "data/posts.parquet").write_bytes(b"PAR1not the pinned dataset")
            with patch.object(p, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    p.load_posts()

    def test_missing_price_day_is_not_forward_filled(self):
        prices = p.load_prices("DOGE").drop(index=5)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            prices.to_csv(root / "data/DOGE-prices.csv", index=False)
            with patch.object(p, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "Missing daily prices"):
                    p.load_prices("DOGE")
        with self.assertRaisesRegex(ValueError, "Unknown coin"):
            p.load_prices("../../secret")

    def test_split_boundaries_and_complete_coin_support(self):
        frame = pd.read_csv(p.ROOT / "data/features.csv")
        frame["date"] = pd.to_datetime(frame.date, utc=True)
        train, val, test = p.split_dates(frame)
        self.assertLess(train.date.max() + pd.Timedelta(days=1), val.date.min())
        self.assertLess(val.date.max() + pd.Timedelta(days=1), test.date.min())
        for a, b in [(train, val), (val, test), (train, test)]:
            self.assertFalse(set(a.date) & set(b.date))
        for part in (train, val, test):
            self.assertEqual(set(part.coin), set(p.COINS))
            self.assertTrue((part.groupby("date").coin.nunique() == 5).all())
            self.assertEqual(part.actual_up.nunique(), 2)
        self.assertTrue((frame.loc[frame.post_count.eq(0), p.SOCIAL] == 0).all().all())

    def test_report_metrics_match_saved_predictions(self):
        report = json.loads((p.ROOT / "docs/report.json").read_text())
        rows = pd.DataFrame(report["predictions"])
        self.assertEqual(len(rows), len(rows.drop_duplicates(["coin", "date"])))
        self.assertEqual(set(rows.coin), set(p.COINS))
        for metric in report["metrics"]:
            self.assertEqual(metric["n"], len(rows))
            self.assertEqual(metric["n_dates"], rows.date.nunique())
        for name, column in [("Market only", "market_prob"), ("Market + attention", "attention_prob"), ("Market + X", "social_prob")]:
            measured = p.scores(rows.actual_up, rows[column])
            saved = next(x for x in report["metrics"] if x["name"] == name)
            for key in measured:
                self.assertAlmostEqual(measured[key], saved[key], places=12)
        for row in report["predictions"]:
            cutoff = pd.Timestamp(row["cutoff"])
            self.assertEqual(pd.Timestamp(row["target_end"]) - cutoff, pd.Timedelta(days=1))
            self.assertEqual(row["actual_up"], int(row["actual_return"] > 0))
            self.assertEqual(row["sentiment"] is None, row["post_count"] == 0)
            for post in row["examples"]:
                available = pd.Timestamp(post["available_at"])
                self.assertLess(available, cutoff)
                self.assertGreaterEqual(available, cutoff - pd.Timedelta(days=1))

    def test_exported_json_coefficients_reproduce_predictions(self):
        exports = json.loads((p.ROOT / "artifacts/coefficients.json").read_text())
        report = json.loads((p.ROOT / "docs/report.json").read_text())
        frame = pd.read_csv(p.ROOT / "data/features.csv")
        frame["date"] = pd.to_datetime(frame.date, utc=True)
        _, _, test = p.split_dates(frame)
        for name, column in [("Market only", "market_prob"), ("Market + attention", "attention_prob"), ("Market + X", "social_prob")]:
            artifact = exports[name]
            names = artifact["feature_names"]
            numeric = [key.removeprefix("numeric__") for key in names if key.startswith("numeric__")]
            matrix = (test[numeric].to_numpy() - artifact["numeric_mean"]) / artifact["numeric_scale"]
            categories = [key.removeprefix("coin__coin_") for key in names if key.startswith("coin__")]
            onehot = np.column_stack([test.coin.eq(coin).to_numpy(float) for coin in categories])
            logits = np.column_stack([matrix, onehot]) @ artifact["coefficients"] + artifact["intercept"][0]
            prob = 1 / (1 + np.exp(-logits))
            np.testing.assert_allclose(prob, [r[column] for r in report["predictions"]], atol=1e-12, rtol=1e-12)

    def test_manifest_binds_code_inputs_and_published_outputs(self):
        manifest = json.loads((p.ROOT / "artifacts/manifest.json").read_text())
        self.assertEqual(manifest["code_sha256"], p.digest(p.ROOT / "pipeline.py"))
        for key in ["docs/report.json", "artifacts/coefficients.json"]:
            self.assertIn(key, manifest["files"])
        archived_cache = "data/sentiment.json"
        for filename, checksum in manifest["files"].items():
            if filename != archived_cache:
                self.assertEqual(checksum, p.digest(p.ROOT / filename))
        if not (p.ROOT / archived_cache).exists():
            self.skipTest("Archived five-coin sentiment cache is local; all other manifest entries checked")
        self.assertEqual(manifest["files"][archived_cache], p.digest(p.ROOT / archived_cache))

    def test_hourly_features_cannot_see_the_cutoff_day(self):
        hourly = p.load_hourly("DOGE")
        bars = pd.read_csv(p.ROOT / "data/market-hourly.csv")
        bars = bars[bars.coin.eq("DOGE")].copy()
        bars["open_time"] = pd.to_datetime(bars.open_time, utc=True)
        prior = bars[bars.open_time.dt.date == pd.Timestamp("2024-04-10").date()].sort_values("open_time")
        cutoff_day = hourly[hourly.date.eq(pd.Timestamp("2024-04-11", tz="UTC"))].iloc[0]
        self.assertAlmostEqual(cutoff_day.hourly_vol1, np.log(prior.close_usdt / prior.open_usdt).std(ddof=1), places=12)
        self.assertAlmostEqual(cutoff_day.last_hour_ret, np.log(prior.close_usdt.iloc[-1] / prior.open_usdt.iloc[-1]), places=12)
        self.assertAlmostEqual(cutoff_day.taker_ratio1, prior.taker_buy_volume_usdt.sum() / prior.volume_usdt.sum(), places=12)
        self.assertEqual(int(prior.open_time.dt.hour.max()), 23)
        # The cutoff day's own hours must not appear anywhere in its features.
        following = bars[bars.open_time.dt.date == pd.Timestamp("2024-04-11").date()]
        self.assertNotAlmostEqual(cutoff_day.hourly_range1, (following.high_usdt.max() - following.low_usdt.min()) / following.close_usdt.iloc[-1], places=6)

    def test_hourly_gap_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            bars = pd.read_csv(p.ROOT / "data/market-hourly.csv")
            bars = bars[bars.coin.eq("DOGE")].reset_index(drop=True).drop(index=100)
            bars.to_csv(root / "data/market-hourly.csv", index=False)
            with patch.object(p, "ROOT", root), patch.object(p, "HOURLY_SHA256", p.digest(root / "data/market-hourly.csv")):
                with self.assertRaisesRegex(ValueError, "contiguous"):
                    p.load_hourly("DOGE")

    def test_pinned_secondary_sources_fail_closed(self):
        for name, attribute, loader in [("market-hourly.csv", "HOURLY_SHA256", "load_hourly"),
                                        ("expanded-daily.csv", "EXPANDED_SHA256", "load_btc")]:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "data").mkdir()
                (root / "data" / name).write_text("coin,tampered\n")
                with patch.object(p, "ROOT", root), patch.object(p, attribute, "0" * 64):
                    with self.assertRaisesRegex(ValueError, "SHA-256"):
                        p.load_hourly("DOGE") if loader == "load_hourly" else p.load_btc()

    def test_btc_context_is_lagged_and_matches_our_own_bars(self):
        btc = p.load_btc()
        self.assertGreater(len(btc), 1000)
        self.assertTrue((btc.date.diff().dropna() == pd.Timedelta(days=1)).all())
        tail = btc.dropna(subset=["btc_ret1", "btc_ret7"])
        self.assertTrue(np.isfinite(tail[["btc_ret1", "btc_ret7"]].to_numpy(float)).all())
        # The independent daily file must reproduce our own Binance bars exactly (checked inside load_btc).
        ours = p.load_prices("DOGE")
        merged = btc.merge(ours[["date", "close"]], on="date", how="inner")
        self.assertGreater(len(merged), 900)

    def test_extended_market_features_are_lagged_too(self):
        prices = p.load_prices("DOGE")
        frame = pd.read_csv(p.ROOT / "data/features.csv")
        frame["date"] = pd.to_datetime(frame.date, utc=True)
        self.assertTrue(np.isfinite(frame[p.MARKET_EXT].to_numpy(float)).all())
        del prices
        for coin in p.COINS:
            hourly = p.load_hourly(coin)
            joined = frame[frame.coin.eq(coin)].merge(hourly, on="date", how="inner", suffixes=("", "_source"))
            self.assertEqual(len(joined), 965)
            for column in p.INTRADAY:
                np.testing.assert_allclose(joined[column].to_numpy(float), joined[column + "_source"].to_numpy(float), rtol=1e-12, atol=1e-15)
            del joined, hourly

    def test_primary_metrics_are_unchanged_by_the_extension(self):
        report = json.loads((p.ROOT / "docs/report.json").read_text())
        published = {"Always up": 0.465284974093264, "Training majority": 0.534715025906736, "Training prevalence": 0.534715025906736,
                     "Previous direction": 0.492227979274611, "Market only": 0.536787564766839,
                     "Market + attention": 0.544041450777202, "Market + X": 0.530569948186528}
        measured = {row["name"]: row["accuracy"] for row in report["metrics"]}
        self.assertEqual(set(measured), set(published))
        for name, value in published.items():
            self.assertAlmostEqual(measured[name], value, places=12)

    def test_extended_metrics_match_saved_predictions(self):
        report = json.loads((p.ROOT / "docs/report.json").read_text())
        rows = pd.DataFrame(report["predictions"])
        pairs = [("Market + BTC + intraday", "extended_market_prob"),
                 ("Market + BTC + intraday + X", "extended_social_prob")]
        for name, column in pairs:
            measured = p.scores(rows.actual_up, rows[column])
            saved = next(x for x in report["extended"]["metrics"] if x["name"] == name)
            for key in measured:
                self.assertAlmostEqual(measured[key], saved[key], places=12)

    def test_independent_scoring_check_is_reproducible(self):
        report = json.loads((p.ROOT / "docs/report.json").read_text())
        checked = report["independent_check"]
        self.assertTrue(checked["available"])
        self.assertNotEqual(checked["our_model_revision"], checked["independent_model_revision"])
        self.assertEqual(checked["shared_posts"], 1122)
        cache_path = p.ROOT / "data/sentiment.json"
        if not cache_path.exists():
            self.skipTest("Archived five-coin sentiment cache is local")
        cache = json.loads(cache_path.read_text())["scores"]
        other = pd.read_csv(p.ROOT / "data/independent-post-scores.csv", usecols=["post_id", "p_bullish"])
        other["post_id"] = other.post_id.astype(str)
        merged = other.drop_duplicates("post_id")
        merged = merged[merged.post_id.isin([str(k) for k in cache])]
        delta = np.abs([cache[k]["bullish"] - merged.loc[merged.post_id.eq(str(k)), "p_bullish"].iloc[0] for k in merged.post_id])
        self.assertAlmostEqual(float(delta.mean()), checked["mean_abs_p_bullish_difference"], places=12)

    def test_empty_subgroup_and_invalid_probabilities(self):
        self.assertEqual(p.scores([], [])["n"], 0)
        self.assertIsNone(p.scores([], [])["accuracy"])
        for values in ([np.nan], [-0.1], [1.1]):
            with self.assertRaisesRegex(ValueError, "Invalid forecast"):
                p.scores([1], values)


if __name__ == "__main__":
    unittest.main()
