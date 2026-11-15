'use strict';

const element = id => document.getElementById(id);
const percent = value => (100 * value).toFixed(2) + '%';
const signed = (value, digits = 2) => (value >= 0 ? '+' : '') + value.toFixed(digits);
const direction = up => up ? 'Up' : 'Not up';

function text(tag, value) {
    const node = document.createElement(tag);
    node.textContent = value;
    return node;
}

function term(target, label, value) {
    const group = document.createElement('div');
    group.append(text('dt', label), text('dd', value));
    target.append(group);
}

function tableRow(target, label, values) {
    const row = document.createElement('tr');
    const header = text('th', label);
    header.scope = 'row';
    row.append(header, ...values.map(value => text('td', value)));
    element(target).append(row);
}

async function readJSON(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error('Could not load the study. Open the notebook for saved results.');
    return response.json();
}

function displayResults(report, context) {
    const test = report.protocol.splits.test;
    const count = report.coverage.test_coin_days_with_posts;
    const metrics = report.metrics;
    term(element('summary'), 'Market history + coin', percent(metrics['Market only'].accuracy));
    term(element('summary'), 'Market history + coin + posts', percent(metrics['Market + sentiment'].accuracy));
    term(element('summary'), 'Test rows with posts', count + ' / ' + test.rows.toLocaleString('en-US'));

    element('period').textContent = test.start + ' to ' + test.end + ' · ' +
        test.rows.toLocaleString('en-US') + ' forecasts across ' + test.dates + ' shared UTC dates.';
    const names = [
        ['Market only', 'Market history + coin'],
        ['Market + sentiment', 'Market history + coin + posts'],
        ['Sentiment + coin', 'Posts + coin'],
        ['Coin only', 'Coin only'],
        ['Training majority', 'Most common training direction'],
        ['Training prevalence', 'Training up rate (' + percent(context.training_up_rate) + ')'],
        ['Always up', 'Always up'],
        ['Previous-day direction', 'Copy yesterday’s direction'],
    ];
    for (const [key, label] of names) {
        const m = metrics[key];
        tableRow('metrics', label, [percent(m.accuracy), percent(m.balanced_accuracy), m.brier.toFixed(4)]);
    }
    const brier = report.paired_accuracy_differences.sentiment_increment_brier;
    const [low, high] = brier.interval95;
    element('comparison').textContent = 'Adding posts increased probability error by ' +
        brier.mean.toFixed(6) + ' (exploratory 95% interval ' + signed(low, 6) + ' to ' + signed(high, 6) + ').';
    for (const kind of report.kinds) {
        const difference = kind.paired_brier_vs_market;
        tableRow('kinds', kind.kind, [kind.test_coin_days_with_kind, signed(difference.mean, 6),
            difference.interval95.map(value => signed(value, 6)).join(' to ')]);
    }
    element('coverage').textContent = 'We found eligible posts for ' + percent(count / test.rows) +
        ' of test forecasts. The other rows had no matching posts in this archive.';
    const train = context.periods.train;
    const source = context.periods.test;
    element('archive-note').textContent = 'The archive tagged ' + (train.archive_sources.discord ?? 0).toLocaleString('en-US') +
        ' of ' + train.posts.toLocaleString('en-US') + ' training posts as Discord. It tagged ' +
        (source.archive_sources.web ?? 0) + ' of ' + source.posts + ' test posts as web. ' +
        source.missing_archive_time + ' test posts lacked archive timestamps and used publication time.';
}

function showPosts(row, posts, ids) {
    const target = element('posts');
    target.replaceChildren();
    if (!row.post_count) {
        target.append(text('p', 'The archive contained no eligible posts for this coin on the previous UTC day.'));
        return;
    }
    if (row.post_count > ids.length) {
        target.append(text('p', 'Showing ' + ids.length + ' of ' + row.post_count + ' posts.'));
    }
    for (const id of ids) {
        const post = posts[id];
        const link = text('a', 'Open post on X');
        link.href = 'https://x.com/i/web/status/' + encodeURIComponent(id);
        link.rel = 'noreferrer';
        const metadata = text('p', post.time_basis + ' · ' + post.time + ' · ' +
            post.tone + ' wording (' + percent(post.score) + ') · ');
        metadata.className = 'post-meta';
        metadata.append(link);
        target.append(text('blockquote', post.text), metadata);
    }
}

function bindReplay(records, context, coins) {
    const coin = element('coin');
    const day = element('day');
    const button = element('reveal');
    const outcome = element('outcome');
    const dates = new Map(coins.map(symbol => [symbol, new Map()]));
    for (const row of records) dates.get(row.coin).set(row.date.slice(0, 10), row);
    for (const symbol of [...coins].sort()) coin.append(new Option(symbol, symbol));

    function showForecast() {
        const row = dates.get(coin.value).get(day.value);
        const market = row['Market only'];
        const social = row['Market + sentiment'];
        const forecast = element('forecast');
        forecast.replaceChildren();
        term(forecast, 'Market history + coin', percent(market) + ' · ' + direction(market >= 0.5));
        term(forecast, 'Market history + coin + posts', percent(social) + ' · ' + direction(social >= 0.5));
        element('delta').textContent = 'Change after adding posts: ' + signed(100 * (social - market)) + ' percentage points.';
        showPosts(row, context.posts, context.examples[coin.value + '/' + day.value] ?? []);
        outcome.textContent = 'Outcome hidden.';
        button.onclick = () => {
            outcome.textContent = 'Actual: ' + direction(row.actual_up) + ' · ' +
                signed(100 * row.actual_return) + '% open to close.';
        };
    }

    function setDays(preferred) {
        day.replaceChildren();
        const available = dates.get(coin.value);
        for (const date of available.keys()) day.append(new Option(date, date));
        if (available.has(preferred)) day.value = preferred;
        showForecast();
    }

    coin.value = 'DOGE';
    setDays('2026-04-30');
    coin.onchange = () => setDays(day.value);
    day.onchange = showForecast;
    coin.disabled = day.disabled = button.disabled = false;
}

async function load() {
    const [report, records, context] = await Promise.all([
        readJSON('headline-report.json'), readJSON('headline-predictions.json'), readJSON('headline-context.json')
    ]);
    if (records.length !== report.protocol.splits.test.rows) throw new Error('The replay files disagree on the test row count.');
    displayResults(report, context);
    bindReplay(records, context, report.protocol.coins);
}

load().catch(error => { element('error').textContent = error.message; });
