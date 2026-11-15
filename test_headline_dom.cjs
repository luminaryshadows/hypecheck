/* Exercise the published replay using the study's real data. */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const paths = ['headline-report.json', 'headline-predictions.json', 'headline-context.json'];
const fixtures = Object.fromEntries(paths.map(path => [path, JSON.parse(fs.readFileSync('docs/' + path))]));
const report = fixtures['headline-report.json'];
const records = fixtures['headline-predictions.json'];

class Element {
    constructor() {
        this.children = [];
        this.textContent = '';
        this.value = '';
        this.disabled = true;
    }
    append(...items) {
        for (const item of items) {
            this.children.push(item);
            if (item.option && !this.value) this.value = item.value;
        }
    }
    replaceChildren() {
        this.children = [];
        this.value = '';
    }
}

async function harness(fail = false) {
    const ids = ['summary', 'metrics', 'period', 'coin', 'day', 'forecast', 'reveal',
        'outcome', 'kinds', 'error', 'delta', 'posts', 'coverage', 'archive-note', 'comparison'];
    const elements = Object.fromEntries(ids.map(id => [id, new Element()]));
    vm.runInNewContext(fs.readFileSync('docs/headlines.js', 'utf8'), {
        document: {
            createElement: () => new Element(),
            getElementById: id => { assert(elements[id], id); return elements[id]; },
        },
        Option: function (label, value) { this.option = true; this.textContent = label; this.value = value; },
        fetch: async path => ({ok: !fail, json: async () => { assert(fixtures[path]); return fixtures[path]; }}),
    });
    await new Promise(resolve => setImmediate(resolve));
    return elements;
}

const collect = node => [node.textContent, ...node.children.map(collect)].join(' ');

(async () => {
    const ui = await harness();
    assert.equal(ui.error.textContent, '');
    assert.equal(ui.metrics.children.length, 8);
    assert.equal(ui.summary.children.length, 3);
    assert.equal(ui.kinds.children.length, 4);
    assert.equal(ui.coin.children.length, report.protocol.coins.length);
    assert.equal(ui.day.children.length, report.protocol.splits.test.dates);
    assert.deepEqual([ui.coin.value, ui.day.value], ['DOGE', '2026-04-30']);
    assert.match(collect(ui.forecast), /50\.07% · Up.*37\.77% · Not up/);
    assert.match(collect(ui.posts), /descending triangle/);
    assert.match(collect(ui.posts), /Publication time/);
    assert.equal(ui.outcome.textContent, 'Outcome hidden.');
    ui.reveal.onclick();
    assert.equal(ui.outcome.textContent, 'Actual: Up · +2.33% open to close.');

    ui.coin.value = 'BOME';
    ui.coin.onchange();
    assert.equal(ui.day.value, '2026-04-30');
    assert.equal(ui.outcome.textContent, 'Outcome hidden.');
    ui.day.value = '2026-04-12';
    ui.day.onchange();
    assert.match(collect(ui.posts), /no eligible posts/);
    ui.reveal.onclick();
    assert.equal(ui.outcome.textContent, 'Actual: Not up · -6.02% open to close.');

    const contexts = fixtures['headline-context.json'];
    for (const row of records) {
        const ids = contexts.examples[row.coin + '/' + row.date.slice(0, 10)] ?? [];
        assert.equal(ids.length, Math.min(row.post_count, 2));
        for (const id of ids) assert(contexts.posts[id]);
    }

    const bad = await harness(true);
    assert.match(bad.error.textContent, /Could not load/);
    assert.equal(bad.reveal.disabled, true);
    console.log('Replay passed: DOGE example, results, date changes, no-post and error states.');
})().catch(error => { console.error(error); process.exit(1); });
