import { readFileSync } from 'node:fs';
const html = readFileSync(new URL('./docs/archive.html', import.meta.url), 'utf8');
const report = JSON.parse(readFileSync(new URL('./docs/report.json', import.meta.url), 'utf8'));

function makeEl(tag) {
  const el = {
    tagName: tag, children: [], attributes: {}, textContent: '', className: '', hidden: false,
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = nodes; },
    setAttribute(key, value) { this.attributes[key] = value; },
    getAttribute(key) { return this.attributes[key]; },
    addEventListener(type, handler) { (this.listeners ??= {})[type] = handler; },
    classList: { add() {}, remove() {} },
  };
  return el;
}
const ids = [...html.matchAll(/id="([^"]+)"/g)].map(m => m[1]);
const store = new Map(ids.map(id => [id, makeEl('div')]));
globalThis.document = {
  getElementById: id => store.get(id) || null,
  createElement: makeEl,
  createElementNS: (namespace, tag) => makeEl(tag),
};
globalThis.fetch = async () => ({ ok: true, json: async () => report });

await import('./docs/app.js');
await new Promise(resolve => setTimeout(resolve, 250));

function walk(el, out = []) {
  if (!el || typeof el !== 'object') return out;
  if (el.tagName === 'tr') out.push(el);
  (el.children || []).forEach(child => walk(child, out));
  return out;
}
const assert = (await import('node:assert/strict')).default;
const extendedRows = walk(store.get('extended'));
const metricRows = walk(store.get('metrics'));
assert.equal(metricRows.length, report.metrics.length, `primary table rows: ${metricRows.length}`);
assert.equal(extendedRows.length, 5, `extended table rows incl. header: ${extendedRows.length}`);
const names = extendedRows.slice(1).map(row => row.children[0].textContent);
assert.deepEqual(names, ['Market only (primary)', 'Market + X (primary)', 'Market + BTC + intraday', 'Market + BTC + intraday + X']);
assert.ok(store.get('extended').children.some(c => c.tagName === 'h3' && c.textContent === 'Declared secondary comparison'));
const replayDate = store.get('date').children.length;
const replayCoin = store.get('coin').children.length;
assert.equal(replayCoin, 5, `coin options: ${replayCoin}`);
assert.ok(replayDate > 100, `date options: ${replayDate}`);
// The reveal control must reveal on demand and hide again, without disturbing the model numbers.
const reveal = store.get('reveal');
const outcome = store.get('outcome');
const collect = (el, out = []) => {
  if (!el || typeof el !== 'object') return out;
  if (el.textContent) out.push(el.textContent);
  (el.children || []).forEach(child => collect(child, out));
  return out;
};
const marketBefore = store.get('market-prob').textContent;
reveal.checked = false;
reveal.listeners.change();
assert.equal(outcome.textContent, 'Outcome hidden. Make your own assessment first.');
reveal.checked = true;
reveal.listeners.change();
const revealed = collect(outcome).join(' ');
assert.match(revealed, /Realized next-24h return/, `revealed outcome text: ${revealed}`);
assert.equal(store.get('market-prob').textContent, marketBefore, 'revealing must not change the reported probabilities');
assert.equal(store.get('actual-key').hidden, false, 'the actual-direction legend must appear only when revealed');
console.log(`DOM smoke passed: ${metricRows.length} primary rows, ${extendedRows.length - 1} extended rows, ${replayCoin} coins, ${replayDate} dates.`);
