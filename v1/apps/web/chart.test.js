// Unit tests for the pure client-side render helpers (chart.js). Run: `node chart.test.js`.
// Also driven from the Python suite via tests/test_web_js.py (skipped if node is absent).
const assert = require('assert');
const { esc, drawChart, swatch, lineSwatch, legendHtml } = require('./chart.js');

// --- esc ---
assert.strictEqual(esc('<a>&'), '&lt;a&gt;&amp;');
assert.strictEqual(esc(null), '');

// --- legendHtml: adapts to the markers present ---
let h = legendHtml({ label: 'T (degC)', marks: [{ kind: 'anomaly' }, { kind: 'borderline' }] });
assert.ok(h.includes('observed T (degC)'));
assert.ok(h.includes('anomaly'));
assert.ok(h.includes('borderline (near-threshold)'));
assert.ok(legendHtml({ label: 'T (degC)', marks: [{ kind: 'max' }] }).includes('maximum'));
assert.ok(legendHtml({ label: 'T (degC)', marks: [], forecast: { x: [1] } }).includes('forecast'));
assert.strictEqual(legendHtml({ label: 'x', marks: [] }), '');   // plain line => no legend

// --- swatch matches the marker styles ---
assert.ok(swatch('borderline').includes('stroke-dasharray'));    // dashed ring
assert.ok(swatch('anomaly').includes('#dc2626'));                // filled red
assert.strictEqual(swatch('unknown'), '');

// --- drawChart: accessible SVG with the right marks ---
const chart = {
  label: 'T (degC)',
  x: [new Date(2014, 0, 1).toISOString(), new Date(2014, 0, 2).toISOString(), new Date(2014, 0, 3).toISOString()],
  y: [1, 5, 2],
  marks: [{ x: new Date(2014, 0, 2).toISOString(), y: 5, kind: 'anomaly' },
          { x: new Date(2014, 0, 3).toISOString(), y: 2, kind: 'borderline' }],
};
const svg = drawChart(chart);
assert.ok(svg.startsWith('<svg'));
assert.ok(svg.includes('role="img"') && svg.includes('<title>'));   // accessible
assert.ok(svg.includes('#dc2626'));                                 // anomaly mark
assert.ok(svg.includes('stroke-dasharray="2 1.5"'));                // borderline ring
assert.ok(svg.trim().endsWith('</svg>'));

// --- drawChart + legendHtml: two-series comparison overlay ---
const cmp = {
  label: 'T (degC) — 2013', monthaxis: true,
  x: [new Date(2000, 0, 1).toISOString(), new Date(2000, 6, 1).toISOString()],
  y: [2, 20],
  series2: { label: 'T (degC) — 2014',
             x: [new Date(2000, 0, 1).toISOString(), new Date(2000, 6, 1).toISOString()],
             y: [4, 24] },
};
const csvg = drawChart(cmp);
assert.ok(csvg.includes('#2563eb') && csvg.includes('#e07a34'));   // both series drawn
assert.ok(csvg.includes('compared with T (degC) — 2014'));         // accessible description
const clg = legendHtml(cmp);
assert.ok(clg.includes('T (degC) — 2013') && clg.includes('T (degC) — 2014'));  // both named
assert.ok(lineSwatch('#e07a34').includes('#e07a34'));

console.log('chart.test.js: all assertions passed');
