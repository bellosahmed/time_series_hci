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

// --- drawChart: x-axis has several intermediate ticks, not just the two end-points ---
const wideChart = {
  label: 'T (degC)',
  x: Array.from({ length: 12 }, (_, i) => new Date(2013, i, 1).toISOString()),
  y: Array.from({ length: 12 }, (_, i) => i),
};
const wideSvg = drawChart(wideChart);
const axLabels = (wideSvg.match(/class="ax"/g) || []).length;
assert.ok(axLabels >= 6, `expected >=6 "ax" labels (2 y-axis + several x-axis ticks), got ${axLabels}`);
// intermediate ticks must actually be different months, not the same label repeated
const monthLabels = [...wideSvg.matchAll(/class="ax">([^<]+)</g)].map(m => m[1]);
assert.ok(new Set(monthLabels).size >= 5, `expected several distinct labels, got ${monthLabels}`);

// --- drawChart: a sub-day span shows hour:minute on the x-axis, not just day/month/year ---
const hourly = {
  label: 'T (degC)',
  x: ['2016-12-31T21:00:00.000Z', '2016-12-31T22:00:00.000Z',
      '2016-12-31T23:00:00.000Z', '2017-01-01T00:00:00.000Z'],
  y: [1, 2, 3, 4],
};
const hourlySvg = drawChart(hourly);
assert.ok(/\d{1,2}:\d{2}/.test(hourlySvg), 'expected an HH:MM-style time label on a sub-day chart');

// --- drawChart: monthaxis (year-on-year comparison) still gets multiple month ticks ---
const cmpTicks = drawChart(cmp);
const cmpAxLabels = (cmpTicks.match(/class="ax"/g) || []).length;
assert.ok(cmpAxLabels >= 5, `expected >=5 "ax" labels on the comparison chart, got ${cmpAxLabels}`);

// --- drawChart: a single-month span shows the day-of-month progression (1..30), not just a
// handful of sparse points spanning a whole year's worth of months.
const month = {
  label: 'T (degC)',
  x: Array.from({ length: 30 }, (_, i) => new Date(2016, 5, i + 1).toISOString()),
  y: Array.from({ length: 30 }, (_, i) => i),
};
const monthSvg = drawChart(month);
const monthDayLabels = [...monthSvg.matchAll(/class="ax">Jun (\d{1,2})</g)].map(m => Number(m[1]));
assert.ok(monthDayLabels.length >= 6, `expected >=6 day-of-month labels, got ${monthDayLabels}`);
assert.ok(new Set(monthDayLabels).size === monthDayLabels.length,
  `expected distinct day numbers, got duplicates: ${monthDayLabels}`);
assert.ok(monthDayLabels[0] <= 2 && monthDayLabels[monthDayLabels.length - 1] >= 29,
  `expected labels spanning close to day 1 through 30, got ${monthDayLabels}`);

// --- drawChart: a short span (e.g. a 6-day "last week" window) must not repeat the same
// day-of-month label on two different ticks -- 8 flat ticks over 6 days can land two ticks less
// than a day apart, which look identical under day-granularity formatting.
const week = {
  label: 'wv (m/s)',
  x: Array.from({ length: 7 }, (_, i) => new Date(2016, 11, 18 + i).toISOString()), // Dec 18-24
  y: [1.23, 1.19, 1.43, 1.44, 2.06, 1.57, 3.03],
};
const weekSvg = drawChart(week);
const weekLabels = [...weekSvg.matchAll(/class="ax">(Dec \d{1,2})</g)].map(m => m[1]);
assert.ok(weekLabels.length >= 5, `expected several distinct day labels, got ${weekLabels}`);
assert.strictEqual(new Set(weekLabels).size, weekLabels.length,
  `expected no duplicate day labels, got ${weekLabels}`);

// --- drawChart: y-axis gets intermediate ticks too, not just the min/max end-points ---
const yAxisChart = {
  label: 'T (degC)',
  x: Array.from({ length: 20 }, (_, i) => new Date(2016, 5, i + 1).toISOString()),
  y: Array.from({ length: 20 }, (_, i) => i), // 0..19, so min=0, max=19
};
const yAxisSvg = drawChart(yAxisChart);
// y-axis labels sit at a fixed x="3"; count how many distinct numeric labels appear there
const yLabels = [...yAxisSvg.matchAll(/x="3" y="[^"]+" class="ax">([^<]+)</g)].map(m => Number(m[1]));
assert.ok(yLabels.length >= 4, `expected >=4 y-axis labels, got ${yLabels}`);
assert.ok(yLabels.includes(0) && yLabels.includes(19), `expected min (0) and max (19) among labels, got ${yLabels}`);
assert.strictEqual(new Set(yLabels).size, yLabels.length, `expected distinct y-axis values, got ${yLabels}`);
// gridlines: one faint horizontal line per y-tick, spanning the plot width
const gridlineCount = (yAxisSvg.match(/<line[^>]*class="grid"/g) || []).length;
assert.strictEqual(gridlineCount, yLabels.length,
  `expected one gridline per y-tick (${yLabels.length}), got ${gridlineCount}`);

// --- drawChart: a marker ABOVE the line's own range must not be clipped off the top. The line is
// often a daily/monthly MEAN (e.g. tops out ~20) while a max marker sits at the true peak (~37);
// the y-axis must expand to include marker values so the dot stays inside the plot area.
const H = 170, P = 30;   // must match drawChart's constants
const clipped = {
  label: 'T (degC)',
  x: Array.from({ length: 5 }, (_, i) => new Date(2015, 6, i + 1).toISOString()),
  y: [18, 19, 20, 19, 18],                                   // mean line, peaks at 20
  marks: [{ x: new Date(2015, 6, 3).toISOString(), y: 37, kind: 'max' }],  // true max well above
};
const clippedSvg = drawChart(clipped);
// the orange max marker's cy must land within the plotting band [P, H-P]
const maxMark = [...clippedSvg.matchAll(/<circle cx="[^"]+" cy="([^"]+)" r="[^"]+" fill="#ea580c"/g)]
  .map(m => Number(m[1]));
assert.strictEqual(maxMark.length, 1, 'expected one max marker drawn');
assert.ok(maxMark[0] >= P && maxMark[0] <= H - P,
  `max marker cy=${maxMark[0]} must be inside the plot band [${P}, ${H - P}] (not clipped)`);
// and a y-axis label must now reach the marker value
const clipYLabels = [...clippedSvg.matchAll(/x="3" y="[^"]+" class="ax">([^<]+)</g)].map(m => Number(m[1]));
assert.ok(Math.max(...clipYLabels) >= 37,
  `expected the y-axis to reach the max marker (37), top label was ${Math.max(...clipYLabels)}`);

console.log('chart.test.js: all assertions passed');
