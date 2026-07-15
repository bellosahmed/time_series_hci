// Pure client-side rendering helpers for the chat UI: HTML escaping, the SVG chart, and the
// adaptive legend. These are side-effect-free (they take data, return strings) so they can be
// unit-tested in Node without a browser/DOM (see chart.test.js). Loaded by index.html as a global
// script, and exported for tests via module.exports at the bottom.

function esc(s){ return (s||'').replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function drawChart(c){
  const W=640,H=170,P=30;
  const hx=c.x.map(Date.parse), hy=c.y;
  let times=[...hx], ys=[...hy];
  const f=c.forecast;
  if(f){ const fx=f.x.map(Date.parse); times=times.concat(fx); ys=ys.concat(f.upper,f.lower,f.mean); }
  const s2=c.series2;  // optional second line (used to overlay two years for a comparison)
  let s2x=null;
  if(s2){ s2x=s2.x.map(Date.parse); times=times.concat(s2x); ys=ys.concat(s2.y); }
  const tMin=Math.min(...times), tMax=Math.max(...times);
  const yMin=Math.min(...ys), yMax=Math.max(...ys);
  const sx=t=> P + (W-2*P)*((t-tMin)/((tMax-tMin)||1));
  const sy=v=> (H-P) - (H-2*P)*((v-yMin)/((yMax-yMin)||1));
  const desc = 'Line chart of ' + (c.label||'sensor') +
    (s2 ? ' compared with ' + (s2.label||'another period') : '') +
    (c.forecast ? ', with 14-day forecast band' : '');
  let svg=`<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="${esc(desc)}"><title>${esc(desc)}</title>`;
  // Faint horizontal gridlines at each y-tick, drawn first so the data line sits on top of them.
  const nYTicks = yMax>yMin ? 4 : 1;
  for(let i=0;i<nYTicks;i++){
    const v = nYTicks>1 ? yMin + (yMax-yMin)*i/(nYTicks-1) : yMin;
    svg+=`<line x1="${P}" y1="${sy(v).toFixed(1)}" x2="${W-P}" y2="${sy(v).toFixed(1)}" class="grid" stroke="#94a3b8" stroke-width="0.5" opacity="0.25"/>`;
  }
  const hpts=hx.map((t,i)=>`${sx(t).toFixed(1)},${sy(hy[i]).toFixed(1)}`).join(' ');
  svg+=`<polyline points="${hpts}" fill="none" stroke="#2563eb" stroke-width="1.6"/>`;
  if(s2){  // amber = the second period, matching the compare legend
    const p2=s2x.map((t,i)=>`${sx(t).toFixed(1)},${sy(s2.y[i]).toFixed(1)}`).join(' ');
    svg+=`<polyline points="${p2}" fill="none" stroke="#e07a34" stroke-width="1.6"/>`;
  }
  if(f){
    const fx=f.x.map(Date.parse);
    const up=fx.map((t,i)=>`${sx(t).toFixed(1)},${sy(f.upper[i]).toFixed(1)}`);
    const lo=fx.map((t,i)=>`${sx(t).toFixed(1)},${sy(f.lower[i]).toFixed(1)}`).reverse();
    svg+=`<polygon points="${up.concat(lo).join(' ')}" fill="#fca5a5" opacity="0.35"/>`;
    svg+=`<polyline points="${fx.map((t,i)=>`${sx(t).toFixed(1)},${sy(f.mean[i]).toFixed(1)}`).join(' ')}" fill="none" stroke="#dc2626" stroke-width="1.6"/>`;
    const dv=sx(Math.max(...hx));
    svg+=`<line x1="${dv.toFixed(1)}" y1="${P-6}" x2="${dv.toFixed(1)}" y2="${H-P}" stroke="#cbd5e1" stroke-dasharray="3 3"/>`;
  }
  (c.marks||[]).forEach(mk=>{ const cx=sx(Date.parse(mk.x)), cy=sy(mk.y);
    if(mk.kind==='borderline'){  // near-threshold: hollow amber ring = "uncertain, not flagged"
      svg+=`<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="3.2" fill="none" stroke="#d97706" stroke-width="1.4" stroke-dasharray="2 1.5"/>`;
      return; }
    const col = mk.kind==='anomaly'?'#dc2626':(mk.kind==='max'?'#ea580c':'#0ea5e9');
    svg+=`<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="3.4" fill="${col}"/>`; });
  // monthaxis: a month-of-year comparison shares a synthetic year, so label months only.
  // Otherwise the label granularity adapts to the plotted span, so a zoomed-in chart (a day or an
  // hour) shows a time-of-day, not just a date that looks identical across every point. `unitMs`
  // is the smallest time difference the chosen format can actually show as a different label --
  // used below to cap tick count so two ticks never render the same-looking date twice.
  const DAY=86400000, span=tMax-tMin;
  let fmt, unitMs;
  if(c.monthaxis){ fmt=d=>new Date(d).toLocaleDateString(undefined,{month:'short'}); unitMs=28*DAY; }
  else if(span<DAY){ fmt=d=>new Date(d).toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'}); unitMs=60000; }
  else if(span<45*DAY){ fmt=d=>new Date(d).toLocaleDateString(undefined,{day:'numeric',month:'short'}); unitMs=DAY; }
  else { fmt=d=>new Date(d).toLocaleDateString(undefined,{month:'short',year:'2-digit'}); unitMs=28*DAY; }
  // y-axis labels at the same ticks the gridlines were drawn at above, not just min/max.
  for(let i=0;i<nYTicks;i++){
    const v = nYTicks>1 ? yMin + (yMax-yMin)*i/(nYTicks-1) : yMin;
    svg+=`<text x="3" y="${(sy(v)+4).toFixed(1)}" class="ax">${v.toFixed(1)}</text>`;
  }
  // Several evenly-spaced ticks along the continuous time axis (not just the two end-points, and
  // not tied to how many raw data points there are) so month/day/hour context is visible all
  // along the axis, not only at its edges. Up to 8 (short labels like "Jun 15" comfortably fit
  // the chart width), but capped by how many *distinct* labels the format can actually produce
  // over this span -- e.g. an 8-tick spread over a 6-day week would otherwise put two ticks
  // under a day apart, rendering the same "Dec 18" twice.
  const maxDistinct = span>0 ? Math.floor(span/unitMs)+1 : 1;
  const nTicks = span>0 ? Math.max(2, Math.min(8, maxDistinct)) : 1;
  for(let i=0;i<nTicks;i++){
    const t = nTicks>1 ? tMin + span*i/(nTicks-1) : tMin;
    const anchor = nTicks===1 || i===0 ? 'start' : i===nTicks-1 ? 'end' : 'middle';
    svg+=`<text x="${sx(t).toFixed(1)}" y="${H-7}" text-anchor="${anchor}" class="ax">${fmt(t)}</text>`;
  }
  return svg+`</svg>`;
}

// A swatch drawn to match each chart marker exactly, so the legend is unambiguous.
function swatch(kind){
  if(kind==='obs')       return '<svg width="20" height="10"><line x1="1" y1="5" x2="19" y2="5" stroke="#2563eb" stroke-width="2"/></svg>';
  if(kind==='anomaly')   return '<svg width="12" height="12"><circle cx="6" cy="6" r="3.4" fill="#dc2626"/></svg>';
  if(kind==='max')       return '<svg width="12" height="12"><circle cx="6" cy="6" r="3.4" fill="#ea580c"/></svg>';
  if(kind==='min')       return '<svg width="12" height="12"><circle cx="6" cy="6" r="3.4" fill="#0ea5e9"/></svg>';
  if(kind==='borderline')return '<svg width="12" height="12"><circle cx="6" cy="6" r="3.2" fill="none" stroke="#d97706" stroke-width="1.4" stroke-dasharray="2 1.5"/></svg>';
  if(kind==='forecast')  return '<svg width="20" height="10"><rect x="1" y="2.5" width="18" height="5" fill="#fca5a5" opacity="0.35"/><line x1="1" y1="5" x2="19" y2="5" stroke="#dc2626" stroke-width="2"/></svg>';
  return '';
}

// A coloured line swatch, for the two-series comparison legend.
function lineSwatch(colour){
  return `<svg width="20" height="10"><line x1="1" y1="5" x2="19" y2="5" stroke="${colour}" stroke-width="2"/></svg>`;
}

function legendHtml(c){
  if(c.series2){        // comparison: name each of the two overlaid lines by colour
    return '<div class="legend" aria-label="Chart legend">' +
      `<span class="item">${lineSwatch('#2563eb')}<span>${esc(c.label||'')}</span></span>` +
      `<span class="item">${lineSwatch('#e07a34')}<span>${esc(c.series2.label||'')}</span></span>` +
      '</div>';
  }
  const kinds = new Set((c.marks||[]).map(m=>m.kind));
  if(kinds.size===0 && !c.forecast) return '';        // plain line only — nothing to explain
  const items = [['obs', 'observed ' + (c.label||'')]];
  if(kinds.has('max'))        items.push(['max','maximum']);
  if(kinds.has('min'))        items.push(['min','minimum']);
  if(kinds.has('anomaly'))    items.push(['anomaly','anomaly']);
  if(kinds.has('borderline')) items.push(['borderline','borderline (near-threshold)']);
  if(c.forecast)              items.push(['forecast','14-day forecast (mean + 80% range)']);
  return '<div class="legend" aria-label="Chart legend">' +
    items.map(([k,lab])=>`<span class="item">${swatch(k)}<span>${esc(lab)}</span></span>`).join('') +
    '</div>';
}

if (typeof module !== 'undefined' && module.exports) {  // Node (tests); no-op in the browser
  module.exports = { esc, drawChart, swatch, lineSwatch, legendHtml };
}
