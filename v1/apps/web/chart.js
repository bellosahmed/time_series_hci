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
  const tMin=Math.min(...times), tMax=Math.max(...times);
  const yMin=Math.min(...ys), yMax=Math.max(...ys);
  const sx=t=> P + (W-2*P)*((t-tMin)/((tMax-tMin)||1));
  const sy=v=> (H-P) - (H-2*P)*((v-yMin)/((yMax-yMin)||1));
  const desc = 'Line chart of ' + (c.label||'sensor') + (c.forecast ? ', with 14-day forecast band' : '');
  let svg=`<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="${esc(desc)}"><title>${esc(desc)}</title>`;
  const hpts=hx.map((t,i)=>`${sx(t).toFixed(1)},${sy(hy[i]).toFixed(1)}`).join(' ');
  svg+=`<polyline points="${hpts}" fill="none" stroke="#2563eb" stroke-width="1.6"/>`;
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
  const fmt=d=>new Date(d).toLocaleDateString(undefined,{day:'numeric',month:'short',year:'2-digit'});
  svg+=`<text x="3" y="${(sy(yMax)+4).toFixed(1)}" class="ax">${yMax.toFixed(1)}</text>`;
  svg+=`<text x="3" y="${(sy(yMin)+4).toFixed(1)}" class="ax">${yMin.toFixed(1)}</text>`;
  svg+=`<text x="${P}" y="${H-7}" class="ax">${fmt(tMin)}</text>`;
  svg+=`<text x="${W-P-52}" y="${H-7}" class="ax">${fmt(tMax)}</text>`;
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

function legendHtml(c){
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
  module.exports = { esc, drawChart, swatch, legendHtml };
}
