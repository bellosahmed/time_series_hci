"""Generate a short supervisor-update PowerPoint for v1.

Run:  python3.12 report/make_slides.py
Out:  report/Supervisor_Update_v1.pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ---- palette -------------------------------------------------------------
INK   = RGBColor(0x1A, 0x1F, 0x2B)   # near-black text
MUTED = RGBColor(0x5B, 0x63, 0x72)   # grey
BRAND = RGBColor(0x1F, 0x6F, 0xEB)   # blue accent
GREEN = RGBColor(0x1A, 0x7F, 0x37)   # results green
AMBER = RGBColor(0xB5, 0x6A, 0x00)   # limitation amber
BG    = RGBColor(0xFF, 0xFF, 0xFF)
PANEL = RGBColor(0xF4, 0xF6, 0xFA)

W, H = Inches(13.333), Inches(7.5)   # 16:9
prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


def slide():
    s = prs.slides.add_slide(BLANK)
    r = s.shapes.add_shape(1, 0, 0, W, H)
    r.fill.solid(); r.fill.fore_color.rgb = BG
    r.line.fill.background(); r.shadow.inherit = False
    s.shapes._spTree.remove(r._element); s.shapes._spTree.insert(2, r._element)
    return s


def box(s, x, y, w, h, fill=None, line=None):
    sp = s.shapes.add_shape(1, x, y, w, h)
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(1)
    sp.shadow.inherit = False
    return sp


def text(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         space_after=6, line_spacing=1.05):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        chunks = para if isinstance(para, list) else [para]
        for ch in chunks:
            t, sz, col, bold = ch
            r = p.add_run(); r.text = t
            r.font.size = Pt(sz); r.font.color.rgb = col
            r.font.bold = bold; r.font.name = "Calibri"
    return tb


def bullets(s, x, y, w, h, items, size=18, gap=10, col=INK, dot=BRAND):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap); p.line_spacing = 1.05
        lead = p.add_run(); lead.text = "•  "
        lead.font.size = Pt(size); lead.font.color.rgb = dot; lead.font.bold = True
        for ch in (it if isinstance(it, list) else [(it, False)]):
            t, bold = ch
            r = p.add_run(); r.text = t
            r.font.size = Pt(size); r.font.color.rgb = col
            r.font.bold = bold; r.font.name = "Calibri"
    return tb


def header(s, kicker, title):
    box(s, Inches(0), Inches(0), Inches(0.18), H, fill=BRAND)
    text(s, Inches(0.6), Inches(0.45), Inches(12), Inches(0.4),
         [[(kicker.upper(), 13, BRAND, True)]])
    text(s, Inches(0.6), Inches(0.8), Inches(12.1), Inches(0.9),
         [[(title, 30, INK, True)]])
    box(s, Inches(0.62), Inches(1.65), Inches(1.1), Pt(3), fill=BRAND)


# ========================================================================
# 1 — TITLE
# ========================================================================
s = slide()
box(s, Inches(0), Inches(0), W, H, fill=PANEL)
box(s, Inches(0), Inches(0), Inches(0.25), H, fill=BRAND)
text(s, Inches(0.9), Inches(1.7), Inches(11.5), Inches(0.5),
     [[("MSc PROJECT  ·  SUPERVISOR UPDATE", 15, BRAND, True)]])
text(s, Inches(0.9), Inches(2.3), Inches(11.6), Inches(1.8),
     [[("Time-Series Data Analysis", 44, INK, True)],
      [("Through a Natural HCI Interface", 44, INK, True)]], line_spacing=1.0)
text(s, Inches(0.92), Inches(4.4), Inches(11), Inches(1.2),
     [[("Ask your data in plain English — ", 20, MUTED, False),
       ("code computes the exact numbers", 20, BRAND, True),
       (",", 20, MUTED, False)],
      [("a local AI only explains them. Fully offline.", 20, MUTED, False)]],
     line_spacing=1.1)
box(s, Inches(0.92), Inches(5.6), Inches(3.2), Pt(3), fill=BRAND)
text(s, Inches(0.9), Inches(5.85), Inches(11), Inches(0.6),
     [[("v1 case study: Jena Climate (2009–2016)", 16, INK, True)]])
text(s, Inches(0.9), Inches(6.5), Inches(11), Inches(0.5),
     [[("Bello Ahmed   ·   29 June 2026", 14, MUTED, False)]])

# ========================================================================
# 2 — WHAT HAS BEEN DONE (brief summary)
# ========================================================================
s = slide()
header(s, "Progress", "What has been done so far")
bullets(s, Inches(0.7), Inches(2.0), Inches(12), Inches(4.2), [
    [("Built the full v1 system", True),
     (" on the Jena climate data (70k hourly readings, 14 sensors).", False)],
    [("Natural-language interface", True),
     (" — ask about extremes, any year / season / day, thresholds, comparisons, trends, forecasts and anomalies.", False)],
    [("Trust design", True),
     (": exact numbers come from code; the local AI only rewords them — it never calculates.", False)],
    [("Anomaly detection + explanation", True),
     (" across sensors (e.g. “why was 26 Jan unusual?”).", False)],
    [("Evaluated & verified", True),
     (": results match pandas exactly, AI invented 0 numbers, 17/17 checks and 48 tests pass.", False)],
    [("Working demo website", True),
     (" + command-line tool; code pushed to GitHub.", False)],
], size=18, gap=14)

# ========================================================================
# 3 — DEMO WEBSITE
# ========================================================================
s = slide()
header(s, "Live demo", "The demo website")
box(s, Inches(0.6), Inches(2.0), Inches(7.4), Inches(4.5), fill=PANEL, line=RGBColor(0xDD,0xE3,0xEC))
# mock browser
box(s, Inches(0.9), Inches(2.3), Inches(6.8), Inches(0.55), fill=BG, line=RGBColor(0xDD,0xE3,0xEC))
for i, c in enumerate([RGBColor(0xE5,0x6B,0x6B), RGBColor(0xE6,0xB0,0x4A), RGBColor(0x5D,0xB0,0x6B)]):
    box(s, Emu(Inches(1.05)+i*Inches(0.28)), Inches(2.5), Inches(0.16), Inches(0.16), fill=c)
text(s, Inches(2.1), Inches(2.36), Inches(5.4), Inches(0.45),
     [[("localhost:3030", 13, MUTED, False)]], anchor=MSO_ANCHOR.MIDDLE)
box(s, Inches(0.9), Inches(2.95), Inches(6.8), Inches(0.6), fill=BG, line=BRAND)
text(s, Inches(1.1), Inches(3.0), Inches(6.4), Inches(0.5),
     [[("Ask:  ", 14, MUTED, False), ("what was the hottest temperature, and when?", 14, INK, True)]],
     anchor=MSO_ANCHOR.MIDDLE)
box(s, Inches(0.9), Inches(3.75), Inches(6.8), Inches(2.5), fill=BG, line=RGBColor(0xDD,0xE3,0xEC))
text(s, Inches(1.1), Inches(3.9), Inches(6.4), Inches(2.2),
     [[("🔢  37.04 °C on 4 Jul 2015", 16, GREEN, True)],
      [("ℹ️  Coincides with the July 2015 European heatwave.", 13.5, MUTED, False)],
      [("🧠  Confirmed across sensors — a real event, not a glitch.", 13.5, MUTED, False)],
      [("[ chart: temperature with the peak marked ]", 12.5, BRAND, False)]],
     line_spacing=1.2)
bullets(s, Inches(8.3), Inches(2.15), Inches(4.5), Inches(4.3), [
    [("Plain-English box", True), (" — no formulas or menus", False)],
    [("Trust labels", True), (":  🔢 exact figure · ℹ️ background · 🧠 reasoning", False)],
    [("“Explain with AI” toggle", True), (" — same numbers, friendlier wording", False)],
    [("Inline charts", True), (" with the answer marked", False)],
    [("100% local", True), (" — runs at localhost:3030, no cloud", False)],
], size=15.5, gap=14)
text(s, Inches(8.3), Inches(6.35), Inches(4.5), Inches(0.6),
     [[("▶  Live demo during the meeting", 15, BRAND, True)]])

# ========================================================================
# 4 — ARCHITECTURE & PACKAGES
# ========================================================================
s = slide()
header(s, "Under the hood", "Architecture & packages used")
# pipeline strip
y = Inches(2.0); bh = Inches(1.15); bw = Inches(2.18); gx = Inches(0.28)
steps = [("Question", MUTED), ("Parse intent", MUTED), ("Query engine\n(EXACT)", GREEN),
         ("Reasoning", MUTED), ("AI explains\n+ trust labels", BRAND)]
x = Inches(0.6)
for i, (t, col) in enumerate(steps):
    fill = RGBColor(0xE7,0xF4,0xEC) if col==GREEN else (RGBColor(0xE8,0xF1,0xFE) if col==BRAND else PANEL)
    box(s, x, y, bw, bh, fill=fill, line=col)
    text(s, x, y, bw, bh, [[(t, 14.5, col, True)]], align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.0)
    if i < 4:
        text(s, x+bw-Inches(0.02), y, gx+Inches(0.1), bh, [[("›", 24, BRAND, True)]],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    x = Emu(x + bw + gx)
# package columns
cols = [
    ("Data & analysis", [("pandas", "numpy"), ("scikit-learn", "statsmodels"),
                          ("(stats, STL,", "Isolation Forest,"), ("ARIMA forecast)", "")]),
    ("Deep models", [("PyTorch", "(GPU)"), ("CNN & LSTM", "autoencoders"),
                     ("LSTM", "forecaster"), ("", "")]),
    ("Local AI / NLP", [("Ollama", "+ llama3.2:3b"), ("TF-IDF", "retriever (RAG)"),
                        ("explains only,", "never computes"), ("", "")]),
    ("Interface & test", [("Flask", "web app (3030)"), ("HTML / JS", "+ SVG charts"),
                          ("Python CLI", "chat tool"), ("pytest", "(48 tests)")]),
]
cw = Inches(2.95); gx = Inches(0.18); x = Inches(0.6); y = Inches(3.5)
for title, rows in cols:
    box(s, x, y, cw, Inches(2.95), fill=PANEL, line=RGBColor(0xDD,0xE3,0xEC))
    box(s, x, y, cw, Inches(0.55), fill=BRAND)
    text(s, x, y, cw, Inches(0.55), [[(title, 14.5, BG, True)]],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    runs = []
    for a, b in rows:
        if a == "" and b == "":
            continue
        line = [(a + ("  " if a and b else ""), 14, INK, bool(a))]
        if b:
            line.append((b, 13, MUTED, False))
        runs.append(line)
    text(s, x+Inches(0.25), y+Inches(0.7), cw-Inches(0.45), Inches(2.1), runs,
         line_spacing=1.15, space_after=5)
    x = Emu(x + cw + gx)
text(s, Inches(0.6), Inches(6.65), Inches(12), Inches(0.5),
     [[("Python 3.12  ·  fully offline  ·  no paid APIs", 14, MUTED, True)]])

# ========================================================================
# 5 — LIMITATIONS
# ========================================================================
s = slide()
header(s, "Honest view", "Limitations")
bullets(s, Inches(0.7), Inches(2.0), Inches(12), Inches(4.3), [
    [("Getting good datasets is the main constraint", True),
     (" — clean, labelled, multi-sensor time-series are hard to find.", False)],
    [("No real anomaly labels", True),
     (" in the Jena data, so detector accuracy is measured on injected (synthetic) spikes.", False)],
    [("Tied to one dataset / schema", True),
     (" right now — the parser is tuned to temperature-style climate columns.", False)],
    [("Small local AI", True),
     (" (llama3.2:3b) — good for wording, occasionally drops a secondary value (never invents one).", False)],
    [("Short-horizon forecasting", True),
     (" (~2 weeks) and heuristic, single-domain reasoning.", False)],
], size=18, gap=16, dot=AMBER)

# ========================================================================
# 6 — FUTURE PLANS
# ========================================================================
s = slide()
header(s, "Roadmap", "Future plans / next updates")
# headline future feature
box(s, Inches(0.6), Inches(2.0), Inches(12.1), Inches(1.45), fill=RGBColor(0xE8,0xF1,0xFE), line=BRAND)
text(s, Inches(0.95), Inches(2.2), Inches(11.5), Inches(1.1),
     [[("★  Next big update:  ", 19, BRAND, True),
       ("make it read ANY dataset", 19, INK, True),
       (" — upload a CSV and ask questions, not just climate data.", 19, INK, False)],
      [("Auto-detect columns, time index and sensors so the same trust chain works on any time-series.", 15, MUTED, False)]])
bullets(s, Inches(0.7), Inches(3.85), Inches(12), Inches(3.0), [
    [("Generalise the interface", True), (" to arbitrary CSV/Excel schemas (the headline feature above).", False)],
    [("Run the HCI user study", True), (" already designed — real usability + trust evidence.", False)],
    [("Add a real labelled benchmark", True), (" (e.g. NAB / SKAB) to strengthen anomaly accuracy.", False)],
    [("Larger / fine-tuned local AI", True), (" for richer explanations, still offline.", False)],
    [("Second case study (v2: trade data)", True), (" to prove it transfers across domains.", False)],
], size=17, gap=12)

# ========================================================================
out = __file__.rsplit("/", 1)[0] + "/Supervisor_Update_v1.pptx"
prs.save(out)
print("saved:", out, "| slides:", len(prs.slides._sldIdLst))
