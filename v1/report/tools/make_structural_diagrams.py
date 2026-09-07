"""Render the two structural diagrams as PNG, following standard UML conventions rather than a
generic auto-layout flowchart look:

  fig_architecture.png    -- a UML *component diagram*: the five layers as components (the small
                              twin-tab icon is the standard UML component symbol), connected by
                              dashed, open-arrowhead UML *dependency* arrows (higher layer depends
                              on the layer below it).
  fig_logic_flowchart.png -- a UML *activity diagram*: filled-circle start node, rounded action
                              states, a diamond decision node with bracketed guard conditions
                              ([yes]/[no], [pass]/[fail] -- the UML notation for a guard), and a
                              bullseye activity-final node.

Monochrome ink on white throughout (one grey fill reserved for the query-engine component, the
single layer the whole architecture's correctness depends on); no gradients, no drop shadows.

Run from code/v1:  python3.12 report/tools/make_structural_diagrams.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "Liberation Serif"   # metric-compatible with Times New Roman

INK = "#111111"
MUTED = "#444444"
GREY_FILL = "#e6e6e6"


def straight_arrow(ax, p1, p2, color=INK, lw=1.1, ls="-", style="-|>", mutation=10):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=mutation, linewidth=lw,
                         color=color, linestyle=ls, shrinkA=0, shrinkB=0,
                         connectionstyle="arc3,rad=0.0", zorder=1)
    ax.add_patch(a)


def diamond(ax, cx, cy, w, h, edge=INK, face="white", lw=1.1):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy), (cx, cy + h / 2)]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(pts, codes), linewidth=lw, edgecolor=edge, facecolor=face, zorder=2))


# ---------------------------------------------------------------------------
# 1. UML component diagram -- five layers, one guaranteed layer
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 8.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12.9)
ax.axis("off")
ax.set_title("System architecture (UML component diagram)", fontsize=12.5, fontweight="bold",
             pad=12, family="Liberation Serif")

layers = [
    ("Interface", "\xabcomponent\xbb", "question -> intent + trust-labelled answer", "white"),
    ("Explanation", "\xabcomponent\xbb",
     "reasoning + retrieval + optional LLM rewording, faithfulness-gated", "white"),
    ("Query engine", "\xabcomponent\xbb -- truth layer",
     "the only component that creates a number; checked against an independent pandas reference",
     GREY_FILL),
    ("Detection", "\xabcomponent\xbb", "anomaly detectors + forecaster", "white"),
    ("Data", "\xabcomponent\xbb", "preprocessing, resampling, rolling statistics", "white"),
]

x0, w = 0.9, 8.2
h, gap = 1.9, 0.45
y = 12.9 - 0.4 - h
tops, bots = [], []
for name, stereotype, sub, face in layers:
    ax.add_patch(Rectangle((x0, y), w, h, linewidth=1.2, edgecolor=INK, facecolor=face, zorder=2))
    # UML component icon: two small tabs straddling the box's left edge
    icon_x = x0 + w - 0.62
    icon_y = y + h - 0.46
    ax.add_patch(Rectangle((icon_x, icon_y), 0.34, 0.15, linewidth=1.0, edgecolor=INK,
                            facecolor="white", zorder=3))
    ax.add_patch(Rectangle((icon_x, icon_y - 0.24), 0.34, 0.15, linewidth=1.0, edgecolor=INK,
                            facecolor="white", zorder=3))
    ax.text(x0 + w / 2, y + h - 0.32, stereotype, ha="center", va="center", fontsize=8,
            style="italic", color=MUTED)
    ax.text(x0 + w / 2, y + h - 0.62, name, ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=INK)
    ax.text(x0 + w / 2, y + 0.46, sub, ha="center", va="center", fontsize=8.4, color=MUTED, wrap=True)
    tops.append((x0 + w / 2, y + h))
    bots.append((x0 + w / 2, y))
    y -= (h + gap)

# UML dependency arrows: dashed line, open (stick) arrowhead, client -> supplier, «use»-labelled
for i in range(len(bots) - 1):
    straight_arrow(ax, bots[i], tops[i + 1], ls=(0, (5, 3)), style="->", mutation=13)
    mid = ((bots[i][0] + tops[i + 1][0]) / 2 + 0.55, (bots[i][1] + tops[i + 1][1]) / 2)
    ax.text(*mid, "\xabuse\xbb", fontsize=7.6, style="italic", color=MUTED, ha="left", va="center")

# faithfulness-gate note, plain leader line
gate_y = bots[1][1] + h / 2
ax.annotate(r"faithfulness gate: $C \subseteq V$" "\nfails -> deterministic template",
            xy=(x0 + w, gate_y), xytext=(x0 + w + 0.25, gate_y - 0.95),
            fontsize=8.2, color=MUTED, ha="left", family="Liberation Serif",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))

fig.tight_layout()
fig.savefig(OUT / "fig_architecture.png", dpi=200)
plt.close(fig)
print("saved fig_architecture.png")

# ---------------------------------------------------------------------------
# 2. UML activity diagram -- query lifecycle + faithfulness-gate fallback loop
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 10.6))
ax.set_xlim(0, 10.2)
ax.set_ylim(0, 15.6)
ax.axis("off")
ax.set_title("Query lifecycle (UML activity diagram)", fontsize=12.5, fontweight="bold", pad=10,
             family="Liberation Serif")

cx = 4.7           # main spine (start, parse, query engine, first decision)
lx = 2.4           # left column: the deterministic / no-LLM default path
rx = 7.3           # right column: the LLM-explained path
bw, bh = 5.0, 0.95


def action(cy, text, w=bw, cx_=cx, h=bh):
    """A UML action state: a rounded rectangle."""
    x, y = cx_ - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.14",
                                 linewidth=1.15, edgecolor=INK, facecolor="white", zorder=2))
    ax.text(cx_, cy, text, ha="center", va="center", fontsize=9.4, fontweight="bold", color=INK)
    return (cx_, y + h), (cx_, y)


def merge_node(cx_, cy_, w=0.5, h=0.42):
    """A UML merge node: same diamond glyph as a decision, used with >=2 incoming edges and
    exactly one outgoing edge (no guards -- unconditional convergence)."""
    diamond(ax, cx_, cy_, w, h)
    return (cx_, cy_ + h / 2), (cx_, cy_ - h / 2), (cx_ - w / 2, cy_), (cx_ + w / 2, cy_)


def guard(x, y, text, ha="center"):
    ax.text(x, y, text, fontsize=8.6, style="italic", color=MUTED, ha=ha, va="center")


# initial node
start = (cx, 15.05)
ax.add_patch(Circle(start, 0.14, facecolor=INK, edgecolor=INK, zorder=3))

p_top, p_bot = action(14.15, "parse: intent, sensor, scope\n(deterministic)")
e_top, e_bot = action(12.65, "query engine computes facts\n-- the truth layer")
straight_arrow(ax, start, p_top)
straight_arrow(ax, p_bot, e_top)

# decision node: is a local LLM available at all?
dcy = 11.05
diamond(ax, cx, dcy, 3.3, 1.2)
ax.text(cx, dcy, "local LLM\navailable?", ha="center", va="center", fontsize=9, fontweight="bold")
straight_arrow(ax, e_bot, (cx, dcy + 0.6))

# [no] -> the deterministic template, down the left column
no_top, no_bot = action(9.35, "use deterministic template\n(exact figure, no LLM)", w=3.6, cx_=lx)
straight_arrow(ax, (cx - 1.65, dcy), no_top)
guard(lx - 0.35, dcy - 1.05, "[no]", ha="right")

# [yes] -> LLM phrasing -> faithfulness gate, down the right column
llm_top, llm_bot = action(9.35, "LLM phrases the answer\n(rewording only)", w=3.4, cx_=rx)
straight_arrow(ax, (cx + 1.65, dcy), llm_top)
guard(rx + 0.35, dcy - 1.05, "[yes]", ha="left")

gdcy = 7.35
diamond(ax, rx, gdcy, 3.0, 1.15)
ax.text(rx, gdcy, r"$C \subseteq V$?", ha="center", va="center", fontsize=9.4, fontweight="bold")
straight_arrow(ax, llm_bot, (rx, gdcy + 0.575))

fail_top, fail_bot = action(5.55, "discard phrasing", w=2.0, cx_=6.1, h=0.8)
pass_top, pass_bot = action(5.55, "show LLM prose", w=2.0, cx_=8.5, h=0.8)
straight_arrow(ax, (rx - 0.5, gdcy - 0.575), fail_top)
straight_arrow(ax, (rx + 0.5, gdcy - 0.575), pass_top)
guard(5.35, gdcy - 0.95, "[fail]")
guard(9.25, gdcy - 0.95, "[pass]")

# convergence: three outcomes (no-LLM template / gate-failed / gate-passed) all reach one answer.
# UML merges take exactly two incoming edges each, so two chained merge nodes combine three paths.
m1_top, m1_bot, m1_left, m1_right = merge_node(lx, 3.85)
straight_arrow(ax, no_bot, m1_top)                 # long straight run down the left column
straight_arrow(ax, fail_bot, m1_right)              # diagonal in from the discard-phrasing branch

m2_top, m2_bot, m2_left, m2_right = merge_node(lx, 2.75)
straight_arrow(ax, m1_bot, m2_top)
straight_arrow(ax, pass_bot, m2_right)              # diagonal in from the show-prose branch

ans_top, ans_bot = action(1.55, "answer shown to user\n(trust-labelled)", w=3.6, cx_=lx)
straight_arrow(ax, m2_bot, ans_top)

# activity-final node (bullseye)
end = (lx, 0.35)
ax.add_patch(Circle(end, 0.19, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=3))
ax.add_patch(Circle(end, 0.1, facecolor=INK, edgecolor=INK, zorder=3))
straight_arrow(ax, ans_bot, (lx, end[1] + 0.19))

ax.set_ylim(-0.1, 15.6)

fig.tight_layout()
fig.savefig(OUT / "fig_logic_flowchart.png", dpi=200)
plt.close(fig)
print("saved fig_logic_flowchart.png")
