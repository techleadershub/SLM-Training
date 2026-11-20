"""Draw a compiled LangGraph graph as a picture (boxes and arrows), without any web service.

    from graph_viz import draw_graph
    draw_graph(compiled_graph, "my graph")

Solid arrows are fixed edges; dashed arrows are conditional edges (the graph chooses). Arrows that go back to an
earlier box are loops.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_graph(app, title=None, width=None):
    graph = app.get_graph()
    names = list(graph.nodes)
    edges = [(e.source, e.target, bool(e.conditional)) for e in graph.edges]

    # level = distance from the start box (breadth-first), so the picture reads left to right
    level = {"__start__": 0}
    queue = ["__start__"]
    while queue:
        node = queue.pop(0)
        for src, dst, _ in edges:
            if src == node and dst not in level:
                level[dst] = level[node] + 1
                queue.append(dst)
    for n in names:
        level.setdefault(n, 1)
    columns = {}
    for n in names:
        columns.setdefault(level[n], []).append(n)
    if "__end__" in names:                                    # the end box always sits in the last column
        last = max(columns) + 1 if len(columns[level["__end__"]]) > 1 else level["__end__"]
        columns[level["__end__"]].remove("__end__")
        columns = {k: v for k, v in columns.items() if v}
        columns.setdefault(last, []).append("__end__")
        level["__end__"] = last

    pos = {}
    for col, members in columns.items():
        for i, n in enumerate(members):
            pos[n] = (col * 2.6, (len(members) - 1) / 2 * 1.0 - i * 1.0)

    ncols, tall = len(columns), max(len(v) for v in columns.values())
    fig, ax = plt.subplots(figsize=(width or max(7, 2.6 * ncols), max(2.6, 0.9 * tall + 1.2)))
    for n, (x, y) in pos.items():
        label = {"__start__": "START", "__end__": "END"}.get(n, n)
        special = n in ("__start__", "__end__")
        ax.add_patch(FancyBboxPatch((x - 0.9, y - 0.28), 1.8, 0.56, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="#e8eef7" if not special else "#f2f2f2", ec="#4c78a8" if not special else "#888", lw=1.5))
        ax.text(x, y, label, ha="center", va="center", fontsize=9)
    for src, dst, cond in edges:
        (x1, y1), (x2, y2) = pos[src], pos[dst]
        back = level[dst] <= level[src]
        rad = 0.35 if back else (0.12 if abs(y1 - y2) > 0.01 else 0.0)
        ax.annotate("", xy=(x2 - 0.9 if not back else x2, y2 + (0.28 if back else 0)), xytext=(x1 + 0.9 if not back else x1, y1 + (0.28 if back else 0)),
                    arrowprops=dict(arrowstyle="->", lw=1.2, ls="--" if cond else "-", color="#e45756" if cond else "#555",
                                    connectionstyle=f"arc3,rad={rad}"))
    ax.set_xlim(-1.3, max(x for x, _ in pos.values()) + 1.3)
    ax.set_ylim(min(y for _, y in pos.values()) - 0.8, max(y for _, y in pos.values()) + 0.9)
    ax.axis("off")
    ax.set_title((title + "   " if title else "") + "(dashed red = the graph chooses)", fontsize=10)
    plt.tight_layout()
    plt.show()
