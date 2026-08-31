"""Interactive HTML visualization of sequence similarity networks."""

import json
import logging
from html import escape
from pathlib import Path

import networkx as nx

from natu.network import get_clusters

log = logging.getLogger(__name__)


# Categorical palette (see project data-viz guidelines), full 8-hue set. Only the first
# three slots are guaranteed to stay visually distinguishable from each other on an
# all-pairs-visible plot like this one (every node color can end up adjacent to any other),
# but a real clustering run routinely produces far more clusters than that -- per explicit
# request, colors are allowed to repeat rather than folding everything past a handful of
# clusters into a neutral "Other" bucket: each cluster gets colorsLight[rank % 8], cycling
# back to the start every eight clusters. This trades the strict color-safety guarantee for
# being able to tell individual clusters apart at all once there are more of them than
# palette slots.
_CLUSTER_COLORS_LIGHT = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
_CLUSTER_COLORS_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]
_OTHER_COLOR_LIGHT = "#898781"
_OTHER_COLOR_DARK = "#898781"
_LEGEND_MAX_CLUSTERS = len(_CLUSTER_COLORS_LIGHT)


def load_network(path: Path) -> nx.Graph:
    """
    Load a similarity network from a GraphML file (as written by ``natu cluster``).

    :param path: Path to a GraphML file.
    :return: networkx Graph.
    """
    return nx.read_graphml(path)


def _assign_cluster_slots(graph: nx.Graph) -> tuple[dict[str, int], list[tuple[int, int]]]:
    """
    Assign each node a cluster "slot": its cluster's rank by size (0 = largest cluster).

    Every cluster gets a real slot -- there is no folded "Other" bucket anymore. Rendering
    wraps the slot around the palette length (slot % len(colors)) so colors repeat once
    there are more clusters than palette entries; see the module-level palette comment.

    :param graph: Similarity network.
    :return: Tuple of (node -> slot mapping, list of (slot, cluster_size) for every
        cluster, largest first).
    """
    clusters = get_clusters(graph)  # already sorted largest first

    node_slot: dict[str, int] = {}
    cluster_sizes: list[tuple[int, int]] = []

    for slot, cluster in enumerate(clusters):
        cluster_sizes.append((slot, len(cluster)))
        for node in cluster:
            node_slot[node] = slot

    return node_slot, cluster_sizes


def _node_label(graph: nx.Graph, node: str) -> str:
    """
    The display string for a node: its polymer sequence if the graph carries one (written
    by ``natu.network.cluster_sequences``), otherwise its header/id.

    This is the single source of truth for "what a node is called" -- both the hover
    tooltip label and --highlight matching go through this function, so whatever a user
    sees on hover is exactly what they can paste into --highlight.

    :param graph: Similarity network.
    :param node: Node id.
    :return: Display string for the node.
    """
    return graph.nodes[node].get("sequence") or node


def _contains_subsequence(needle: str, haystack: str) -> bool:
    """
    Whether ``needle``'s monomers occur as a contiguous run inside ``haystack``'s, both
    given in the pipe-joined label format (e.g. "leucine|glycine" inside
    "serine|leucine|glycine|alanine").

    Matching happens on the split monomer lists, not the raw joined strings, so a query can
    never accidentally match by straddling a "|" boundary or matching only part of a longer
    monomer name (e.g. "serine" must never match inside a hypothetical "D-serine").

    :param needle: Query sequence, pipe-joined.
    :param haystack: Candidate node sequence, pipe-joined.
    :return: True if needle's monomers occur, in order and contiguously, in haystack's.
    """
    needle_monomers = needle.split("|") if needle else []
    haystack_monomers = haystack.split("|") if haystack else []

    n, m = len(needle_monomers), len(haystack_monomers)
    if n == 0 or n > m:
        return False

    return any(haystack_monomers[i : i + n] == needle_monomers for i in range(m - n + 1))


def _assign_highlight_slots(
    graph: nx.Graph, highlight: list[str], contains: bool = False
) -> tuple[dict[str, int], list[dict[str, object]]]:
    """
    Assign cluster slots for highlight mode: only the cluster(s) containing a node that
    matches one of the requested ``highlight`` strings get a real color slot (-1, the
    neutral background color, for every other node).

    By default matching is exact, against the same string ``_node_label`` uses for a
    node's hover label -- a node's polymer sequence when the network file carries one, its
    header otherwise -- so whatever ``natu draw``'s tooltip shows for a node is exactly what
    can be pasted in as a --highlight value. When ``contains`` is True, matching instead
    accepts any node whose full sequence contains the query as a contiguous run of monomers
    (see ``_contains_subsequence``) -- so a short motif like "leucine|glycine" highlights
    every cluster with a node that has that pair back-to-back anywhere in its sequence, not
    just a node whose entire sequence is exactly that pair. This is a strict superset of
    exact matching (a sequence always "contains" itself in full).

    Highlight queries are assigned colors in the order given (cycling past 8 the same way
    normal cluster coloring does). If two queries both match nodes in the same cluster, the
    first query (in argument order) to claim that cluster keeps it -- the later query's
    legend row still reports what it matched, just reusing that color rather than being
    silently dropped. A query matching nothing in the network gets its own legend row
    saying so, rather than disappearing without explanation.

    :param graph: Similarity network.
    :param highlight: Sequence strings to highlight, in the order given on the command line
        (i.e. ``args.highlight`` from repeated -H/--highlight flags, plus anything read from
        --highlight-fasta).
    :param contains: If True, match by containment (a node's sequence contains the query as
        a contiguous run of monomers) instead of exact equality.
    :return: Tuple of (node -> slot mapping, legend entries). Legend entries are
        ``{"label": str, "slot": int}``, with ``slot`` possibly -1 for a no-match or
        "Other" row.
    """
    clusters = get_clusters(graph)  # largest first

    node_to_cluster_idx: dict[str, int] = {}
    for idx, cluster in enumerate(clusters):
        for node in cluster:
            node_to_cluster_idx[node] = idx

    node_slot: dict[str, int] = dict.fromkeys(graph.nodes, -1)
    cluster_slot: dict[int, int] = {}
    legend_entries: list[dict[str, object]] = []
    match_kind = "contains" if contains else "exact match"

    def matches(label: str, query: str) -> bool:
        return _contains_subsequence(query, label) if contains else label == query

    next_slot = 0
    for query in highlight:
        matching_nodes = [node for node in graph.nodes if matches(_node_label(graph, node), query)]

        if not matching_nodes:
            log.warning(
                "draw --highlight: no node in the network matches sequence %r (%s)", query, match_kind
            )
            legend_entries.append({"label": f'"{query}" -- no match found ({match_kind})', "slot": -1})
            continue

        matched_cluster_indices = sorted({node_to_cluster_idx[node] for node in matching_nodes})
        newly_claimed = [ci for ci in matched_cluster_indices if ci not in cluster_slot]

        if newly_claimed:
            slot = next_slot
            next_slot += 1
            for ci in newly_claimed:
                cluster_slot[ci] = slot
        else:
            # Every cluster this query matched was already claimed by an earlier query --
            # still report the match, reusing that color instead of assigning a new one.
            slot = cluster_slot[matched_cluster_indices[0]]

        total_nodes = sum(len(clusters[ci]) for ci in matched_cluster_indices)
        n_clusters = len(matched_cluster_indices)
        cluster_word = "cluster" if n_clusters == 1 else "clusters"
        legend_entries.append(
            {
                "label": f'"{query}" ({match_kind}, {total_nodes} nodes, {n_clusters} {cluster_word})',
                "slot": slot,
            }
        )

    for cluster_idx, slot in cluster_slot.items():
        for node in clusters[cluster_idx]:
            node_slot[node] = slot

    background_nodes = sum(1 for slot in node_slot.values() if slot < 0)
    if background_nodes:
        legend_entries.append({"label": f"Other ({background_nodes} nodes)", "slot": -1})

    return node_slot, legend_entries


def _normalize_positions(
    positions: dict[str, tuple[float, float]], padding: float = 40.0, scale: float = 900.0
) -> dict[str, tuple[float, float]]:
    """
    Rescale a networkx layout (typically in roughly [-1, 1]) to a fixed-size world space.

    :param positions: Node -> (x, y) mapping, as returned by a networkx layout function.
    :param padding: Padding (in world units) added around the rescaled layout.
    :param scale: Target width/height (in world units) of the layout before padding.
    :return: Node -> (x, y) mapping in world space.
    """
    if not positions:
        return {}

    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0

    normalized = {}
    for node, (x, y) in positions.items():
        nx_ = padding + (x - x_min) / x_range * scale
        ny_ = padding + (y - y_min) / y_range * scale
        normalized[node] = (nx_, ny_)

    return normalized


def network_to_html(
    graph: nx.Graph,
    title: str = "NATU Similarity Network",
    seed: int = 0,
    highlight: list[str] | None = None,
    highlight_contains: bool = False,
) -> str:
    """
    Render a similarity network as a self-contained, interactive HTML page.

    Uses a canvas (not per-node SVG/DOM elements) so it stays responsive on networks with
    many thousands of nodes and edges. Supports pan (drag), zoom (scroll), and hover
    (highlights a node's neighbors and shows a tooltip with its sequence, header, cluster,
    and degree). No external scripts or stylesheets are loaded, so the file works fully
    offline.

    :param graph: Similarity network, as returned by ``load_network`` or
        ``natu.network.build_similarity_network``.
    :param title: Page title, shown in the browser tab and page header.
    :param seed: Random seed for the spring layout, for reproducible node placement.
    :param highlight: If given, switches from the default rank-based cluster coloring to
        highlight mode: only the cluster(s) containing a matching node get a real color
        (one each, in the order given); every other node/edge is rendered in a single
        neutral background color, dimmed to make the highlighted cluster(s) stand out.
        ``None`` or an empty list keeps the default behavior. See ``highlight_contains`` for
        what "matching" means.
    :param highlight_contains: If True, a node matches a highlight query when the query's
        monomers occur as a contiguous run anywhere in the node's sequence (e.g.
        "leucine|glycine" matches "serine|leucine|glycine|alanine"), instead of requiring
        the node's whole sequence to equal the query. Ignored when ``highlight`` is not
        given.
    :return: Complete standalone HTML document as a string.
    """
    highlight_mode = bool(highlight)
    if highlight_mode:
        node_slot, legend_entries = _assign_highlight_slots(graph, highlight, contains=highlight_contains)
    else:
        node_slot, cluster_sizes = _assign_cluster_slots(graph)
        # Every cluster has a real color slot now (see _assign_cluster_slots), but listing
        # thousands of clusters in the legend would be useless -- show one row per palette
        # color (the clusters that get a color no other listed cluster shares) and
        # summarize the rest, since anything past this point repeats a hue already shown
        # above it.
        legend_entries = []
        for slot, size in cluster_sizes[:_LEGEND_MAX_CLUSTERS]:
            legend_entries.append({"label": f"Cluster {slot + 1} ({size} nodes)", "slot": slot})
        remaining = cluster_sizes[_LEGEND_MAX_CLUSTERS:]
        if remaining:
            remaining_nodes = sum(size for _, size in remaining)
            legend_entries.append(
                {
                    "label": f"+{len(remaining)} more clusters ({remaining_nodes} nodes, colors repeat)",
                    "slot": None,
                }
            )

    positions = nx.spring_layout(graph, seed=seed) if graph.number_of_nodes() > 0 else {}
    positions = _normalize_positions(positions)

    node_list = list(graph.nodes())
    node_index = {node: i for i, node in enumerate(node_list)}

    nodes_json = []
    for node in node_list:
        x, y = positions.get(node, (0.0, 0.0))
        nodes_json.append(
            {
                "id": node,
                "label": _node_label(graph, node),
                "x": round(x, 2),
                "y": round(y, 2),
                "slot": node_slot.get(node, 0),
                "degree": graph.degree[node],
            }
        )

    edges_json = []
    for u, v, data in graph.edges(data=True):
        edges_json.append(
            {
                "s": node_index[u],
                "t": node_index[v],
                "w": round(float(data.get("weight", 1.0)), 4),
            }
        )

    payload = {
        "nodes": nodes_json,
        "edges": edges_json,
        "legend": legend_entries,
        "colorsLight": _CLUSTER_COLORS_LIGHT,
        "colorsDark": _CLUSTER_COLORS_DARK,
        "otherLight": _OTHER_COLOR_LIGHT,
        "otherDark": _OTHER_COLOR_DARK,
        "highlightMode": highlight_mode,
    }

    stats = (
        f"{graph.number_of_nodes()} sequences &middot; {graph.number_of_edges()} edges "
        f"&middot; {len(get_clusters(graph))} clusters"
    )
    if highlight_mode:
        mode_note = "containing" if highlight_contains else "matching"
        stats += f" &middot; highlighting sequences {mode_note} " + ", ".join(f'"{h}"' for h in highlight)

    return _HTML_TEMPLATE.format(
        title=escape(title),
        stats=stats,
        payload_json=json.dumps(payload),
    )


def write_html(
    graph: nx.Graph,
    output: Path,
    title: str = "NATU Similarity Network",
    seed: int = 0,
    highlight: list[str] | None = None,
    highlight_contains: bool = False,
) -> None:
    """
    Render a similarity network to an interactive HTML file.

    :param graph: Similarity network to render.
    :param output: Output HTML file path.
    :param title: Page title.
    :param seed: Random seed for the spring layout.
    :param highlight: See ``network_to_html``.
    :param highlight_contains: See ``network_to_html``.
    """
    html = network_to_html(
        graph, title=title, seed=seed, highlight=highlight, highlight_contains=highlight_contains
    )
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(html)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --border: rgba(11,11,11,0.10);
    --edge-color: 137,135,129;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      color-scheme: dark;
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --border: rgba(255,255,255,0.10);
      --edge-color: 137,135,129;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--page);
    color: var(--text-primary);
  }}
  header {{
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--surface-1);
  }}
  h1 {{ font-size: 16px; margin: 0 0 4px 0; }}
  .stats {{ font-size: 13px; color: var(--text-secondary); }}
  .viz-wrap {{
    position: relative;
    height: calc(100vh - 64px);
    background: var(--surface-1);
  }}
  canvas {{ display: block; width: 100%; height: 100%; cursor: grab; }}
  canvas.dragging {{ cursor: grabbing; }}
  .legend {{
    position: absolute;
    top: 12px;
    left: 12px;
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
    color: var(--text-secondary);
  }}
  .legend-row {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 50%; flex: none; }}
  .hint {{
    position: absolute;
    bottom: 12px;
    left: 12px;
    font-size: 11px;
    color: var(--text-muted);
  }}
  .tooltip {{
    position: absolute;
    pointer-events: none;
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
    color: var(--text-primary);
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    display: none;
    max-width: 320px;
    word-break: break-word;
  }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="stats">{stats}</div>
</header>
<div class="viz-wrap">
  <canvas id="viz"></canvas>
  <div class="legend" id="legend"></div>
  <div class="hint">Drag to pan &middot; scroll to zoom &middot; hover a node for details</div>
  <div class="tooltip" id="tooltip"></div>
</div>
<script>
const DATA = {payload_json};

const canvas = document.getElementById("viz");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const legendEl = document.getElementById("legend");
const wrap = document.querySelector(".viz-wrap");

const isDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
const clusterColors = isDark ? DATA.colorsDark : DATA.colorsLight;
const otherColor = isDark ? DATA.otherDark : DATA.otherLight;

function colorForSlot(slot) {{
  // Real clusters always have slot >= 0 now; colors repeat by cycling through the
  // palette. `slot` is null/undefined only for the summary "+N more clusters" legend
  // row, which has no single color of its own.
  if (slot === null || slot === undefined || slot < 0) return otherColor;
  return clusterColors[slot % clusterColors.length];
}}

// Build adjacency (indices) for hover highlighting.
const neighbors = DATA.nodes.map(() => []);
for (const e of DATA.edges) {{
  neighbors[e.s].push(e.t);
  neighbors[e.t].push(e.s);
}}

// Legend
for (const entry of DATA.legend) {{
  const row = document.createElement("div");
  row.className = "legend-row";
  const sw = document.createElement("span");
  sw.className = "swatch";
  sw.style.background = colorForSlot(entry.slot);
  row.appendChild(sw);
  const label = document.createElement("span");
  label.textContent = entry.label;
  row.appendChild(label);
  legendEl.appendChild(row);
}}

let dpr = window.devicePixelRatio || 1;
let panX = 0, panY = 0, zoom = 1;
let dragging = false, lastX = 0, lastY = 0, moved = false;
let hoverIndex = -1;

function resize() {{
  const rect = wrap.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  if (zoom === 1 && panX === 0 && panY === 0) {{
    // Center the layout on first draw.
    panX = rect.width / 2 - 450;
    panY = rect.height / 2 - 450;
  }}
  draw();
}}

function worldToScreen(x, y) {{
  return [(x + panX) * zoom, (y + panY) * zoom];
}}

function screenToWorld(sx, sy) {{
  return [sx / zoom - panX, sy / zoom - panY];
}}

function nodeRadius(degree) {{
  return Math.min(3 + Math.sqrt(degree) * 1.3, 9);
}}

function draw() {{
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Edges
  for (const e of DATA.edges) {{
    const a = DATA.nodes[e.s], b = DATA.nodes[e.t];
    const dim = hoverIndex >= 0 && e.s !== hoverIndex && e.t !== hoverIndex;
    // In highlight mode, edges entirely within the un-highlighted background fade further
    // so the highlighted cluster's own edges stand out.
    const inBackground = DATA.highlightMode && a.slot < 0 && b.slot < 0;
    const baseAlpha = inBackground ? 0.03 : 0.10 + e.w * 0.5;
    const alpha = dim ? baseAlpha * 0.4 : baseAlpha;
    const [ax, ay] = worldToScreen(a.x, a.y);
    const [bx, by] = worldToScreen(b.x, b.y);
    ctx.strokeStyle = `rgba(137,135,129,${{alpha}})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }}

  // Nodes
  for (let i = 0; i < DATA.nodes.length; i++) {{
    const n = DATA.nodes[i];
    const [sx, sy] = worldToScreen(n.x, n.y);
    const r = nodeRadius(n.degree) * Math.min(zoom, 1.6);
    const isHover = i === hoverIndex;
    const isNeighbor = hoverIndex >= 0 && neighbors[hoverIndex].includes(i);
    const dim = hoverIndex >= 0 && !isHover && !isNeighbor;
    // In highlight mode, nodes outside every highlighted cluster recede into the
    // background by default (not just on hover), so the highlighted cluster(s) pop.
    const inBackground = DATA.highlightMode && n.slot < 0;
    const baseAlpha = inBackground ? 0.15 : 1.0;

    ctx.globalAlpha = dim ? baseAlpha * 0.4 : baseAlpha;
    ctx.beginPath();
    ctx.arc(sx, sy, isHover ? r * 1.4 : r, 0, Math.PI * 2);
    ctx.fillStyle = colorForSlot(n.slot);
    ctx.fill();
    if (isHover) {{
      ctx.lineWidth = 2;
      ctx.strokeStyle = isDark ? "#ffffff" : "#0b0b0b";
      ctx.stroke();
    }}
    ctx.globalAlpha = 1.0;
  }}

  ctx.restore();
}}

function findNodeAt(sx, sy) {{
  const [wx, wy] = screenToWorld(sx, sy);
  let best = -1, bestDist = Infinity;
  for (let i = 0; i < DATA.nodes.length; i++) {{
    const n = DATA.nodes[i];
    const dx = n.x - wx, dy = n.y - wy;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const r = nodeRadius(n.degree) + 4 / zoom;
    if (dist <= r && dist < bestDist) {{
      best = i;
      bestDist = dist;
    }}
  }}
  return best;
}}

canvas.addEventListener("mousedown", (ev) => {{
  dragging = true;
  moved = false;
  lastX = ev.clientX;
  lastY = ev.clientY;
  canvas.classList.add("dragging");
}});

window.addEventListener("mouseup", () => {{
  dragging = false;
  canvas.classList.remove("dragging");
}});

window.addEventListener("mousemove", (ev) => {{
  if (dragging) {{
    const dx = ev.clientX - lastX, dy = ev.clientY - lastY;
    if (Math.abs(dx) > 2 || Math.abs(dy) > 2) moved = true;
    panX += dx / zoom;
    panY += dy / zoom;
    lastX = ev.clientX;
    lastY = ev.clientY;
    draw();
    return;
  }}

  const rect = canvas.getBoundingClientRect();
  const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
  const idx = findNodeAt(sx, sy);

  if (idx !== hoverIndex) {{
    hoverIndex = idx;
    draw();
  }}

  if (idx >= 0) {{
    const n = DATA.nodes[idx];
    const clusterLabel = n.slot >= 0 ? `Cluster ${{n.slot + 1}}` : "Not highlighted";
    const escapedLabel = String(n.label).replace(/</g, "&lt;");
    const escapedId = String(n.id).replace(/</g, "&lt;");
    tooltip.style.display = "block";
    tooltip.style.left = (sx + 16) + "px";
    tooltip.style.top = (sy + 16) + "px";
    tooltip.innerHTML = `<strong>${{escapedLabel}}</strong><br>`
      + `<span style="opacity:0.7">${{escapedId}}</span><br>`
      + `${{clusterLabel}} &middot; degree ${{n.degree}}`;
  }} else {{
    tooltip.style.display = "none";
  }}
}});

canvas.addEventListener("wheel", (ev) => {{
  ev.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
  const [wx, wy] = screenToWorld(sx, sy);
  const factor = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
  zoom = Math.max(0.05, Math.min(20, zoom * factor));
  panX = sx / zoom - wx;
  panY = sy / zoom - wy;
  draw();
}}, {{ passive: false }});

window.addEventListener("resize", resize);
resize();
</script>
</body>
</html>
"""
