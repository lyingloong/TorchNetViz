import itertools
import math
import os
import networkx as nx
import plotly.graph_objects as go
from typing import Optional, Union

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def visualize_structure(
    structure: list[dict[str, str]],
    connections: list[dict[str, Union[str, None]]],
    inputs: Optional[list[str]] = None,
    output_path: Optional[str] = None,
    showgrid: bool = False
) -> go.Figure:
    """
    Visualize model architecture with clear layered layout, skip connections, and interactivity.
    """

    def _create_nodes(structure: list[dict[str, str]]) -> tuple[nx.DiGraph, dict]:
        G = nx.DiGraph()
        layer_types = {}
        for layer in structure:
            name = layer['name']
            ltype = layer.get('type', 'Unknown')
            G.add_node(name, layer_type=ltype)
            layer_types[name] = ltype
        return G, layer_types

    G, layer_types = _create_nodes(structure)

    def _classify_edges(G: nx.DiGraph, connections: list[dict[str, Union[str, None]]]) -> tuple[list, list, list]:
        normal_edges, skip_edges, output_edges = [], [], []
        for conn in connections:
            src, tgt = conn['source'], conn['target']
            etype = conn.get('type', None)
            if etype == 'skip':
                skip_edges.append((src, tgt))
            elif etype == 'output':
                output_edges.append((src, tgt))
            else:
                normal_edges.append((src, tgt))
            G.add_edge(src, tgt)
        return normal_edges, skip_edges, output_edges

    normal_edges, skip_edges, output_edges = _classify_edges(G, connections)

    def _add_input_nodes(G: nx.DiGraph, inputs: Optional[list[str]], structure: list[dict[str, str]],
                         normal_edges: list) -> list:
        if inputs:
            for inp in inputs:
                G.add_node(inp, layer_type='Input')
                first_node = structure[0]['name']
                G.add_edge(inp, first_node)
                normal_edges.insert(0, (inp, first_node))
        return normal_edges

    normal_edges = _add_input_nodes(G, inputs, structure, normal_edges)

    # 拓扑排序 + 分层布局
    def topological_layered_layout(G: nx.DiGraph) -> dict:
        layers = {}
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            if not preds:
                layers[node] = 0
            else:
                layers[node] = max(layers[p] for p in preds) + 1

        # 分组层
        layer_nodes = {}
        for node, layer in layers.items():
            layer_nodes.setdefault(layer, []).append(node)

        # 排布坐标：每层 y 固定，x 均匀
        pos = {}
        for y_level, nodes in layer_nodes.items():
            num = len(nodes)
            for i, node in enumerate(nodes):
                x = i - num / 2
                y = -y_level  # y 从上往下
                pos[node] = (x, y)
        return pos

    pos = topological_layered_layout(G)
    print(f"Node positions computed: {pos}")

    import random
    random.seed(42)
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173"
    ]
    color_cycle = itertools.cycle(palette)
    all_types = set(nx.get_node_attributes(G, 'layer_type').values())
    type_to_color = {lt: next(color_cycle) for lt in sorted(all_types)}
    type_to_color.update({"Unknown": "#7f7f7f"})

    # --- 边绘制 ---
    def _create_edge_trace(edge_list, dash_style):
        x, y = [], []
        for u, v in edge_list:
            x += [pos[u][0], pos[v][0], None]
            y += [pos[u][1], pos[v][1], None]
        return go.Scatter(
            x=x, y=y, mode='lines',
            line=dict(width=2, dash=dash_style),
            hoverinfo='none'
        )

    edge_traces = [
        _create_edge_trace(normal_edges, 'solid'),
        _create_edge_trace(skip_edges, 'dash'),
        _create_edge_trace(output_edges, 'dot')
    ]

    # --- 节点绘制 ---
    node_x, node_y, node_text, node_color = [], [], [], []
    for node, data in G.nodes(data=True):
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
        label = f"{node} ({data.get('layer_type', 'Unknown')})"
        node_text.append(label)
        node_color.append(type_to_color[data.get('layer_type', 'Unknown')])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            color=node_color,
            size=24,
            line=dict(width=2, color='black'),
            symbol='circle'
        )
    )

    # --- 布局绘制 ---
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=showgrid, zeroline=False, visible=False),
        yaxis=dict(showgrid=showgrid, zeroline=False, visible=False),
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=20),
    )

    # --- 保存或显示 ---
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        print(f"Model visualization saved to {output_path}")
    else:
        fig.show()

    return fig