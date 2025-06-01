import logging
import itertools
import math
import os
import networkx as nx
import plotly.graph_objects as go
from typing import Optional, Union, Tuple, Dict, List, Any

from networkx import DiGraph

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger(__name__)

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

    def _create_nodes(structure: list[dict[str, str]]) -> DiGraph:
        G = nx.DiGraph()
        layer_types = {}
        for layer in structure:
            name = layer['name']
            ltype = layer.get('type', 'Unknown')
            G.add_node(name, layer_type=ltype)
            layer_types[name] = ltype
        return G

    G = _create_nodes(structure)

    def _classify_edges(G: nx.DiGraph, connections: list[dict[str, Union[str, None]]]) -> tuple[
        DiGraph, dict[str, list[Any]]]:
        edge_dictionary = {'normal': [], 'skip': [], 'output': []}
        for conn in connections:
            src, tgt = conn['source'], conn['target']
            etype = conn.get('type', 'normal')
            edge_dictionary.setdefault(etype, edge_dictionary['normal']).append((src, tgt))
            G.add_edge(src, tgt)
        return G, edge_dictionary
    
    G, edge_dict = _classify_edges(G, connections)

    def _add_input_nodes(G: nx.DiGraph, inputs: Optional[list[str]]) -> nx.DiGraph:
        if inputs:
            for inp in inputs:
                G.add_node(inp, layer_type='Input')
        return G
    def _add_output_nodes(G: nx.DiGraph, outputs: Optional[list[str]]) -> nx.DiGraph:
        if outputs:
            for out in outputs:
                G.add_node(out, layer_type='Output')
        return G

    G = _add_input_nodes(G, inputs)

    def _build_hierarchy(structure: list[dict[str, Any]]) -> dict[str, dict]:
        """构建模块层次结构字典"""
        hierarchy = {}
        for node in structure:
            if 'submodules' in node:
                hierarchy[node['name']] = {
                    'children': node['submodules'],
                    'depth': 0
                }
        return hierarchy

    hierarchy = _build_hierarchy(structure)

    def topological_layered_layout(G: nx.DiGraph, hierarchy: dict) -> dict:
        layers = {}
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            if not preds:
                layers[node] = 0
            else:
                layers[node] = max(layers[p] for p in preds) + 1

        # 嵌套处理：子节点在父节点下一层
        for parent, info in hierarchy.items():
            if parent not in layers:
                continue
            parent_layer = layers[parent]
            for child in info['children']:
                if child in layers:
                    layers[child] = max(layers[child], parent_layer + 1)

        # 分组
        layer_nodes = {}
        for node, layer in layers.items():
            layer_nodes.setdefault(layer, []).append(node)

        # 坐标计算
        pos = {}
        for y_level, nodes in layer_nodes.items():
            num = len(nodes)
            for i, node in enumerate(nodes):
                x = i - num / 2
                y = -y_level
                pos[node] = (x, y)
        return pos

    if not nx.is_directed_acyclic_graph(G):
        logger.info(f"Graph is not a DAG. Cycles found: {list(nx.simple_cycles(G))}")
        # 合并强连通组件为超级节点
        scc = list(nx.strongly_connected_components(G))
        mapping = {}
        component_info = {}
        for i, comp in enumerate(scc):
            if len(comp) > 1:
                comp_id = f"Component_{i}"
                component_info[comp_id] = list(comp)
                for node in comp:
                    mapping[node] = f"Component_{i}"
                logger.debug(f"Contracting component node {comp_id}, details: {component_info[comp_id]}")
        G_contracted = nx.DiGraph()

        # 添加组件节点和非组件节点
        for node, data in G.nodes(data=True):
            if node in mapping:
                comp_id = mapping[node]
                G_contracted.add_node(comp_id, layer_type="Component", details=component_info[comp_id])
            else:
                G_contracted.add_node(node, **data)

        # 添加边（跳过内部边）
        added_edges = set()
        for u, v in G.edges():
            src = mapping.get(u, u)
            tgt = mapping.get(v, v)
            if src != tgt and (src, tgt) not in added_edges:
                G_contracted.add_edge(src, tgt)
                added_edges.add((src, tgt))

        G = G_contracted
    else:
        logger.info("Graph is a DAG. No need to contract components.")

    pos = topological_layered_layout(G, hierarchy)
    logger.info(f"Node positions computed: {pos}")

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
    type_to_color.update({"Unknown": "#7f7f7f", "Component": "#a0522d"})

    # --- 边绘制 ---
    def _create_edge_trace(edge_list, dash_style):
        x, y = [], []
        for u, v in edge_list:
            if u not in pos or v not in pos:
                logger.warning(f"Skipping edge {u} -> {v} due to missing node position.")
                continue
            x += [pos[u][0], pos[v][0], None]
            y += [pos[u][1], pos[v][1], None]
        return go.Scatter(
            x=x, y=y, mode='lines',
            line=dict(width=2, dash=dash_style),
            hoverinfo='none'
        )
    
    logger.debug(f"Nodes in graph: {G.nodes}")
    logger.debug(f"Edges classified: normal {edge_dict['normal']} skip {edge_dict['skip']} output {edge_dict['output']}")

    edge_traces = [
        _create_edge_trace(edge_dict['normal'], 'solid'),
        _create_edge_trace(edge_dict['skip'], 'dash'),
        _create_edge_trace(edge_dict['output'], 'dot')
    ]

    # --- 节点绘制 ---
    node_x, node_y, node_text, node_color, node_customdata = [], [], [], [], []

    for node, data in G.nodes(data=True):
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
        if data.get("layer_type") == "Component":
            label = f"{node} (Composite)"
            details = ", ".join(sorted(data.get("details", [])))
        else:
            label = f"{node} ({data.get('layer_type', 'Unknown')})"
            details = ""
        node_text.append(label)
        node_customdata.append(details)
        node_color.append(type_to_color[data.get('layer_type', 'Unknown')])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        hoverinfo='text',
        customdata=node_customdata,
        hovertemplate='<b>%{text}</b><br><i>Details:</i><br>%{customdata}<extra></extra>',
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

    # 在fig上绘制父模块分组框
    for parent, info in hierarchy.items():
        children = [c for c in info['children'] if c in pos]
        if not children or parent not in pos:
            continue

        x_coords = [pos[c][0] for c in children]
        y_coords = [pos[c][1] for c in children]

        fig.add_shape(
            type="rect",
            x0=min(x_coords) - 0.3, y0=min(y_coords) - 0.2,
            x1=max(x_coords) + 0.3, y1=max(y_coords) + 0.2,
            line=dict(color=type_to_color.get('Sequential', '#999'), width=2),
            fillcolor="rgba(200,200,200,0.05)",
            layer="below"
        )

        fig.add_annotation(
            x=pos[parent][0], y=pos[parent][1] + 0.5,
            text=f"Sequential: {parent}",
            showarrow=False,
            font=dict(size=10, color=type_to_color.get('Sequential', '#999'))
        )

    # --- 保存或显示 ---
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        print(f"Model visualization saved to {output_path}")
    else:
        fig.show()

    return fig