import logging
import itertools
import math
import os
from collections import defaultdict

import networkx as nx
import plotly.graph_objects as go
from typing import Optional, Union, Tuple, Dict, List, Any
from networkx import DiGraph

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logger = logging.getLogger(__name__)

def visualize_structure_v2(
    structure: list[dict[str, str]],
    connections: list[dict[str, Union[str, None]]],
    inputs: Optional[list[str]] = None,
    output_path: Optional[str] = None,
    showgrid: bool = False,
    top_scale: float = 8.0,
    top_k: float = 1.2,
    iterations: int = 200,
    container_weight: float = 5.0
) -> go.Figure:
    # Step 1: 构建节点图与层次关系
    def _create_nodes(structure: list[dict[str, str]]) -> DiGraph:
        G = nx.DiGraph()
        hierarchy = {}
        for layer in structure:
            name = layer['name']
            ltype = layer.get('type', 'Unknown')

            node_attrs = {
                'layer_type': ltype
            }
            for key, value in layer.items():
                if key not in ('name', 'type'):
                    node_attrs[key] = value

            G.add_node(name, **node_attrs)
            if 'submodules' in layer:
                hierarchy[name] = {
                    'children': layer['submodules'],
                    'type': ltype
                }
        return G, hierarchy

    def _add_input_nodes(G: nx.DiGraph, inputs: Optional[list[str]]) -> nx.DiGraph:
        if inputs:
            for inp in inputs:
                G.add_node(inp, layer_type='Input')
        return G

    G, hierarchy = _create_nodes(structure)
    G = _add_input_nodes(G, inputs)

    # Step 2: 加边
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

    # Step 3: 处理强连通部件，删除孤立节点
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

    # 获取孤立节点列表
    isolated_nodes = [n for n in nx.isolates(G)
                      # if G.nodes[n].get('layer_type') != 'Input'
                      ]
    logger.debug(f"Isolated nodes: {isolated_nodes}")
    # # 递归删除孤立容器节点及其子节点
    # def remove_isolated_container(container):
    #     if container in hierarchy:
    #         # 先删除所有子节点
    #         for child in hierarchy[container]['children']:
    #             if child in G:
    #                 G.remove_node(child)
    #         # 再删除容器本身
    #         G.remove_node(container)
    #         # 更新层次结构
    #         del hierarchy[container]
    #
    # 遍历所有孤立节点
    for node in list(isolated_nodes):  # 转换为list避免迭代时修改图结构
        if node not in G:
            continue

        # 处理普通节点
        if node not in hierarchy:
            G.remove_node(node)
            # 从层次结构中删除作为子节点的情况
            for parent in hierarchy.values():
                if node in parent['children']:
                    parent['children'].remove(node)

    # # 清理空容器
    # empty_containers = [k for k, v in hierarchy.items() if not v['children']]
    # for ec in empty_containers:
    #     remove_isolated_container(ec)

    # Step 4: 布局
    def _hierarchical_layout(G: nx.DiGraph, hierarchy: dict,
                             scale: float = 8.0, k: float = 1.2, iterations: int = 200, node_weight: float=1.0
                             ) -> Tuple[dict, dict]:
        """分层嵌套布局算法（改进版）"""
        from networkx import spring_layout
        import random

        # ---- 阶段1：构建包含容器和顶层节点的简化图 ----
        G_top = nx.DiGraph()
        child_to_parent = defaultdict(str)  # 子节点到父容器的映射
        for parent, info in hierarchy.items():
            for child in info['children']:
                child_to_parent[child] = parent

        # 添加顶层节点和容器节点，容器节点设置更大权重
        node_weights = {}
        for node in G.nodes:
            if node in hierarchy:
                G_top.add_node(node)
                node_weights[node] = node_weight  # 容器节点权重更大
            elif node not in child_to_parent:
                G_top.add_node(node)
                node_weights[node] = 1.0  # 普通顶层节点

        # 添加简化后的边（将子节点的边连接到父容器）
        for u, v in G.edges():
            u_parent = child_to_parent.get(u, u)
            v_parent = child_to_parent.get(v, v)
            if u_parent != v_parent:
                G_top.add_edge(u_parent, v_parent)

        # ---- 阶段2：对简化图进行力导向布局 ----
        pos_top = spring_layout(
            G_top,
            weight="weight",
            scale=scale,  # 增大整体布局范围
            k=k,  # 增强节点间排斥力
            iterations=iterations
        )

        # ---- 阶段3：容器内部子节点动态布局 ----
        pos_all = {}
        bbox_dict = {}

        # 处理非容器节点时检查与容器的距离
        for node in pos_top:
            if node not in hierarchy:
                # 防止非容器节点过于靠近容器中心
                min_dist = 3.0  # 最小安全距离
                adjusted = False
                for container in hierarchy:
                    dx = pos_top[node][0] - pos_top[container][0]
                    dy = pos_top[node][1] - pos_top[container][1]
                    dist = math.sqrt(dx ** 2 + dy ** 2)
                    if dist < min_dist:
                        # 将节点推离容器区域
                        push_dir = math.atan2(dy, dx)
                        pos_all[node] = (
                            pos_top[container][0] + min_dist * math.cos(push_dir),
                            pos_top[container][1] + min_dist * math.sin(push_dir)
                        )
                        adjusted = True
                        break
                if not adjusted:
                    pos_all[node] = pos_top[node]

        # 对每个容器进行内部布局
        for container, info in hierarchy.items():
            children = info['children']
            if not children:
                continue

            # 构建包含父容器的子图（用于布局）
            subG = nx.DiGraph()
            subG.add_node(container)  # 关键修复：添加父容器节点
            subG.add_nodes_from(children)
            # 添加容器内部连接（仅子节点间连接）
            for u, v in G.edges():
                if u in children and v in children:
                    subG.add_edge(u, v)

            # 动态布局参数
            scale_factor = 1.5 + 0.1 * len(children)  # 动态缩放
            k_factor = 0.5 / (1 + math.log(len(children) + 1))  # 动态调整作用力

            # 初始化位置（父容器位置固定）
            initial_pos = {container: pos_top[container]}
            for child in children:
                initial_pos[child] = (
                    pos_top[container][0] + (random.random() - 0.5) * 0.1,
                    pos_top[container][1] + (random.random() - 0.5) * 0.1
                )

                # 执行力导向布局（增加边缘约束）
                child_pos = nx.spring_layout(
                    subG,
                    pos=initial_pos,
                    fixed=[container],
                    scale=scale_factor,
                    k=k_factor,
                    iterations=iterations,  # 增加迭代次数
                    center=pos_top[container],  # 新增中心约束
                    seed=42  # 固定随机种子保证稳定性
                )

                # 提取坐标时限制子节点范围
                max_radius = 1.5 * scale_factor  # 最大允许半径
                center_x, center_y = pos_top[container]
                for child in children:
                    x, y = child_pos[child]
                    dx = x - center_x
                    dy = y - center_y
                    dist = math.sqrt(dx ** 2 + dy ** 2)
                    if dist > max_radius:
                        # 将超出范围的子节点拉回
                        ratio = max_radius / dist
                        pos_all[child] = (
                            center_x + dx * ratio,
                            center_y + dy * ratio
                        )
                    else:
                        pos_all[child] = (x, y)

                # 计算包围盒时仅使用子节点坐标
                child_coords = [pos_all[c] for c in children]
                xs = [p[0] for p in child_coords]
                ys = [p[1] for p in child_coords]
                margin = 0.5 + 0.08 * len(children)
                bbox_dict[container] = (
                    min(xs) - margin,
                    min(ys) - margin,
                    max(xs) + margin,
                    max(ys) + margin
                )

        return pos_all, bbox_dict

    pos, bbox_dict = _hierarchical_layout(G, hierarchy, scale=top_scale, k=top_k, iterations=iterations, node_weight=container_weight)

    # Step 5: 节点和边颜色定义
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

    # Step 6: 绘制边
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

    # Step 7: 绘制节点
    node_x, node_y, node_text, node_color, node_customdata = [], [], [], [], []
    for node, data in G.nodes(data=True):
        if node not in pos:
            logger.warning(f"{node} not found in pos. Skipping.")
            continue
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
        if data.get("layer_type") == "Component":
            label = f"{node} (Composite)"
            details = ", ".join(sorted(data.get("details", [])))
        else:
            label = f"{node} ({data.get('layer_type', 'Unknown')})"
            # 自动提取所有非系统属性
            excluded_keys = {"name", "type", "layer_type", "source"}
            details = [
                f"{k}={v}" for k, v in sorted(data.items()) if k not in excluded_keys
            ]
            details = ", ".join(details)
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

    # Step 8: 绘制容器框与注释
    fig = go.Figure(data=edge_traces + [node_trace])

    for parent, (x0, y0, x1, y1) in bbox_dict.items():
        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color=type_to_color.get('Sequential', "#AAAAAA"), width=2, dash="dot"),
            fillcolor="rgba(200,200,200,0.05)",
            layer="below"
        )
        fig.add_annotation(
            x=(x0 + x1) / 2, y=y1 + 0.3,
            text=f"{parent} (Container)",
            showarrow=False,
            font=dict(size=10, color=type_to_color.get('Sequential', "#AAAAAA"))
        )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=showgrid, zeroline=False, visible=False),
        yaxis=dict(showgrid=showgrid, zeroline=False, visible=False),
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=20)
    )

    # Step 9: 保存或显示
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        print(f"Model visualization saved to {output_path}")
    else:
        fig.show()

    return fig


def visualize_structure_v1(
        structure: list[dict[str, str]],
        connections: list[dict[str, Union[str, None]]],
        inputs: Optional[list[str]] = None,
        output_path: Optional[str] = None,
        showgrid: bool = False
) -> go.Figure:
    def _create_nodes(structure: list[dict[str, str]]) -> DiGraph:
        G = nx.DiGraph()
        for layer in structure:
            name = layer['name']
            ltype = layer.get('type', 'Unknown')

            node_attrs = {
                'layer_type': ltype
            }
            for key, value in layer.items():
                if key not in ('name', 'type'):
                    node_attrs[key] = value

            G.add_node(name, **node_attrs)
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

    G = _add_input_nodes(G, inputs)

    # 删除孤立节点
    isolated_nodes = [
        n for n in nx.isolates(G)
        # if G.nodes[n].get('layer_type') != 'Input'
    ]

    # 删除孤立节点
    for node in isolated_nodes:
        G.remove_node(node)
        logger.info(f"Removed isolated node: {node}")

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

    pos = topological_layered_layout(G)
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
    logger.debug(
        f"Edges classified: normal {edge_dict['normal']} skip {edge_dict['skip']} output {edge_dict['output']}")

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
            excluded_keys = {"name", "type", "layer_type", "source"}

            # 自动提取所有非系统属性作为 details
            details = [f"{key}={value}" for key, value in data.items() if key not in excluded_keys]
            details = ", ".join(sorted(details))
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

    # --- 保存或显示 ---
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        print(f"Model visualization saved to {output_path}")
    else:
        fig.show()

    return fig