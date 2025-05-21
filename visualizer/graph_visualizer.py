import os
import networkx as nx
import plotly.graph_objects as go
from typing import Type, Dict, Any, Optional, List, Tuple, Union

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def visualize_structure(
    structure: list[dict[str, str]],
    connections: list[dict[str, Union[str, None]]],
    inputs: Optional[list[str]] = None,
    output_path: Optional[str] = None,
    showgrid: bool = False
) -> go.Figure:
    """
    Visualize model architecture using Plotly with support for skip connections (dashed edges).

    Args:
        structure (list of dict): List of layer information. Each item must contain 'name' and 'type'.
        connections (list of dict): List of connection info. Each item contains 'source', 'target', and optional 'type'
                                   ('skip', 'input', 'output').
        inputs (Optional[list[str]], optional): Input node names. Defaults to None.
        output_path (Optional[str], optional): Path to save the visualization HTML file. If not provided, show in browser.
        showgrid (bool, optional): Whether to display grid background. Defaults to False.

    Returns:
        go.Figure: A Plotly figure object representing the model structure.
    """
    # Build directed graph
    G = nx.DiGraph()

    # Add nodes from structure
    for layer_info in structure:
        G.add_node(layer_info['name'], layer_type=layer_info.get('type', 'Unknown'))

    # Categorize edge types
    normal_edges = []
    skip_edges = []
    output_edges = []

    for edge in connections:
        src, tgt = edge['source'], edge['target']
        edge_type = edge.get('type', None)
        if edge_type == 'skip':
            skip_edges.append((src, tgt))
        elif edge_type == 'output':
            output_edges.append((src, tgt))
        else:
            normal_edges.append((src, tgt))

    # Add all edges to graph
    for u, v in normal_edges + skip_edges + output_edges:
        G.add_edge(u, v)

    # Handle input placeholders
    if inputs:
        for inp in inputs:
            G.add_node(inp, layer_type='Input')
            if len(G.nodes()) > 0:
                first_node = next(iter(G.nodes()))
                G.add_edge(inp, first_node)

    # Layout computation
    pos = nx.kamada_kawai_layout(G)
    print(f"Node positions computed: {pos}")

    # Edge drawing helper
    def _create_edge_trace(edge_list, dash_style):
        x = []
        y = []
        for u, v in edge_list:
            x += [pos[u][0], pos[v][0], None]
            y += [pos[u][1], pos[v][1], None]

        return go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(width=2, dash=dash_style),
            hoverinfo='none'
        )

    # Create different types of edges
    edge_traces = [
        _create_edge_trace(normal_edges, 'solid'),
        _create_edge_trace(skip_edges, 'dash'),
        _create_edge_trace(output_edges, 'dot')
    ]

    # Node positioning and labels
    node_x, node_y, node_text = [], [], []
    for node, data in G.nodes(data=True):
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
        label = f"{node} ({data.get('layer_type', 'Unknown')})"
        node_text.append(label)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        hoverinfo='text',
        marker=dict(symbol='circle', size=20)
    )

    # Assemble final figure
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=showgrid, zeroline=False, visible=False),
        yaxis=dict(showgrid=showgrid, zeroline=False, visible=False),
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=20)
    )

    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        print(f"Model visualization saved to {output_path}")
    else:
        fig.show()

    return fig
