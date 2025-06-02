import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.fx.node import Node

logger = logging.getLogger(__name__)

def get_placeholders(traced_model) -> List[str]:
    """
    Extract input placeholder node names from the traced graph.

    Args:
        traced_model: The traced PyTorch model.

    Returns:
        List[str]: Names of placeholder nodes.
    """
    return [node.name for node in traced_model.graph.nodes if node.op == "placeholder"]


def get_model_connections(traced_model) -> List[Dict[str, Union[str, None]]]:
    """
    Extract inter-module connections including skip connections and output edges.

    Args:
        traced_model: The traced PyTorch model.

    Returns:
        List[Dict[str, Union[str, None]]]: List of connection dictionaries.
    """
    connections = []
    inputs = []
    module_nodes: Dict[Any, Node] = {}

    # Build aggregated module-to-node mapping with constant merging
    constant_groups = defaultdict(list)

    # First pass: group all constant-like nodes by their base name
    for node in traced_model.graph.nodes:
        match = re.match(r'([a-zA-Z_]+)(\d+)(?:_(\d+))?', node.name)
        if match:
            base_name, main_id, _ = match.groups()
            base_key = f"{base_name}{main_id}"
            constant_groups[base_key].append(node)
            # logger.debug(f"Node {node.name} matched base key {base_key}")
        else:
            # Add non-constant nodes to module_nodes
            module_nodes[node.target] = node

    # Second pass: build module_nodes with merged info
    for base_key, nodes in constant_groups.items():
        base_node = next((n for n in nodes if '_' not in n.name), nodes[0])
        # Merge args and kwargs
        merged_args = []
        merged_kwargs = {}

        for node in nodes:
            if node != base_node:
                if node.args:
                    merged_args.extend(node.args)
                if node.kwargs:
                    merged_kwargs.update(node.kwargs)
        # Update base node to include merged info
        new_args = list(base_node.args) + merged_args
        new_kwargs = dict(base_node.kwargs)
        new_kwargs.update(merged_kwargs)
        base_node._update_args_kwargs(tuple(new_args), new_kwargs)

        module_nodes[base_node.target] = base_node

    # Third pass: redirect constant-args of other nodes to base node
    for node in module_nodes.values():
        # Rebuild args
        new_args = []
        for arg in node.args:
            try:
                if isinstance(arg, Node):
                    new_args += [arg]
                elif isinstance(arg, tuple):
                    for a in arg:
                        if not isinstance(a, Node):
                            raise
                        new_args += [a]
                else:
                    raise
            except Exception as e:
                logger.debug(f"Passing arg \"{arg}\" in node \"{node.name}\" with args {node.args}")
        node.args = tuple(new_args)
    for node in module_nodes.values():
        new_args = []
        for arg in node.args:
            if isinstance(arg, Node):
                match = re.match(r'([a-zA-Z_]+)(\d+)_(\d+)', arg.name)
                if match:
                    new_args.append(module_nodes[arg.target])
                else:
                    new_args.append(arg)
            else:
                raise
        node._update_args_kwargs(tuple(new_args), node.kwargs)
        for node in new_args:
            node._update_args_kwargs(node.args, node.kwargs)



    # Build user map
    user_map = build_user_map(module_nodes)

    # Preprocess inputs and tensor_constants
    input_counter = 0
    for node in module_nodes.values():
        if node.op == "placeholder":
            inputs.append({"name": node.target, "id": input_counter})
            input_counter += 1
        elif node.op == "get_attr":
            # redirect to inputs
            id = re.search(r'constant(\d+)', node.target)
            for input in inputs:
                if input["id"] == int(id.group(1)):
                    node.target = input["name"]
                    node.name = input["name"]
                    break
    # Extract connections
    for src_node in module_nodes.values():
        for user_node in user_map[src_node]:
            if user_node.op == "call_function":
                for next_node in user_map.get(user_node, []):
                    if next_node.op == "call_module":
                        connections.append({
                            "source": src_node.target,
                            "target": next_node.target,
                            "type": "skip"
                        })
                connections.append({
                    "source": src_node.target,
                    "target": user_node.target,
                    "type": "normal"
                })
            elif user_node.op == "call_method":
                for next_node in user_map.get(user_node, []):
                    if next_node.op == "call_module":
                        connections.append({
                            "source": str(src_node.target),
                            "target": str(next_node.target),
                            "type": "skip"
                        })
                connections.append({
                    "source": src_node.target,
                    "target": user_node.target,
                    "type": "normal"
                })
            elif user_node.op == "call_module":
                connections.append({
                    "source": src_node.target,
                    "target": user_node.target,
                    "type": "normal"
                })
            elif user_node.op == "output":
                connections.append({
                    "source": src_node.target,
                    "target": "output",
                    "type": "output"
                })

    return connections

def build_user_map(module_nodes: Dict[Any, Node]) -> Dict[Node, List[Node]]:
    """
    Build a mapping from each node to its consumer nodes.

    Args:
        module_nodes (Dict[Any, Node]): Mapping from module targets to Nodes.

    Returns:
        Dict[Node, List[Node]]: A dictionary where keys are nodes and values are lists of consumer nodes.
    """
    user_map = defaultdict(list)
    for src_node in module_nodes.values():
        new_users = []
        for user in src_node.users:
            if isinstance(user, Node):
                # Match sub-constant pattern like _tensor_constant0_1
                match = re.match(r'([a-zA-Z_]+)(\d+)_(\d+)', user.name)
                if match:
                    base_name = f"{match.group(1)}{match.group(2)}"
                    # Find the base constant node in module_nodes
                    base_key = None
                    for node in module_nodes.values():
                        if node.name == base_name:
                            base_key = node
                            break
                    if base_key is not None:
                        # Replace the sub-constant with the base constant node
                        new_users.append(base_key)
                else:
                    new_users.append(user)
            else:
                raise
        src_node.users = new_users

    for node in module_nodes.values():
        for user_node in node.users:
            user_map[node].append(user_node)
    return user_map

def _print_node_info(node: Node) -> None:
    print(f"Node {node.name}")
    print(f"\top: {node.op}")
    print(f"\ttarget: {node.target}")
    print(f"\tall_input_nodes: {node.all_input_nodes}")
    print(f"\targs: {node.args}")
    print(f"\tkwargs: {node.kwargs}")
    # print(f"\tmeta: {node.meta}")
    print(f"\tusers: {node.users}")
    print(f"\tnext: {node.next}")
    print(f"\tprev: {node.prev}")
    print(f"\ttype: {node.type}")