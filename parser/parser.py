import inspect
import itertools
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any
from torch import nn, Tensor
import torch
from torch.fx import symbolic_trace
from torch.fx.node import Node
from torch.jit import ScriptModule
import warnings
import json


def parse_model(model: Union[nn.Module, ScriptModule], input_shapes: Optional[Dict[str, Tuple]] = None) -> Dict[str, Any]:
    """
    Unified entry point for model structure parsing.

    Args:
        model (Union[nn.Module, ScriptModule]): The PyTorch model to be analyzed.
        input_shapes (Optional[Dict[str, Tuple]], optional): Expected input shapes. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'structure': List of module info dicts
            - 'connections': List of connection info dicts
            - 'inputs': List of input placeholder names
    """
    if isinstance(model, nn.Module):
        if input_shapes is None:
            input_shapes = get_input_shapes(model)
        example_inputs = create_example_inputs(model, input_shapes)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traced_model = symbolic_trace(model, concrete_args=example_inputs)

        structure = parse_model_structure(model)
        connections = get_model_connections(traced_model)
        placeholders = get_placeholders(traced_model)

        return {
            "structure": structure,
            "connections": connections,
            "inputs": placeholders
        }

    elif isinstance(model, ScriptModule):
        # Placeholder for future ScriptModule handling logic
        raise NotImplementedError("ScriptModule support not yet implemented in this version.")
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")


def get_placeholders(traced_model) -> List[str]:
    """
    Extract input placeholder node names from the traced graph.

    Args:
        traced_model: The traced PyTorch model.

    Returns:
        List[str]: Names of placeholder nodes.
    """
    return [node.name for node in traced_model.graph.nodes if node.op == "placeholder"]


def parse_model_structure(model: nn.Module) -> List[Dict[str, str]]:
    """
    Parse the hierarchical structure of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        List[Dict[str, str]]: List of module information dictionaries.
    """
    structure = []
    for name, module in model.named_modules():
        if name:
            info = extract_module_info(module)
            info["name"] = name
            if info["type"] == 'Sequential':
                info["submodules"] = []
                for idx, sub_module in enumerate(module):
                    if isinstance(sub_module, nn.Module):
                        sub_name = f"{name}.{idx}"
                        info["submodules"].append(sub_name)
            structure.append(info)
    return structure


def extract_module_info(module: nn.Module) -> Dict[str, Any]:
    """
    Extracts key structural information from a PyTorch module in a generic way.

    Args:
        module (nn.Module): The module to analyze.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'type': The class name of the module (e.g., 'Conv2d', 'Linear').
            - 'params': A dictionary of constructor parameters (e.g., in_channels, out_channels).
            - 'has_weight': Boolean indicating whether the module has learnable weights.
            - 'has_bias': Boolean indicating whether the module has a bias parameter.
            - 'source': Source code of the module's class (for visualization or documentation).
    """
    info = {
        "type": type(module).__name__,
        "params": {},
        "has_weight": False,
        "has_bias": False,
        "source": "",
    }

    # Extract constructor parameters (from __dict__)
    if hasattr(module, "__dict__"):
        for key, value in module.__dict__.items():
            if not key.startswith("_") and not isinstance(value, (nn.Module, list, dict, tuple)):
                info["params"][key] = str(value)

    # Check for weight and bias parameters
    if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        info["has_weight"] = True
    if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
        info["has_bias"] = True

    # Try to retrieve source code for display/documentation purposes
    try:
        info["source"] = inspect.getsource(module.__class__)
    except Exception:
        info["source"] = ""

    return info


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
            if isinstance(arg, Node):
                new_args += [arg]
            else:
                for a in arg:
                    if not isinstance(a, Node):
                        raise
                    new_args += [a]
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


def create_example_inputs(
    model: Union[nn.Module, ScriptModule],
    input_shapes: Dict[str, Tuple]
) -> Dict[str, Tensor]:
    """
    Generate dummy input tensors based on model signature or TorchScript schema.

    Args:
        model (Union[nn.Module, ScriptModule]): Target model.
        input_shapes (Dict[str, Tuple]): Shape dict like {'x': (1, 3, 224, 224)}.

    Returns:
        Dict[str, Tensor]: Dictionary of example inputs.
    """
    example_inputs = {}

    if isinstance(model, nn.Module):
        sig = inspect.signature(model.forward)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.default != inspect.Parameter.empty:
                continue
            if name not in input_shapes:
                raise ValueError(f"Missing shape for required input parameter '{name}'")
            example_inputs[name] = torch.randn(input_shapes[name])

        if not example_inputs:
            raise ValueError("Model has no required input parameters.")

    elif isinstance(model, ScriptModule):
        try:
            # Try schema-based extraction
            sig = model.forward.schema
            for arg in sig.arguments:
                if arg.name == "self":
                    continue
                if arg.name not in input_shapes:
                    raise ValueError(f"Missing shape for ScriptModule input '{arg.name}'")
                example_inputs[arg.name] = torch.randn(input_shapes[arg.name])
        except Exception as e:
            print(f"Schema extraction failed: {e}")
            # Fallback to graph.inputs()
            for node in model.graph.inputs():
                name = node.debugName()
                if name == "self":
                    continue
                if name not in input_shapes:
                    raise ValueError(f"Missing shape for ScriptModule input '{name}'")
                example_inputs[name] = torch.randn(input_shapes[name])

    else:
        raise TypeError("Unsupported model type provided.")

    return example_inputs


def get_input_shapes(model: Union[nn.Module, ScriptModule], max_args: int = 10) -> Dict[str, Tuple[int, ...]]:
    """
    Automatically infer tensor input shapes by probing forward() method.

    Args:
        model (Union[nn.Module, ScriptModule]): Model to analyze.
        max_args (int, optional): Max number of tensor args to try. Defaults to 6.

    Raises:
        RuntimeError: If inference fails after all attempts.

    Returns:
        Dict[str, Tuple[int, ...]]: Mapping from input name to shape.
    """
    input_shapes: Dict[str, Tuple[int, ...]] = {}

    if isinstance(model, ScriptModule):
        try:
            # Attempt schema-based input extraction
            sig = model.forward.schema
            for arg in sig.arguments:
                if arg.name == "self":
                    continue
                if "Tensor" in str(arg.type):
                    input_shapes[arg.name] = tuple(int(d) for d in arg.type.sizes())
            if input_shapes:
                return input_shapes
        except Exception as e:
            print(f"Schema extraction failed: {e}")

        # Fallback to graph.inputs()
        for node in model.graph.inputs():
            name = node.debugName()
            if name == "self":
                continue
            type_info = node.type()
            if isinstance(type_info, torch._C.TensorType):
                input_shapes[name] = tuple(int(d) for d in type_info.sizes())

        return input_shapes

    elif isinstance(model, nn.Module):
        sig = inspect.signature(model.forward)
        tensor_args = []
        non_tensor_defaults = {}

        for name, param in sig.parameters.items():
            if name in ("self", "args", "kwargs"):
                continue

            annotation = param.annotation
            is_tensor = annotation == Tensor or "Tensor" in str(annotation)

            if is_tensor or annotation == inspect.Parameter.empty:
                tensor_args.append(name)
            elif param.default != inspect.Parameter.empty:
                non_tensor_defaults[name] = param.default
            else:
                # Assign safe defaults
                if annotation == bool:
                    non_tensor_defaults[name] = False
                elif annotation == int:
                    non_tensor_defaults[name] = 1
                elif annotation == float:
                    non_tensor_defaults[name] = 0.0
                elif annotation == str:
                    non_tensor_defaults[name] = "default"
                elif annotation in (tuple, list):
                    non_tensor_defaults[name] = []
                else:
                    non_tensor_defaults[name] = None

        if len(tensor_args) > max_args:
            raise RuntimeError(f"Too many tensor inputs to infer (> {max_args})")

        # Probe input shapes
        ndim_choices = [1, 2, 3, 4]
        dim_vals = [1, 3, 8, 16, 32, 64]

        for ndim in ndim_choices:
            for dims in itertools.product(dim_vals, repeat=ndim):
                try:
                    input_kwargs = {
                        name: torch.randn((1,) + dims) for name in tensor_args
                    }
                    input_kwargs.update(non_tensor_defaults)

                    model.eval()
                    with torch.no_grad():
                        model(**input_kwargs)

                    return {
                        name: tuple(t.shape) for name, t in input_kwargs.items() if isinstance(t, Tensor)
                    }

                except Exception:
                    continue

        raise RuntimeError("Failed to infer input shape by all attempts.")
    else:
        raise TypeError("Unsupported model type provided.")


def save_model_structure(result: Dict[str, Any], filepath: str = "output/model_structure.json") -> None:
    """
    Save parsed model structure to JSON file.

    Args:
        result (Dict[str, Any]): Output from [parse_model](file://D:\Code\TorchNetViz\parser\parser.py#L13-L42).
        filepath (str, optional): Output path. Defaults to "output/model_structure.json".
    """
    for item in result.get("structure", []) + result.get("connections", []):
        if "source" in item:
            item["source"] = str(item["source"])[:100] + "..." if len(str(item["source"])) > 100 else str(item["source"])
        if "target" in item:
            item["target"] = str(item["target"])

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"Model structure saved to {filepath}")
