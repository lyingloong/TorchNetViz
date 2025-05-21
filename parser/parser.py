import inspect
import itertools
import os
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
            module_type = type(module).__name__
            source_code = inspect.getsource(module.__class__)
            structure.append({
                "name": name,
                "type": module_type,
                "source": source_code
            })
    return structure


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
    user_map: Dict[Node, List[Node]] = {}

    # Build module-to-node mapping
    for node in traced_model.graph.nodes:
        module_nodes[node.target] = node
        user_map[node] = []

    # Build user map
    for node in traced_model.graph.nodes:
        for arg in node.args:
            if isinstance(arg, Node):
                user_map[arg].append(node)
            else:
                for a in arg:
                    if isinstance(a, Node):
                        user_map[a].append(node)

    # Extract connections
    for src_node in module_nodes.values():
        if src_node.op == "placeholder":
            inputs.append(src_node.target)
            continue
        if src_node.op == "get_attr":
            src_node.target = None

        for user_node in user_map[src_node]:
            if user_node.op == "call_function":
                for next_node in user_map.get(user_node, []):
                    if next_node.op == "call_module":
                        connections.append({
                            "source": str(src_node.target),
                            "target": str(next_node.target),
                            "type": "skip"
                        })
            elif user_node.op == "call_module":
                connections.append({
                    "source": str(src_node.target),
                    "target": str(user_node.target),
                    "type": "normal"
                })
            elif user_node.op == "output":
                connections.append({
                    "source": str(src_node.target),
                    "target": "output",
                    "type": "output"
                })

    # Patch missing sources with input names
    for input_name in inputs:
        for conn in connections:
            if conn["source"] is None:
                conn["source"] = str(input_name)
                conn["type"] = "input"
                break

    return connections


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


def get_input_shapes(model: Union[nn.Module, ScriptModule], max_args: int = 6) -> Dict[str, Tuple[int, ...]]:
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
