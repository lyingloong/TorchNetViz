import argparse
import ast
import base64
import inspect
import os
import json
from typing import Optional, Dict, Tuple, Union, List, Any

from parser.model_parser import parse_model, save_model_structure
from visualizer.graph_visualizer import visualize_structure_v1, visualize_structure_v2
from models.custom_layers import *
from parser.utils import auto_instantiate

def parse_input_shapes(input_str: Optional[str]) -> Optional[Dict[str, Tuple[int, ...]]]:
    """
    Parse input shapes from command line input.

    Supports:
        - JSON string: '{"x": [1, 3, 224, 224]}'
        - Base64 encoded JSON (recommended for CLI)

    Returns:
        Dict[str, Tuple]: Mapped input names to shape tuples.
    """
    if not input_str:
        return None

    try:
        shapes = json.loads(input_str)
    except json.JSONDecodeError:
        try:
            decoded = base64.b64decode(input_str).decode("utf-8")
            shapes = json.loads(decoded)
        except Exception as decode_error:
            raise ValueError(f"Invalid input_shapes format. Not valid JSON or Base64: {decode_error}")

    return {k: tuple(v) for k, v in shapes.items()}


def parse_args_argument(arg_list_str: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """
    Parse --args list where each element is a base64-encoded config string.

    Args:
        arg_list_str (Optional[str]): Command-line argument list string.

    Returns:
        Optional[List[Dict[str, Any]]]: Processed arguments with parsed values and types.
    """
    if not arg_list_str:
        return None

    try:
        base64_list = ast.literal_eval(arg_list_str)
        if not isinstance(base64_list, list):
            raise ValueError("Argument must be a list of base64 strings.")
    except Exception as e:
        raise ValueError(f"Failed to parse argument list: {e}")

    results = []

    for encoded in base64_list:
        # 尝试解码 Base64
        decoded = base64.b64decode(encoded).decode("utf-8")
        # 尝试解析为 JSON
        item = json.loads(decoded)
        arg_value = item["value"]
        arg_type = item["type"].lower()

        if arg_type in ("list", "tuple"):
            try:
                parsed = json.loads(arg_value.replace('(', '[').replace(')', ']'))
                if arg_type == "tuple":
                    parsed = tuple(parsed) if isinstance(parsed, list) else parsed
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse list/tuple argument: {arg_value}")
        elif arg_type == "int":
            try:
                parsed = int(arg_value)
            except ValueError:
                raise ValueError(f"Cannot convert to integer: {arg_value}")
        elif arg_type == "float":
            try:
                parsed = float(arg_value)
            except ValueError:
                raise ValueError(f"Cannot convert to float: {arg_value}")
        elif arg_type == "configs":
            # Create dynamic Config class
            class Configs:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            parsed = Configs(**arg_value)
        elif arg_type == "class":
            # Dynamically import class
            module_name, class_name = "models.builtin", arg_value
            try:
                module = __import__(module_name, fromlist=[class_name])
                parsed = getattr(module, class_name)
                if not inspect.isclass(parsed) or not issubclass(parsed, nn.Module):
                    raise TypeError(f"{class_name} is not a subclass of torch.nn.Module")
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Failed to import model class {class_name} from {module_name}: {e}")
        else:
            raise ValueError(f"Unsupported argument type: {arg_type}")

        results.append({"parsed": parsed, "type": arg_type})

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TorchNetViz")

    parser.add_argument("--model", type=str, default="Linear",
                        help="Model name (default: Linear)")
    parser.add_argument("--pt", type=str, default=None,
                        help="(optional) Path to load PyTorch .pt model")
    parser.add_argument("--args", type=str, default=None,
                        help="(optional) Model constructor args, e.g., '[1, 3, 224, 224]'")
    parser.add_argument("--input_shapes", type=str, default=None,
                        help='(optional) Manually specify input shapes, e.g., {"x": [1, 3, 224, 224]}')
    parser.add_argument("--model_file", type=str, default=None,
                        help="Custom model filename (without .py), defaults to built-in models")
    parser.add_argument("--input_dim_sizes", type=str, default=None,
                        help="Reference dimension sizes for input shape inference")
    parser.add_argument("--ndims", type=int, default=3,
                        help="Maximum number of dimensions per input (including batch dim)")
    args = parser.parse_args()

    if args.pt:
        raise NotImplementedError("Loading PyTorch .pt models is not yet supported.")
    elif args.model_file:
        # Import user-defined model class
        module_name = f"models.{args.model_file}"
        try:
            module = __import__(module_name, fromlist=[args.model])
            model_class = getattr(module, args.model)
            if not issubclass(model_class, nn.Module):
                raise TypeError(f"{args.model} is not a subclass of torch.nn.Module")
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import {args.model} from {module_name}: {e}")
    else:
        # Auto-discover all nn.Module subclasses in current scope
        MODEL_MAP = {
            name.lower(): cls
            for name, cls in globals().items()
            if inspect.isclass(cls) and issubclass(cls, nn.Module)
        }

        model_name = args.model.lower()
        if model_name not in MODEL_MAP:
            raise ValueError(f"Unsupported model: {args.model}")
        model_class = MODEL_MAP[model_name]

    # Instantiate model
    if args.args:
        args_list = parse_args_argument(args.args)
        model = model_class(*[arg["parsed"] for arg in args_list])
    else:
        model = auto_instantiate(model_class)

    if args.input_dim_sizes:
        input_dim_sizes = [int(x) for x in args.input_dim_sizes.split(",")]
    else:
        input_dim_sizes = None

    if args.input_shapes:
        input_shapes = parse_input_shapes(args.input_shapes)
    else:
        input_shapes = None

    os.makedirs(f"./output/{args.model}", exist_ok=True)

    result = parse_model(model, input_shapes, dim_sizes=input_dim_sizes, ndims=args.ndims)
    save_model_structure(result, f"output/{args.model}/{args.model}.json")
    visualize_structure_v1(result["structure"], result["connections"],
                        inputs=result["inputs"], output_path=f"output/{args.model}/{args.model}_v1.html")
    for id, t_scale, t_k, c_weight in [(0, 6.0, 1.5, 5.0), (1, 10.0, 2.0, 8.0), (2, 15.0, 2.5, 10.0)]:
        visualize_structure_v2(result["structure"], result["connections"],
                            inputs=result["inputs"], output_path=f"output/{args.model}/{args.model}_v2_{id}.html",
                            top_scale=t_scale, top_k=t_k, iterations=300, container_weight=c_weight)
