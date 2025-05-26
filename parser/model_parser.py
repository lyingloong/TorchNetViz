import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from torch import nn
from torch.fx import symbolic_trace
from torch.jit import ScriptModule
import warnings
import json
from .shape_inference import get_input_shapes
from .input_generator import create_example_inputs
from .module_inspector import parse_model_structure
from .graph_extractor import get_model_connections, get_placeholders

# 配置日志记录器
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_model(model: Union[nn.Module, ScriptModule], input_shapes: Optional[Dict[str, Tuple]] = None, dim_sizes: Optional[List[int]] = None,) -> Dict[str, Any]:
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
            input_shapes = get_input_shapes(model, dim_sizes=dim_sizes)
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