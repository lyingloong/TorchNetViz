import inspect
from typing import Dict, List, Optional, Tuple, Union, Any
from torch import nn, Tensor


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
