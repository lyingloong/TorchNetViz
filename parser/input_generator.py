import inspect
from typing import Dict, List, Optional, Tuple, Union, Any
from torch import nn, Tensor
import torch
from torch.jit import ScriptModule


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
    if input_shapes is None:
        raise ValueError(f"Input shapes must be provided, but {input_shapes} is valid.")

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