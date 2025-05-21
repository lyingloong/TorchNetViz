import torch
from torch import nn, jit
from typing import Type, Dict, Any, Optional, List, Tuple, Union
import inspect
import itertools

def to_torchscript(model: nn.Module, save_path: Optional[str] = None) -> jit.ScriptModule:
    """
    Converts a PyTorch model into TorchScript format for serialization and deployment.

    Args:
        model (nn.Module): The PyTorch model to be scripted.
        save_path (Optional[str], optional): File path to save the scripted model.
                                          If not provided, the model won't be saved to disk.

    Returns:
        jit.ScriptModule: A TorchScript-compiled version of the input model.

    Raises:
        RuntimeError: If scripting or saving the model fails.
    """
    try:
        # Convert the model using TorchScript
        scripted_model = torch.jit.script(model)

        # Save to disk if a save path is provided
        if save_path:
            scripted_model.save(save_path)
            print(f"Model successfully saved to {save_path}")

        return scripted_model

    except Exception as e:
        raise RuntimeError(f"Failed to script model: {e}")


def auto_instantiate(module_class: Type[nn.Module]) -> nn.Module:
    """
    Automatically instantiate a PyTorch module by inferring constructor arguments.

    This function attempts two strategies:
    1. Uses type annotations to infer default values for required parameters.
    2. Falls back to brute-force search over common default value combinations.

    Args:
        module_class (Type[nn.Module]): The class of the module to instantiate.

    Returns:
        nn.Module: An instantiated instance of the given module class.

    Raises:
        RuntimeError: If no valid argument combination can be found.
    """
    sig = inspect.signature(module_class.__init__)
    params = sig.parameters
    args_dict: Dict[str, Any] = {}

    # === Phase 1: Try type annotation-based construction ===
    try:
        for name, param in params.items():
            if name in ("self", "args", "kwargs"):
                continue

            if param.default == inspect.Parameter.empty:
                # Handle common types with safe defaults
                if param.annotation == int:
                    args_dict[name] = 16
                elif param.annotation == float:
                    args_dict[name] = 0.5
                elif param.annotation in (tuple, list):
                    args_dict[name] = (3, 3)
                elif param.annotation == str:
                    args_dict[name] = "default"
                elif param.annotation == bool:
                    args_dict[name] = False
                elif param.annotation == Optional[int]:
                    args_dict[name] = None
                else:
                    raise ValueError(f"Unsupported or missing annotation: '{name}' -> {param.annotation}")
            else:
                args_dict[name] = param.default

        return module_class(**args_dict)

    except Exception as e:
        print(f"[!] Failed type-based instantiation: {e}. Falling back to brute-force parameter search...")

    # === Phase 2: Brute-force search over common parameter combinations ===
    param_names = []
    candidate_lists: List[List[Any]] = []

    for name, param in params.items():
        if name in ("self", "args", "kwargs"):
            continue
        if param.default != inspect.Parameter.empty:
            continue  # Skip parameters with default values

        param_names.append(name)

        def generate_candidates(param_name: str) -> List[Any]:
            lname = param_name.lower()
            if any(kw in lname for kw in ["channel", "feature", "hidden", "dim"]):
                return [8, 16, 32]
            elif any(kw in lname for kw in ["dropout", "rate"]):
                return [0.1, 0.5]
            elif any(kw in lname for kw in ["kernel", "size"]):
                return [(3, 3), 3, (1, 1)]
            elif "stride" in lname:
                return [1, 2]
            elif "padding" in lname:
                return [0, 1]
            elif "bias" in lname:
                return [True, False]
            elif "num" in lname:
                return [1, 2]
            elif "name" in lname:
                return ["default"]
            elif "path" in lname:
                return ["./dummy"]
            else:
                return [1, 16, 32, None]

        candidate_lists.append(generate_candidates(name))

    for combo in itertools.product(*candidate_lists):
        args = dict(zip(param_names, combo))
        try:
            model = module_class(**args)
            print(f"[*] Successfully instantiated {module_class.__name__} with args: {args}")
            return model
        except Exception:
            continue

    raise RuntimeError(f"Failed to instantiate {module_class.__name__}. Please provide explicit arguments.")
