import torch
import itertools
import inspect
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union
from torch import nn, Tensor
from torch.jit import ScriptModule
from itertools import product
import logging

logger = logging.getLogger(__name__)

def get_input_shapes(
    model: Union[nn.Module, ScriptModule],
    max_args: int = 10,
    dim_sizes: Optional[List[int]] = None,
) -> Dict[str, Tuple[int, ...]]:
    input_shapes: Dict[str, Tuple[int, ...]] = {}

    if isinstance(model, ScriptModule):
        try:
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

        # ---------- 动态剪枝部分 ----------
        failed_combos = set()
        min_required_dims = 1

        def generate_candidate_shapes(ndims: int = 3) -> List[Tuple[int, ...]]:
            sizes = dim_sizes or [1, 2, 3, 8, 16, 32, 64, 96, 128, 256, 512]
            shapes = set()
            for ndim in range(min_required_dims, ndims + 1):
                for dims in itertools.product(sizes, repeat=ndim):
                    if np.prod(dims) <= 512 * 512:
                        shapes.add(dims)
            return sorted(shapes, key=lambda x: (len(x), x))

        raw_shapes = generate_candidate_shapes(ndims=3)[:500]
        batch_size = 1
        candidate_shapes = [[(batch_size,) + shape for shape in raw_shapes] for _ in tensor_args]

        logger.info(f"Total input shape combinations to try: {len(list(product(*candidate_shapes)))}")

        def try_shape_combo(shape_combo):
            nonlocal min_required_dims
            if shape_combo in failed_combos:
                return None
            try:
                input_kwargs = {
                    name: torch.randn(shape).float()
                    for name, shape in zip(tensor_args, shape_combo)
                }
                input_kwargs.update(non_tensor_defaults)
                shapes = {name: tuple(tensor.shape) for name, tensor in input_kwargs.items()}

                model.eval()
                with torch.no_grad():
                    model(**input_kwargs)

                return shapes
            except Exception as e:
                msg = str(e)
                logger.debug(f"Failed combo {shape_combo}: {e}")
                failed_combos.add(shape_combo)

                # ---------- 错误感知逻辑 ----------
                if "expected 3, got 2" in msg:
                    min_required_dims = max(min_required_dims, 3)
                elif "expected 4, got" in msg:
                    min_required_dims = max(min_required_dims, 4)

                return None

        # ---------- 多线程尝试 ----------
        shape_combos = list(product(*candidate_shapes))
        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() * 4)) as executor:
            future_to_shape = {executor.submit(try_shape_combo, combo): combo for combo in shape_combos}

            for future in as_completed(future_to_shape):
                try:
                    result = future.result(timeout=10)
                    if result:
                        for f in future_to_shape:
                            f.cancel()
                        return result
                except Exception as e:
                    logger.debug(f"Error during shape combo evaluation: {e}")

        raise RuntimeError("Failed to infer input shape by all attempts.")

    else:
        raise TypeError("Unsupported model type provided.")
