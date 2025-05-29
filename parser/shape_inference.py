import random
import re
import threading
import warnings
from collections import defaultdict
from threading import Lock, RLock

import torch
import itertools
import inspect
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union
from torch import nn, Tensor
from torch.jit import ScriptModule
import logging

logger = logging.getLogger(__name__)
random.seed(42)

def get_input_shapes(
    model: Union[nn.Module, ScriptModule],
    max_args: int = 6,
    ndims: int = 3,
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
            if name in ("self"):
                continue
            elif name in ("args", "kwargs"):
                warnings.warn("args and kwargs are not supported, and will be passed here")
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
                    warnings.warn(f"No annotation for param: {param}, defaulting to None")
                    non_tensor_defaults[name] = None

        if len(tensor_args) > max_args:
            raise RuntimeError(f"Too many tensor inputs to infer (> {max_args})")

        batch_size = 1
        per_input_min_dims = {input_name: 1 for input_name in tensor_args}
        failed_combos = set()
        postponed_combos = set()
        failed_combos_lock = RLock()
        postponed_combos_lock = RLock()
        if dim_sizes is None:
            dim_sizes = [1, 2, 8, 16, 32, 64, 128, 256]

        # ---------- 生成候选形状 ----------
        def generate_candidate_shapes(ndims: int, min_required_dims: int) -> List[Tuple[int, ...]]:
            shapes = set()
            for ndim in range(min_required_dims, ndims + 1):
                for dims in itertools.product(dim_sizes, repeat=ndim):
                    shapes.add(dims)
            return list(shapes)

        # ---------- 初始化所有输入候选 shape ----------
        def build_candidate_shapes():
            raw_shapes_map = {}
            candidate_shapes = []
            for name in tensor_args:
                min_dims = per_input_min_dims.get(name, 1)
                raw_shapes = generate_candidate_shapes(ndims=ndims-1, min_required_dims=min_dims)
                raw_shapes_map[name] = raw_shapes
                candidate_shapes.append([(batch_size,) + shape for shape in raw_shapes])
            return raw_shapes_map, candidate_shapes

        raw_shapes_map, candidate_shapes = build_candidate_shapes()
        logger.debug(f"Raw_shapes without batch size: {raw_shapes_map}")
        shape_combos = list(itertools.product(*candidate_shapes))
        random.shuffle(shape_combos)
        logger.info(f"Total input shape combinations to try: {len(shape_combos)}")

        # prebuild shape-map for error feedback
        dim_to_combos = defaultdict(list)
        for combo in shape_combos:
            for i, shape in enumerate(combo):
                dim_to_combos[(shape, i)].append(combo)
                dim_to_combos[len(shape)].append(combo)

        tensor_cache = {}
        def get_tensor(shape):
            if shape not in tensor_cache:
                tensor_cache[shape] = torch.randn(shape).float()
            return tensor_cache[shape]

        # ---------- 主逻辑 ----------
        def try_shape_combo(shape_combo):
            """
            Try one combo of input shapes.
            :param shape_combo:
            :return:
            """
            thread_id = threading.get_ident()  # 获取当前线程ID
            logger.debug(f"Thread ID {thread_id} is processing combo {future_to_shape[future]}")

            if shape_combo in failed_combos or shape_combo in postponed_combos:
                return None
            try:
                input_kwargs = {
                    name: get_tensor(shape)
                    for name, shape in zip(tensor_args, shape_combo)
                }
                input_kwargs.update(non_tensor_defaults)

                model.eval()
                with torch.no_grad():
                    model(**input_kwargs)

                logger.info(f"Success with shape combo: {shape_combo}")
                return {name: tuple(tensor.shape) for name, tensor in input_kwargs.items()}

            except Exception as e:
                msg = str(e)
                logger.debug(f"Failed combo {shape_combo}: {msg}")
                with failed_combos_lock:
                    failed_combos.add(shape_combo)

                max_similar = 10000

                error_match_0 = re.search(r"not enough values to unpack.*?expected (\d+).*?got (\d+)", msg)
                if error_match_0:
                    expected_dims = int(error_match_0.group(1))
                    actual_dims = int(error_match_0.group(2))

                    similar_list = dim_to_combos.get(actual_dims, [])
                    # randomly pick a postion
                    total = len(similar_list)
                    start = random.randint(0, max(0, total - 1))
                    similar_combos = similar_list[start:start + max_similar]
                    similar_combos = [
                        combo for combo in similar_combos
                        if combo not in failed_combos and combo not in postponed_combos
                    ]

                    logger.debug(f"Postponing shape combos due to dim mismatch like: {similar_combos[:1]}")
                    logger.debug(f"Current postponed combos: {len(postponed_combos)}")
                    with postponed_combos_lock:
                        postponed_combos.update(set(similar_combos))

                error_match_1 = re.search("for tensor number (\d+) in the list", msg)
                if error_match_1:
                    tensor_index = int(error_match_1.group(1))
                    failed_shape = shape_combo[tensor_index]
                    similar_list = dim_to_combos.get((failed_shape, tensor_index), [])
                    # randomly pick a postion
                    total = len(similar_list)
                    start = random.randint(0, max(0, total - 1))
                    similar_combos = similar_list[start:start + max_similar]
                    similar_combos = [
                        combo for combo in similar_combos
                        if combo not in failed_combos and combo not in postponed_combos
                    ]
                    logger.debug(f"Postponing shape combos due to dim mismatch like: {similar_combos[:1]}")
                    logger.debug(f"Current postponed combos: {len(postponed_combos)}")
                    with postponed_combos_lock:
                        postponed_combos.update(set(similar_combos))

                return None

        # ---------- 分批多线程尝试主组合 ----------
        max_workers = min(16, os.cpu_count() * 4)
        batch_size = 4096  # 每次尝试最多4096个组合，避免爆内存
        result = None

        for i in range(0, len(shape_combos), batch_size):
            current_batch = shape_combos[i:i + batch_size]
            future_to_shape = {}

            def handle_future(future):
                try:
                    res = future.result(timeout=10)
                    return res
                except Exception as e:
                    logger.debug(f"Error during shape combo evaluation: {e}")
                    return None

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for combo in current_batch:
                    try:
                        future = executor.submit(try_shape_combo, combo)
                        future_to_shape[future] = combo
                    except Exception as e:
                        logger.debug(f"Failed to submit task for combo {combo}: {e}")
                        continue
                for future in as_completed(future_to_shape):
                    result = handle_future(future)
                    if result:
                        for f in future_to_shape:
                            f.cancel()
                        break

            if result:
                break

        # ---------- 延后组合重试 ----------
        if not result:
            logger.info("Retrying postponed shape combinations")
            for shape_combo in list(postponed_combos):
                if shape_combo in failed_combos:
                    continue
                try:
                    input_kwargs = {
                        name: torch.randn(shape).float()
                        for name, shape in zip(tensor_args, shape_combo)
                    }
                    input_kwargs.update(non_tensor_defaults)

                    model.eval()
                    with torch.no_grad():
                        model(**input_kwargs)

                    logger.info(f"Recovered with original shape combo: {shape_combo}")
                    result = {name: tuple(tensor.shape) for name, tensor in input_kwargs.items()}
                    break
                except Exception as e2:
                    logger.debug(f"Retry failed for combo {shape_combo}: {e2}")
                    failed_combos.add(shape_combo)
            if result:
                return result

        if result:
            return result
        else:
            raise RuntimeError("Failed to infer input shape by all attempts.")

    else:
        raise TypeError("Unsupported model type provided.")
