import argparse
import inspect
import os
import torch
from parser.parser import parse_model, save_model_structure
from visualizer.graph_visualizer import visualize_structure
import torch.nn as nn
import json
from typing import Optional, Dict, Tuple, Union, List
from models.custom_layers import *
from parser.utils import auto_instantiate

def parse_input_shapes(input_str: Optional[str]) -> Optional[Dict[str, Tuple[int, ...]]]:
    """
    将用户传入的 JSON 字符串解析为 Dict[str, shape]
    示例输入：'{"x": [1, 3, 224, 224], "cls_token": [1, 1, 768]}'
    """
    if not input_str:
        return None
    try:
        shapes = json.loads(input_str)
        # 转换 list to tuple
        return {k: tuple(v) for k, v in shapes.items()}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid input_shapes JSON format: {e}")

def parse_args_argument(arg_str: Optional[str]) -> Optional[Union[List, Tuple]]:
    """
    解析用户传入的 --args 参数，格式如 '[1, 3, 224, 224]'。

    支持以下格式：
        - "[1, 3, 224, 224]"   -> list
        - "(1, 3, 224, 224)"   -> tuple
        - "[[3], [64, 3, 3]]"  -> nested list
        - "128"                -> int
        - "0.5"                -> float

    Args:
        arg_str (Optional[str]): 命令行传入的字符串参数

    Returns:
        Optional[Union[List, Tuple]]: 解析后的 Python 数据结构
    """
    if not arg_str:
        return None

    try:
        # 使用 json.loads 处理标准 JSON 格式
        return json.loads(arg_str.replace('(', '[').replace(')', ']'))
    except json.JSONDecodeError:
        # 如果不是标准 JSON，尝试简单类型转换
        try:
            # 尝试转换为 int
            return int(arg_str)
        except ValueError:
            try:
                # 尝试转换为 float
                return float(arg_str)
            except ValueError:
                raise ValueError(f"无法解析参数: {arg_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TorchNetViz")

    parser.add_argument("--model", type=str, default="Linear",
                        help="模型类型")
    parser.add_argument("--pt", type=str, default=None,
                        help="(optional)加载 PyTorch 模型路径")
    parser.add_argument("--args", type=str, default=None,
                        help="(optional)模型参数，格式如 '[1, 3, 224, 224]'")
    parser.add_argument("--input_shapes", type=str, default=None,
                        help='(optional)手动指定输入形状，格式如 {"x": [1, 3, 224, 224]}')

    args = parser.parse_args()

    if args.pt:
        raise NotImplementedError("暂不支持加载 PyTorch 模型")
    else:
        # 自动收集所有 nn.Module 子类
        MODEL_MAP = {
            name.lower(): cls
            for name, cls in globals().items()
            if inspect.isclass(cls) and issubclass(cls, nn.Module)
        }

        # 自动映射模型类
        model_name = args.model.lower()
        if model_name not in MODEL_MAP:
            raise ValueError(f"Unsupported model: {args.model}")
        model_class = MODEL_MAP[model_name]

        if args.args:
            model = model_class(args.args)
        else:
            model = auto_instantiate(model_class)

        os.makedirs(f"./output/{args.model}", exist_ok=True)

        result = parse_model(model, args.input_shapes)
        save_model_structure(result, f"output/{args.model}/{args.model}.json")
        visualize_structure(result["structure"], result["connections"],
                            inputs=result["inputs"], output_path=f"output/{args.model}/{args.model}.html")
