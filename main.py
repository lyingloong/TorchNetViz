import argparse
import ast
import base64
import inspect
import os
from parser.model_parser import parse_model, save_model_structure
from visualizer.graph_visualizer import visualize_structure
import json
from typing import Optional, Dict, Tuple, Union, List, Any
from models.custom_layers import *
from parser.utils import auto_instantiate

def parse_input_shapes(input_str: Optional[str]) -> Optional[Dict[str, Tuple[int, ...]]]:
    """
    将用户传入的 input_shapes 参数解析为 Dict[str, shape]
    支持两种格式：
        - JSON 字符串：'{"x": [1, 3, 224, 224]}'
        - Base64 编码过的 JSON 字符串（推荐用于命令行）
    """
    if not input_str:
        return None

    try:
        # 先尝试直接解析为 JSON
        shapes = json.loads(input_str)
    except json.JSONDecodeError:
        try:
            # 如果失败，尝试解码 Base64 再解析
            decoded = base64.b64decode(input_str).decode("utf-8")
            shapes = json.loads(decoded)
        except Exception as decode_error:
            raise ValueError(f"Invalid input_shapes format. Not valid JSON or Base64: {decode_error}")

    # 转换 list to tuple
    return {k: tuple(v) for k, v in shapes.items()}


def parse_args_argument(arg_list_str: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """
    解析用户传入的 --args 参数列表，每个元素是一个 base64 编码的配置字符串或基本类型。

    Args:
        arg_list_str (Optional[str]): 命令行传入的字符串参数列表

    Returns:
        Optional[List[Dict[str, Any]]]: 每个参数经过处理后的内容
    """
    if not arg_list_str:
        return None

    try:
        # 安全地将字符串转为 Python 列表（比 json.loads 更宽容）
        base64_list = ast.literal_eval(arg_list_str)
        if not isinstance(base64_list, list):
            raise ValueError("参数应为 base64 字符串组成的列表")
    except Exception as e:
        raise ValueError(f"无法将参数解析为列表: {e}")

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
                raise ValueError(f"无法解析列表参数: {arg_value}")
        elif arg_type == "int":
            try:
                parsed = int(arg_value)
            except ValueError:
                raise ValueError(f"无法转换为整数: {arg_value}")
        elif arg_type == "float":
            try:
                parsed = float(arg_value)
            except ValueError:
                raise ValueError(f"无法转换为浮点数: {arg_value}")
        elif arg_type == "configs":
            # 动态创建配置类并实例化
            class Configs:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            # 使用字典创建配置对象
            parsed = Configs(**arg_value)
        elif arg_type == "class":
            # 动态导入类
            module_name, class_name = "models.builtin", arg_value
            try:
                module = __import__(module_name, fromlist=[class_name])
                parsed = getattr(module, class_name)
                if not inspect.isclass(parsed) or not issubclass(parsed, nn.Module):
                    raise TypeError(f"{class_name} 不是 torch.nn.Module 子类")
            except (ImportError, AttributeError) as e:
                raise ValueError(f"无法从 {module_name} 导入模型类 {class_name}: {e}")
        else:
            raise ValueError(f"未知参数类型: {arg_type}")

        results.append({"parsed": parsed, "type": arg_type})

    return results


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
    parser.add_argument("--model_file", type=str, default=None,
                        help="自定义模型所在文件名（不含.py），默认从内置模型中查找")
    parser.add_argument("--input_dim_sizes", type=str,  default=None,
                        help="输入维度大小参考列表")
    parser.add_argument("--ndims", type=int, default=3,
                        help="forward输入最大维度数(含batch)")
    args = parser.parse_args()

    if args.pt:
        raise NotImplementedError("暂不支持加载 PyTorch 模型")
    elif args.model_file:
        # 动态导入用户自定义模块中的模型类
        module_name = f"models.{args.model_file}"
        try:
            module = __import__(module_name, fromlist=[args.model])
            model_class = getattr(module, args.model)
            if not issubclass(model_class, nn.Module):
                raise TypeError(f"{args.model} 不是 torch.nn.Module 的子类")
        except (ImportError, AttributeError) as e:
            raise ValueError(f"无法从 {module_name} 导入模型类 {args.model}: {e}")
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
    visualize_structure(result["structure"], result["connections"],
                        inputs=result["inputs"], output_path=f"output/{args.model}/{args.model}.html")
