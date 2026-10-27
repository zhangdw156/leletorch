"""题目：实现带工具调用的 Agent 循环。

请实现 run_agent(question, tools, max_steps=10) 的内部 Agent 循环，
模型调用可直接使用 llm.chat()。该函数需要支持工具调用参数的 JSON Schema
校验，需要限制最大循环轮数；工具调用或执行发生异常时，不应直接抛出异常，
而应将错误信息作为 observation 返回给大模型，以便其继续决策和调用工具；
同一工具以相同参数连续调用达到 3 次及以上时，应终止循环，避免陷入重复调用。

答案的接口约定：
1. 依赖 jsonschema；调用前由外部在本模块绑定 llm 对象。
2. llm.chat(messages=..., tools=...) 返回一条消息字典，包含 content 和
   可选的 tool_calls；工具调用格式见函数内的注释。
3. tools 是名称到工具配置的映射；配置包含 func、parameters（JSON Schema）
   和可选的 description。func 接收关键字参数，返回值应尽量可 JSON 序列化。
4. max_steps 限制模型调用轮数，每轮可以包含多个工具调用。
5. 第三次连续相同调用出现时直接终止，不再执行该次工具调用；失败调用也计数。
6. 返回字典中的 status 为 completed、repeated_call 或 max_steps。
   模型服务本身的网络错误重试由 llm.chat() 外层处理。

思路：模型决策 -> 解析参数 -> 检测重复 -> 校验并执行工具 -> 回传 observation。
使用排序后的 JSON 作为调用签名，使参数键顺序和 JSON 格式空白不影响比较。
连续次数按工具调用顺序统计：A -> A -> B -> A 不会触发三次重复限制。
"""

import json

from jsonschema import ValidationError, validate


def run_agent(question, tools, max_steps=10):
    """执行 Agent 循环，返回最终回答或终止原因。"""
    messages = [{"role": "user", "content": question}]

    # 只将工具描述和参数 Schema 交给模型，Python 函数保留在本地执行。
    tool_specs = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": tool["parameters"],
            },
        }
        for name, tool in tools.items()
    ]

    last_signature = None
    consecutive_count = 0

    for _ in range(max_steps):
        # llm 由调用方提供。返回示例：
        # {
        #     "content": None,
        #     "tool_calls": [{
        #         "id": "call_1",
        #         "function": {
        #             "name": "add",
        #             "arguments": '{"a": 1, "b": 2}'
        #         }
        #     }]
        # }
        reply = llm.chat(messages=messages, tools=tool_specs)
        calls = reply.get("tool_calls") or []

        if not calls:
            return {
                "status": "completed",
                "answer": reply.get("content") or "",
            }

        # 必须先保留模型发出的工具调用，再追加对应的执行结果。
        messages.append({
            "role": "assistant",
            "content": reply.get("content"),
            "tool_calls": calls,
        })

        for call in calls:
            function = call["function"]
            name = function["name"]
            raw_args = function.get("arguments", "{}")

            # 解析参数并生成规范化签名；非法参数同样参与重复检测。
            parse_error = None
            try:
                args = (
                    json.loads(raw_args)
                    if isinstance(raw_args, str)
                    else raw_args
                )
                canonical_args = json.dumps(
                    args,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                signature = (name, "json", canonical_args)
            except (ValueError, TypeError) as exc:
                parse_error = str(exc)
                signature = (name, "invalid", repr(raw_args))

            if signature == last_signature:
                consecutive_count += 1
            else:
                last_signature = signature
                consecutive_count = 1

            if consecutive_count >= 3:
                return {
                    "status": "repeated_call",
                    "error": f"工具 {name} 连续三次使用相同参数",
                }

            try:
                if parse_error is not None:
                    raise ValueError(f"参数不是合法 JSON：{parse_error}")

                if name not in tools:
                    raise ValueError(f"未知工具：{name}")

                if not isinstance(args, dict):
                    raise ValueError("工具参数必须是 JSON 对象")

                tool = tools[name]
                validate(instance=args, schema=tool["parameters"])

                result = tool["func"](**args)
                observation = json.dumps(
                    {"ok": True, "result": result},
                    ensure_ascii=False,
                    default=str,
                )
            except Exception as exc:
                # 在工具边界捕获异常，让模型看到错误后有机会修正调用。
                error = (
                    f"参数校验失败：{exc.message}"
                    if isinstance(exc, ValidationError)
                    else f"{type(exc).__name__}: {exc}"
                )
                observation = json.dumps(
                    {"ok": False, "error": error},
                    ensure_ascii=False,
                )

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": observation,
            })

    return {
        "status": "max_steps",
        "error": f"已达到最大循环轮数 {max_steps}",
    }


# 工具定义示例；接入 llm 后可调用 run_agent("1 加 2 等于多少？", EXAMPLE_TOOLS)。
EXAMPLE_TOOLS = {
    "add": {
        "description": "计算两个数的和",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        "func": lambda a, b: a + b,
    }
}
