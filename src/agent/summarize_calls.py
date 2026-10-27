"""原创练习 1：乱序轨迹中的工具调用统计。

题目：实现 summarize_calls(messages)，返回
{'total': int, 'success': int, 'failure': int, 'unknown': int}。

输入格式见文件末尾的例子。规则：
1. 只统计 assistant 消息中 function.name == 'Task' 的调用，按 id 去重。
2. tool 消息靠 tool_call_id 配对，结果可能出现在调用声明之前。
3. 同一 id 有多个结果时，以列表中第一次出现的结果为准，包括空结果。
4. 没有结果的调用计入 unknown；没有对应 Task 调用的结果忽略。
5. 普通 assistant 消息可能没有 tool_calls，tool 消息可能没有 content。
6. classify 已提供：None、空白内容为 unknown，包含失败标记为 failure，
   其余为 success。其他结构字段保证符合示例，同一 id 对应同一次调用。

思路：先分别收集调用 id 和第一次结果，再逐个调用计数，天然支持乱序。
检查：total == success + failure + unknown。
"""


FAILURE_MARKERS = ('agent task failed', '<tool_use_error>')


def classify(result):
    """题目提供的辅助函数。"""
    if result is None or not str(result).strip():
        return 'unknown'
    text = str(result).strip().lower()
    if any(marker in text for marker in FAILURE_MARKERS):
        return 'failure'
    return 'success'


def summarize_calls(messages):
    task_ids = set()
    results = {}

    for msg in messages:
        if msg['role'] == 'assistant':
            for call in msg.get('tool_calls', []):
                if call['function']['name'] == 'Task':
                    task_ids.add(call['id'])
        elif msg['role'] == 'tool':
            call_id = msg['tool_call_id']
            if call_id not in results:
                results[call_id] = msg.get('content')

    counts = {'total': len(task_ids), 'success': 0, 'failure': 0, 'unknown': 0}
    for call_id in task_ids:
        status = classify(results.get(call_id))
        counts[status] += 1
    return counts


if __name__ == '__main__':
    messages = [
        # 返回先到，也能与后面的声明配对。
        {'role': 'tool', 'tool_call_id': 'c1', 'content': '完成'},
        {'role': 'assistant', 'content': '开始处理'},
        {'role': 'assistant', 'tool_calls': [
            {'id': 'c1', 'function': {'name': 'Task'}},
            {'id': 'c2', 'function': {'name': 'Task'}},
            {'id': 'c3', 'function': {'name': 'Task'}},
            {'id': 'b1', 'function': {'name': 'Bash'}},
        ]},
        {'role': 'tool', 'tool_call_id': 'c2', 'content': '<tool_use_error>boom'},
        # 重复结果不能覆盖第一次结果；c3 始终没有返回。
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'agent task failed'},
        {'role': 'tool', 'tool_call_id': 'b1', 'content': 'a.py'},
    ]
    answer = summarize_calls(messages)
    assert answer == {'total': 3, 'success': 1, 'failure': 1, 'unknown': 1}
    print(answer)
