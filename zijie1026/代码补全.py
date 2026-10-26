CONVERSATIONS = [
    {'role': 'user', 'content': '帮我理清这个仓库'},
    {'role': 'assistant', 'tool_calls': [
        {'id': 'call_1', 'function': {'name': 'Task', 'arguments': '{"subagent_type": "explorer"}'}},
        {'id': 'call_2', 'function': {'name': 'Bash', 'arguments': '{"cmd": "ls"}'}},
    ]},
    {'role': 'tool', 'tool_call_id': 'call_1', 'content': '依赖分析完成，共 3 个'},
    {'role': 'tool', 'tool_call_id': 'call_2', 'content': 'a.py b.py'},
    {'role': 'assistant', 'tool_calls': [
        {'id': 'call_3', 'function': {'name': 'Task', 'arguments': '{"subagent_type": "tester"}'}},
        {'id': 'call_4', 'function': {'name': 'Task', 'arguments': '{"subagent_type": "writer"}'}},
    ]},
    {'role': 'tool', 'tool_call_id': 'call_3', 'content': '<tool_use_error>boom'},
]

FAILURE_MARKERS = ('agent task failed', '<tool_use_error>')

def classify(result):
    """把一次子任务的返回内容判成 'success' / 'failure' / 'unknown'。"""
    if result is None:
        return 'unknown'
    text = str(result).strip()
    if not text:
        return 'unknown'
    lowered = text.lower()
    if any(marker in lowered for marker in FAILURE_MARKERS):
        return 'failure'
    return 'success'

def count_calls(conversations):
    """统计这条轨迹里所有 Task 子任务调用的数量和状态分布。

    规则：
      1. 只统计 function.name == 'Task' 的调用，同一轮里的其他工具调用不算。
      2. 每个调用的状态用 classify(它的返回内容) 得到，返回内容靠 tool_call_id 配对。
      3. 调用有可能压根没有对应的 tool 结果，这种也要算进 total。

    返回：{'total': int, 'success': int, 'failure': int, 'unknown': int}
    """
    res={'total': 0, 'success': 0, 'failure': 0, 'unknown': 0}
    st=set()
    for msg in conversations:
        if msg['role']=='assistant':
            tool_calls=msg['tool_calls']
            for tool_call in tool_calls:
                name=tool_call['function']['name']
                if name=='Task':
                    st.add(tool_call['id'])
                    res['total']+=1
        if msg['role']=='tool':
            call_id=msg['tool_call_id']
            if call_id in st:
                result=(classify(msg['content']))
                res[result]+=1
                st.remove(call_id)
    return res
    
    

def resolve_status(status, payload):
    has_payload = bool(payload and str(payload).strip())

    if status == 'completed':
        # 声称完成且有产出 -> 'returned'
        # 声称完成但没有产出 -> 不能算成功，要能跟正常完成区分开
        return 'returned',payload if has_payload else 'completed',None  # 空 1

    if status == 'timeout':
        # 等超时了但产出已经回来了 -> 按拿到算
        # 超时且没有产出 -> 'timeout'
        return 'completed',payload if has_payload else 'timeout',None  # 空 2

    if status == 'failed':
        # 失败态，注意 payload 里装的是失败原因
        return 'failed',payload   # 空 3

    return status, None
    
################################
def count_calls(conversations):
    res = {'total': 0, 'success': 0, 'failure': 0, 'unknown': 0}
    pending = set()

    for msg in conversations:
        if msg['role'] == 'assistant':
            for tool_call in msg.get('tool_calls', []):
                if tool_call['function']['name'] == 'Task':
                    pending.add(tool_call['id'])
                    res['total'] += 1

        elif msg['role'] == 'tool':
            call_id = msg['tool_call_id']
            if call_id in pending:
                status = classify(msg.get('content'))
                res[status] += 1
                pending.remove(call_id)

    # 没有对应返回结果的调用，计入 unknown
    res['unknown'] += len(pending)  # 这里有歧义
    return res


def resolve_status(status, payload):
    has_payload = bool(payload and str(payload).strip())

    if status == 'completed':
        return ('returned', payload) if has_payload else ('completed', None)

    if status == 'timeout':
        return ('returned', payload) if has_payload else ('timeout', None)

    if status == 'failed':
        return 'failed', payload

    return status, None