"""原创练习 2：拼接流式返回的 Tool Call。

题目：实现 merge_tool_deltas(chunks)。
每个片段包含整数 index，以及可选的 id、name、arguments 字符串。
不同调用的片段可能交错；同一调用的 name、arguments 都可能被拆开。

规则：
1. 按 index 合并；id 第一次出现时记录，name 和 arguments 按到达顺序拼接。
2. 输入结束后才解析 arguments，它必须是合法 JSON 对象。
3. 按 index 升序返回列表，每项包含 id、name、arguments、error。
4. 成功时 arguments 是字典、error 是 None；失败时 arguments 是 None、
   error 是错误字符串。一个调用出错不影响其他调用。
5. 保证各片段的字段类型正确，每个调用至少提供一次 id，且 id 一致。

思路：用字典按 index 聚合字符串，最后逐项解析 JSON 并单独处理异常。
"""

import json


def merge_tool_deltas(chunks):
    calls = {}
    for chunk in chunks:
        index = chunk['index']
        if index not in calls:
            calls[index] = {'id': None, 'name': '', 'arguments': ''}
        call = calls[index]
        if call['id'] is None and 'id' in chunk:
            call['id'] = chunk['id']
        call['name'] += chunk.get('name', '')
        call['arguments'] += chunk.get('arguments', '')

    results = []
    for index in sorted(calls):
        call = calls[index]
        try:
            args = json.loads(call['arguments'])
            if not isinstance(args, dict):
                raise ValueError('参数必须是 JSON 对象')
            call['arguments'] = args
            call['error'] = None
        except ValueError as exc:
            call['arguments'] = None
            call['error'] = str(exc)
        results.append(call)
    return results


if __name__ == '__main__':
    chunks = [
        {'index': 1, 'id': 'c2', 'name': 'add', 'arguments': '{"a":1,"b":'},
        {'index': 0, 'id': 'c1', 'name': 'se', 'arguments': '{"city":"北'},
        {'index': 1, 'arguments': '2}'},
        {'index': 0, 'name': 'arch', 'arguments': '京"}'},
        {'index': 2, 'id': 'c3', 'name': 'broken', 'arguments': '{'},
    ]
    answer = merge_tool_deltas(chunks)
    assert answer[0] == {
        'id': 'c1', 'name': 'search', 'arguments': {'city': '北京'}, 'error': None,
    }
    assert answer[1]['arguments'] == {'a': 1, 'b': 2}
    assert answer[2]['arguments'] is None and answer[2]['error']
    print(answer)
