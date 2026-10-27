"""原创练习 4：带有限重试的工具执行器。

题目：实现 call_with_retry(fn, kwargs, max_attempts, sleep_fn)。
通过 fn(**kwargs) 执行工具，返回包含 ok/result/error/attempts 的字典。

规则：
1. max_attempts 保证是至少为 1 的整数，包含第一次执行。
2. 仅 TimeoutError、ConnectionError 可以重试；其他异常立即返回失败。
3. 两次执行之间通过 sleep_fn 等待，时长依次为 0.1、0.2、0.4……秒。
4. 最后一次失败后不再等待；attempts 记录实际执行次数。
5. 正常返回 None 也算成功。成功时 error=None；失败时 result=None，
   error 为最后一次异常的信息。保证 sleep_fn 自身不会抛异常。

思路：按尝试次数循环，成功立即返回；失败时决定返回还是等待后重试。
末尾例子用 list.append 记录等待时间，不会真的等待，也不需要网络。
"""


def call_with_retry(fn, kwargs, max_attempts, sleep_fn):
    for attempt in range(1, max_attempts + 1):
        try:
            result = fn(**kwargs)
            return {'ok': True, 'result': result, 'error': None, 'attempts': attempt}
        except Exception as exc:
            retryable = isinstance(exc, (TimeoutError, ConnectionError))
            if not retryable or attempt == max_attempts:
                return {
                    'ok': False,
                    'result': None,
                    'error': f'{type(exc).__name__}: {exc}',
                    'attempts': attempt,
                }
        sleep_fn(0.1 * 2 ** (attempt - 1))


if __name__ == '__main__':
    responses = iter([TimeoutError('超时'), ConnectionError('断开连接'), None])

    def fake_tool():
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    waits = []
    answer = call_with_retry(fake_tool, {}, 3, waits.append)
    assert answer == {'ok': True, 'result': None, 'error': None, 'attempts': 3}
    assert waits == [0.1, 0.2]
    print(answer)
    print('等待记录：', waits)
