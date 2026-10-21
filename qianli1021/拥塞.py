"""千里科技：拥塞。二分 + BFS，O(n² log A)；Python 常数优化版。

保留 TTL 限制和四方向搜索。能否满足原平台 3 秒限制需在原平台验证。
"""

from collections import deque
import sys


def max_packet(n, ttl, rows):
    if ttl < 2 * (n - 1):
        return 0

    # 外围补一圈 -1，四个方向都可直接访问，无须 divmod 和边界判断。
    width = n + 2
    size = width * width
    capacity = [-1] * size
    for r, row in enumerate(rows, 1):
        start = r * width + 1
        capacity[start:start + n] = row
    start = width + 1
    target = n * width + n
    capacity[start] = capacity[target] = 1_000_001

    # 每轮最多标记一次，20 轮以内，bytearray 足够保存搜索编号。
    visited = bytearray(size)
    stamp = 0

    def check(packet):
        nonlocal stamp
        stamp += 1
        mark = stamp
        seen = visited
        cap = capacity
        stride = width
        end = target
        q = deque([start])
        append = q.append
        pop = q.popleft
        seen[start] = mark

        for _ in range(ttl):
            for _ in range(len(q)):
                p = pop()
                # 先尝试朝右下角的方向；BFS 最短路性质不变。
                nxt = p + 1
                if seen[nxt] != mark and cap[nxt] >= packet:
                    if nxt == end:
                        return True
                    seen[nxt] = mark
                    append(nxt)
                nxt = p + stride
                if seen[nxt] != mark and cap[nxt] >= packet:
                    if nxt == end:
                        return True
                    seen[nxt] = mark
                    append(nxt)
                nxt = p - 1
                if seen[nxt] != mark and cap[nxt] >= packet:
                    seen[nxt] = mark
                    append(nxt)
                nxt = p - stride
                if seen[nxt] != mark and cap[nxt] >= packet:
                    seen[nxt] = mark
                    append(nxt)
            if not q:
                return False
        return False

    low, high = 1, 1_000_000
    answer = 0
    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            answer = mid
            low = mid + 1
        else:
            high = mid - 1
    return answer


def main():
    read = sys.stdin.buffer.readline
    n, ttl = map(int, read().split())
    if ttl < 2 * (n - 1):
        print(0)
        return
    # 逐行读取，避免一次保留 100 万个输入字符串。
    rows = (list(map(int, read().split())) for _ in range(n))
    print(max_packet(n, ttl, rows))


if __name__ == "__main__":
    main()
