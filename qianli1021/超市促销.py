"""千里科技：超市促销。双堆 + 数组模拟双向链表，O(n log n)。"""

import heapq
import sys


def assign_products(a, k):
    n = len(a)
    if n == 0:
        return [], []
    heaps = [
        [(a[i], i) for i in range(n)],
        [(-a[i], -i) for i in range(n)],
    ]
    for heap in heaps:
        heapq.heapify(heap)

    left = [i - 1 for i in range(n)]
    right = [i + 1 for i in range(n)]
    right[-1] = -1
    owner = [-1] * n
    remaining = n

    def remove(i, player):
        nonlocal remaining
        owner[i] = player
        remaining -= 1
        l, r = left[i], right[i]
        if l != -1:
            right[l] = r
        if r != -1:
            left[r] = l

    player = 0
    while remaining:
        while True:
            _, index = heapq.heappop(heaps[player])
            i = index if player == 0 else -index
            if owner[i] == -1:
                break
        l, r = left[i], right[i]
        remove(i, player)
        # 左侧沿 left 走，右侧沿 right 走；删除前保存下一个位置。
        for j, direction in ((l, left), (r, right)):
            for _ in range(k):
                if j == -1:
                    break
                nxt = direction[j]
                remove(j, player)
                j = nxt
        player ^= 1

    answer = [[], []]
    # 按原编号扫描，直接获得升序答案，无须再排序。
    for i, player in enumerate(owner, 1):
        answer[player].append(i)
    return answer


def main():
    data = list(map(int, sys.stdin.buffer.read().split()))
    n, k = data[:2]
    for result in assign_products(data[2:2 + n], k):
        print(*result)


if __name__ == "__main__":
    main()
