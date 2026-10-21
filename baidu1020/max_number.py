def main(n: int, a: list[int]) -> None:
    a = sorted(set(a), reverse=True)
    s = str(n)

    # i：当前处理的位置
    # num：已经组成的数字
    # tight：是否还受 n 对应数位的限制
    def dfs(i, num, tight):
        if i == len(s):
            return num if num > 0 or 0 in a else -1

        limit = int(s[i]) if tight else 9

        # 贪心：从大到小尝试
        for d in a:
            if d > limit or (num == 0 and d == 0):
                continue

            ans = dfs(i + 1, num * 10 + d, tight and d == limit)
            if ans != -1:
                return ans

            # 后续无解，回溯尝试下一个数字

        # 还没开始选数字，可以跳过当前位
        # 位数变少后，一定小于 n，因此 tight=False
        if num == 0:
            return dfs(i + 1, 0, False)

        return -1

    print(dfs(0, 0, True))


if __name__ == "__main__":
    main(200, [2, 4, 9])    # 99
    main(23121, [2, 4, 9])  # 22999
    main(2240, [2, 4, 9])   # 2229