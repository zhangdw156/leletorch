"""LeetCode 22：括号生成（中等）。

题目：https://leetcode.cn/problems/generate-parentheses/description/

给定整数 n，生成由 n 对括号组成的所有有效括号组合。
这里的 n 表示括号的对数，每个结果的长度为 2 * n。
有效组合要求左右括号总数相同，且任意前缀的右括号数不超过左括号数。

示例 1：
    输入：n = 3
    输出：["((()))", "(()())", "(())()", "()(())", "()()()"]
示例 2：
    输入：n = 1
    输出：["()"]
约束：1 <= n <= 8。

思路：回溯维护当前路径和已使用的左右括号数。
    左括号不足 n 个时可以添加左括号；右括号少于左括号时可以添加右括号。
    路径长度达到 2 * n 时记录答案。这样只扩展有效前缀，无须事后去重。
复杂度：设 C_n = comb(2 * n, n) // (n + 1) 为第 n 个卡特兰数，
    时间 O(n * C_n)，辅助空间 O(n)，保存所有答案需要 O(n * C_n) 空间。
"""


class Solution:
    def generateParenthesis(self, n: int) -> list[str]:
        result = []
        path = []

        def backtrack(left: int, right: int) -> None:
            if len(path) == 2 * n:
                result.append("".join(path))
                return

            if left < n:
                path.append("(")
                backtrack(left + 1, right)
                path.pop()

            # 只有存在尚未配对的左括号时，才能添加右括号。
            if right < left:
                path.append(")")
                backtrack(left, right + 1)
                path.pop()

        backtrack(0, 0)
        return result


if __name__ == "__main__":
    solution = Solution()
    print(solution.generateParenthesis(3))
    print(solution.generateParenthesis(1))
