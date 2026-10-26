"""LeetCode 11：盛最多水的容器（中等）。

题目：https://leetcode.cn/problems/container-with-most-water/description/

给定长度为 n 的整数数组 height。第 i 条垂线从 (i, 0) 延伸到
(i, height[i])。选择两条垂线，与 x 轴围成一个不能倾斜的容器，
返回它能容纳的最大水量。
选择下标 left < right 时，水量为
    (right - left) * min(height[left], height[right])。

示例 1：
    输入：height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    输出：49
    解释：选择下标 1 和 8，宽为 7，水高为 7，水量为 49。
示例 2：
    输入：height = [1, 1]
    输出：1
约束：n == len(height)，2 <= n <= 10**5，0 <= height[i] <= 10**4。

思路：双指针从数组两端向中间收缩，每次计算水量，再移动较短的一侧。
    保留短边而向内移动另一端时，宽度减小，水高又不会超过当前短边，
    因而水量不可能变大，可以排除当前短边；两边等高时移动任意一侧即可。
复杂度：时间 O(n)，辅助空间 O(1)。
"""


class Solution:
    def maxArea(self, height: list[int]) -> int:
        left, right = 0, len(height) - 1
        best = 0

        while left < right:
            area = (right - left) * min(height[left], height[right])
            best = max(best, area)

            # 短边限制水高，只有移动短边才可能得到更大的水量。
            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1

        return best


if __name__ == "__main__":
    solution = Solution()
    print(solution.maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]))
    print(solution.maxArea([1, 1]))
