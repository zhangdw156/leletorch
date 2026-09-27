def single_number(nums):
    ans=0
    for num in nums:
      ans^=num
    return ans


if __name__ == "__main__":
    n = int(input())
    nums = list(map(int, input().split()))
    print(single_number(nums))