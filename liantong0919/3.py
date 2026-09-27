def search(nums, target):
    l,r=0,len(nums)-1
    while l<=r:
      mid=l+(r-l)//2
      if nums[mid]==target:
         return mid
      if nums[l]<=nums[mid]:
         if nums[l]<=target<nums[mid]:
            r=mid-1
         else:
            l=mid+1
    else:
        if nums[mid]<target<=nums[r]:
            l=mid+1
        else:
            r=mid-1
    return -1


if __name__ == "__main__":
    # n, target = map(int, input().split())
    # nums = list(map(int, input().split())) 
    n,target = 7, 0
    nums = [4,5,6,7,0,1,2]
    print(search(nums, target))
