

n,k=map(int,input().split())
a=list(map(int,input().split()))
a.sort()
pre=[0]*(n+1)
for i in range(n):
    pre[i+1]=pre[i]+a[i]

def can(m:int):
    for l in range(0,n-m+1):
        r=l+m-1
        mid=l+m//2
        left_cost=a[mid]*(mid-l)-pre[mid]+pre[l]
        right_cost=pre[r+1]-pre[mid+1]-a[mid]*(r-mid)
        if left_cost+right_cost<=k:
            return True
    return False

lo,hi=1,n
while lo<hi:
    mid=(lo+hi+1)//2
    if can(mid):
        lo=mid
    else:
        hi=mid-1
print(lo)