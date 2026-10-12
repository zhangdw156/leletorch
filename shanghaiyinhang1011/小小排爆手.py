from collections import Counter

def main(n,k,a):
    cnt=Counter(a)
    ans=0
    vis=set()
    for x in cnt:
        if x in vis:
            continue
        complement=k-x
        if x==complement:
            if cnt[x]>1:
                ans+=cnt[x]-1
            vis.add(x)
        elif complement in cnt:
            ans+=min(cnt[x],cnt[complement])
            vis.add(x)
            vis.add(complement)
        else:
            vis.add(x)
    print(ans)

if __name__=='__main__':
    # n,k=map(int,input().split())
    # a=list(map(int,input().split()))
    n,k=7,10
    a=[2,8,7,3,2,7,1]
    main(n,k,a)
    