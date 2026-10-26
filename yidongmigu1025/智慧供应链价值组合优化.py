from math import inf
import bisect


def main1():
    """
    单调栈是错误的，只能通过35%数据，原因是dp[i]的状态转移不一定是单调的
    """
    n=int(input())
    a=list(map(int,input().split()))
    b=[x-i for i,x in enumerate(a)]
    st=[(-inf,0)]
    dp=[0]*(n+1)
    for i in range(n):
        while st[-1][0]>b[i]:
            st.pop()
        dp[i+1]=st[-1][1]+b[i]+i
        st.append((b[i],dp[i+1]))
    print(max(dp))


def main2():
    """
    树状数组超时
    """
    n=int(input())
    a=list(map(int,input().split()))
    b=[x-i for i,x in enumerate(a)]
    vals=sorted(set(b))
    rank={v:i+1 for i,v in enumerate(vals)}
    m=len(vals)

    bit=[0]*(m+1)

    ans=-inf

    for i in range(n):
        r=rank[b[i]]
        best=0
        k=r
        while k>0:
            best=max(best,bit[k])
            k-=k&-k
        cur=a[i]+best
        ans=max(ans,cur)
        k=r
        while k<=m:
            bit[k]=max(bit[k],cur)
            k+=k&-k
    print(ans)


class FenwickTreeMax:
    def __init__(self, n):
        self.n = n
        self.tree = [-10**18] * (n + 5)

    def update(self, index, value):
        while index <= self.n:
            if value > self.tree[index]:
                self.tree[index] = value
            index += index & -index

    def query(self, index):
        result = -10**18
        while index > 0:
            result = max(result, self.tree[index])
            index -= index & -index
        return result


def main3():
    """
    BIT也不行，73%
    """
    n = int(input())
    a = list(map(int, input().split()))
    b = [x - i for i, x in enumerate(a)]
    vals = sorted(set(b))
    rank = {v: i + 1 for i, v in enumerate(vals)}
    m = len(vals) 
    ft=FenwickTreeMax(m)
    ans=-10**18
    for i in range(n):
        bi=b[i]
        r=rank[bi]
        best_prev=ft.query(r)
        dp_i=a[i]
        if best_prev!=-10**18:
            dp_i+=best_prev
        if dp_i>ans:
            ans=dp_i
        ft.update(r,dp_i)

    print(ans)



class SegmentTreeMax:
    def __init__(self, size):
        self.n = 1
        while self.n < size:
            self.n <<=1
        self.tree = [-10**18] * (2 * self.n)

    def update(self, index, value, node=1, start=1, end=None):
        if end is None:
            end = self.n
        if start == end:
            self.tree[node] = max(self.tree[node], value)
        else:
            mid = (start + end) // 2
            if index <= mid:
                self.update(index, value, 2 * node, start, mid)
            else:
                self.update(index, value, 2 * node + 1, mid + 1, end)
            self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, left, right, node=1, start=1, end=None):
        if end is None:
            end = self.n
        if right < start or left > end:
            return -10**18
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        left_query = self.query(left, right, 2 * node, start, mid)
        right_query = self.query(left, right, 2 * node + 1, mid + 1, end)
        return max(left_query, right_query)


def main4():
    """
    73%
    """
    n = int(input())
    a = list(map(int, input().split()))
    b = [x - i for i, x in enumerate(a)]
    vals = sorted(set(b))
    rank = {v: i + 1 for i, v in enumerate(vals)}
    m = len(vals)
    st=SegmentTreeMax(m)
    ans=-10**18
    for i in range(n):
        bi=b[i]
        rk=bisect.bisect_right(vals,bi)
        best_prev=st.query(1,rk)
        dp_i=a[i]
        if best_prev!=-10**18:
            dp_i+=best_prev
        if dp_i>ans:
            ans=dp_i
        st.update(rk,dp_i)
    print(ans)

if __name__=="__main__":
    main4()