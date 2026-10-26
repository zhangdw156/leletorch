


class ModelReleaseTracker:
    def __init__(self,n:int,m:int):
        self.n=n
        self.m=m
        self.left={}
        self.right={}
        self.longest=0

    def release(self,x:int)->None:
        x-=1
        l=self.left.get(x-1,x)
        r=self.right.get(x+1,x)
        if l==0:
            self.longest=max(self.longest,r-l+1)
        self.left[r]=l
        self.right[l]=r
        return


if __name__=="__main__":
    n=int(input())
    m=int(input())
    tracker=ModelReleaseTracker(n,m)
    for _ in range(m):
        i=input()
        if i=='longest':
            print(tracker.longest)
        else:
            _,x=i.split()
            tracker.release(int(x))