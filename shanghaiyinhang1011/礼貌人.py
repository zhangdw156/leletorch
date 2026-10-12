from collections import deque

def main():
    n=int(input())
    m=int(input())
    dir=1
    res=deque(list(range(1,n+1)))
    for _ in range(m):
        ch=input()
        if ch=='A':
            if dir==1:
                res.append(res.popleft())
            else:
                res.appendleft(res.pop())
        else:
            dir=-dir
    res=list(res)
    if dir==1:
        for i in range(n):
            print(res[i])
    else:
        for i in range(n-1,-1,-1):
            print(res[i])
    return

if __name__=='__main__':
    main()
