from math import hypot

def main():
    n,k=map(int,input().split())
    xa,ya,xb,yb=map(int,input().split())
    diffs=[]
    s=0
    for _ in range(n):
        x,y=map(int,input().split())
        da=hypot(x-xa,y-ya)
        db=hypot(x-xb,y-yb)
        diffs.append(da-db)
        s+=db
    diffs.sort()
    for i in range(k):
        s+=diffs[i]
    print(f'{s:.8f}')


if __name__ == '__main__':
    main()