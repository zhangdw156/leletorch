import heapq
import math


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))

def main():
    sx,sy=map(int,input().split())
    tx,ty=map(int,input().split())
    n=int(input())
    id_map={}
    pts=[]
    start=Point(sx,sy)
    target=Point(tx,ty)
    pts.append(start)
    pts.append(target)
    id_map[start]=0
    id_map[target]=1
    special=[]
    for _ in range(n):
        x1,y1,x2,y2,c=map(int,input().split())
        p1=Point(x1,y1)
        p2=Point(x2,y2)
        special.append((p1,p2,c))
        if p1 not in id_map:
            id_map[p1]=len(pts)
            pts.append(p1)
        if p2 not in id_map:
            id_map[p2]=len(pts)
            pts.append(p2)
    m=len(pts)
    g=[[] for _ in range(m)]
    for i in range(m):
        u=pts[i]
        for j in range(m):
            if i==j:
                continue
            v=pts[j]
            d=abs(u.x-v.x)+abs(u.y-v.y)
            g[i].append((j,d))
    for p1,p2,c in special:
        u=id_map[p1]
        v=id_map[p2]
        g[u].append((v,c))
    dist=[10**18]*m
    dist[0]=0
    h=[]
    heapq.heappush(h, (0, 0))
    while h:
        d,u=heapq.heappop(h)
        if d>dist[u]:
            continue
        for v,w in g[u]:
            if dist[v]>dist[u]+w:
                dist[v]=dist[u]+w
                heapq.heappush(h,(dist[v],v))
    ans=dist[1]
    print(ans)

if __name__=="__main__":
    main()





