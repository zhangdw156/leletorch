from collections import deque


def main():
    n,m=map(int,input().split())
    grid=[]
    for _ in range(n):
        grid.append(input().strip())
    dist=[[-1]*m for _ in range(n)]
    q=deque()
    dirs=[(-1,0),(1,0),(0,-1),(0,1)]
    for i in range(n):
        for j in range(m):
            if grid[i][j]=='1':
                dist[i][j]=0
                q.append((i,j))
    while q:
        x,y=q.popleft()
        for dx,dy in dirs:
            nx,ny=x+dx,y+dy
            if 0<=nx<n and 0<=ny<m and dist[nx][ny]==-1 and grid[nx][ny]=='0':
                dist[nx][ny]=dist[x][y]+1
                q.append((nx,ny))
    max_dist=0
    cnt=0
    for i in range(n):
        for j in range(m):
            d=dist[i][j]
            if d>0:
                if d>max_dist:
                    max_dist=d
                    cnt=1
                elif d==max_dist:
                    cnt+=1
    print(max_dist,cnt)


if __name__ == '__main__':
    main()