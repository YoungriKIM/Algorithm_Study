<<<<<<< HEAD
import sys
from collections import deque
 
n, m = map(int, sys.stdin.readline().split())
maze = []
 
for i in range(m):
    maze.append(sys.stdin.readline())
 
visited = [[-1]*n for _ in range(m)]
visited[0][0] = 0
q = deque()
dx = [1,-1,0,0]
dy = [0,0,1,-1]
q.append([0, 0])
 
while q:
    x, y = q.popleft()
    for i in range(4):
        nx, ny = x+dx[i], y+dy[i]
        if nx>=0 and nx<m and ny >=0 and ny<n:
            if visited[nx][ny] == -1:
                if maze[nx][ny] == '0':
                    q.appendleft([nx, ny])
                    visited[nx][ny] = visited[x][y]
                elif maze[nx][ny] == '1':
                    q.append([nx, ny])
                    visited[nx][ny] = visited[x][y]+1
 
print(visited[m-1][n-1])
=======
N,M,V=map(int,input().split())
matrix=[[0]*(N+1) for i in range(N+1)]
for i in range(M):
    a,b = map(int,input().split())
    matrix[a][b]=matrix[b][a]=1
visit_list=[0]*(N+1)

def dfs(V):
    visit_list[V]=1 #방문한 점 1로 표시
    print(V, end=' ')
    for i in range(1,N+1):
        if(visit_list[i]==0 and matrix[V][i]==1):
            dfs(i)

def bfs(V):
    queue=[V] #들려야 할 정점 저장
    visit_list[V]=0 #방문한 점 0으로 표시
    while queue:
        V=queue.pop(0)
        print(V, end=' ')
        for i in range(1, N+1):
            if(visit_list[i]==1 and matrix[V][i]==1):
                queue.append(i)
                visit_list[i]=0

dfs(V)
print()
bfs(V)
>>>>>>> ea2f4fb63a2143e40310eb4b6c03742e684f73ec
