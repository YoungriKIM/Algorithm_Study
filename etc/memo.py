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