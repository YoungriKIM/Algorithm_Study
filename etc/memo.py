# 답안 예시

# ---------------------------------------------------------
# DFS로 특정 노드를 방문하고 연결된 모든 노드들도 방문하자
def dfs(x,y):
    if x <= -1 or x >= n or y <= -1 or y >= m:
        return False
    if graph[x][y] == 0:    # 해당 칸이 0이라면
        # 방문했다고 처리하자
        graph[x][y] = 1
        # 그리고 상하좌우 칸도 재귀적으로 호출
        dfs(x-1, y)
        dfs(x+1, y) 
        dfs(x, y-1) 
        dfs(x, y+1) 
        return True
    return False

# ---------------------------------------------------------
# n,m 을 공백을 기준으로 구분하여 받기
n,m = map(int, input().split())

# 2차원 리스트의 맵 정보 입력 받기
graph = []
for i in range(n):
    graph.append(list(map(int, input().split())))

# 모든 노드에 대하여 음료수 채우기 해서 True값만 세기
result = 0
for i in range(n):
    for j in range(m):
        # 현재 위치에서 DFS 수행
        if dfs(i,j) == True:
            result += 1
        
print(result)

1 1
0
2 2
0 1
1 0
3 2
1 1 1
1 1 1
5 4
1 0 1 0 0
1 0 0 0 0
1 0 1 0 1
1 0 0 1 0
5 4
1 1 1 0 1
1 0 1 0 1
1 0 1 0 1
1 0 1 1 1
5 5
1 0 1 0 1
0 0 0 0 0
1 0 1 0 1
0 0 0 0 0
1 0 1 0 1
0 0