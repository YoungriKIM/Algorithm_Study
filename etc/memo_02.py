from collections import deque


N, M, V = map(int, input().split())
graph = [[0,0]]
for _ in range(M):
    graph.append(list(map(int, input().split())))

if 1 > N or N > 1000 or 1 > M or M > 10000: pass
else: 
    # -----------------------------------------------
    nums=[]
    for i in graph:
        for v in i :
            nums.append(v)
    stand = list(j for j in set(nums) if j != 0)

    newgraph = [[0,0]]
    for x in stand:
        num = x
        numlist = []
        for y in graph:
            if x == y[0]:
                numlist.append(y[1])
            if x == y[1]:
                numlist.append(y[0])
        newgraph.append(set(numlist))

    visited1 = [False]*(N+1)
    # -----------------------------------------------
    def dfs(graph, v, visited):
        visited[v] = True
        print(v, end =' ')
        for i in graph[1:]:
            if v not in i:
                for c in i:
                    if visited[c] == False:
                        visited[c] = True
                        dfs(graph, c, visited)

    dfs(newgraph, V, visited1);print()
    # -----------------------------------------------
    visited2 = [False]*(N+1)

    def bfs(graph, V, visited):
        q = deque([V])
        visited[V] = True
        while q:
            v = q.popleft()
            print(v, end=' ')
            for i in graph[1:]:
                if v not in i:
                    for c in i:
                        if visited[c] == False:
                            q.append(c)
                            visited[c] = True

    bfs(newgraph, V, visited2)