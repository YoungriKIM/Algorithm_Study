# 참고코드

from collections import deque

MAX = 100001
n, k = map(int, input().split())
array = [0]*MAX

def bfs():
    q = deque([n])
    while q:
        x = q.popleft()
        if x == k:
            return array[x]
        for nx in (x-1, x+1, x*2):
            if 0 <= nx < MAX and not array[nx]:
                if nx == x*2 and x != 0:
                    array[nx] = array[x]
                    q.appendleft(nx)
                else:
                    array[nx] = array[x] + 1
                    q.append(nx)

print(bfs())
