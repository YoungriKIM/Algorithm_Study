# https://programmers.co.kr/skill_checks/276538


# test 1
'''
n = 8
a = 4
b = 7

def solution(n,a,b):
    a, b = min(a,b), max(a,b)
    answer = 1
    while True:
        if a % 2 != 0 and b == a + 1:
            return answer
        
        if a % 2 != 0:
            a = int(a/2)+1
        elif a % 2 == 0:
             a = int(a/2)
        if b % 2 != 0:
            b = int(b/2)+1
        elif b % 2 == 0:
             b = int(b/2)
        answer += 1
        '''

# test 2

def solution(n):
    count = 0
    for i in range(1, n):
        s = i
        for j in range(i+1, n):
            s+= j
            if s == n:
                count += 1
                break
            elif s > n:
                break
    return count+1


solution(15)

def solution(n):
    count = 1
    for i in range(n):
        if i == n : count += 1
        elif i + (i+1) == n : count += 1
        elif i + (i+1) + (i+2) == n : count += 1
        elif i + (i+1) + (i+2) + (i+3) == n : count += 1
        elif i + (i+1) + (i+2) + (i+3) + (i+4) == n : count += 1
    print(count)

solution(15)
