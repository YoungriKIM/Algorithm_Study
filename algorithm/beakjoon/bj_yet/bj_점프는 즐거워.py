import sys

data = []
for line in sys.stdin:
    data.append(line.split())
    
for a_l in data:
    n = int(a_l[0]);a_l = a_l[1:]
    if n == 1 : print('Jolly')
    elif n > 1 : 
        r_l = list(set([abs(int(a_l[i])-int(a_l[i+1])) for i in range(len(a_l)-1) if int(a_l[i]) != int(a_l[i+1])]))
        if r_l == list(range(1, n)) : print('Jolly')
        else: print('Not jolly')
    else: print('Not jolly')
