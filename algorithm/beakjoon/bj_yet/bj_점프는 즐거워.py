import sys
data = list(sys.stdin.readline().split())
for sample in data:
    n = int(sample[0]);sample = sample[1:]
    if n == 1 : print('Jolly')
    elif n > 1 : 
        r_l = list(set([abs(int(sample[i])-int(sample[i+1])) for i in range(len(sample)-1) if int(sample[i]) != int(sample[i+1])]))
        if r_l == list(range(1, n)) : print('Jolly')
        else: print('Not jolly')
    else: print('Not jolly')