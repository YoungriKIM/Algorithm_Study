n = 3
start = 1
end = 3
# roads = [[1, 2, 2], [3, 2, 3]]
# traps = [2]

roads = [[1, 2, 1], [3, 2, 1], [2, 4, 1]]
traps = [2, 3]

time = 0

def goto(my, road):
    my = road[1]
    time += road[2]
    print(my)
    return my


def find_road(my):
    uses = [v for v in roads if my == v[0] or my == v[1]]
    if my in traps:
        for use in uses:
            uses.append([use[1], use[0], use[2]])
            uses = uses[1:]
    if len(uses) < 2 : result = uses[0]
    else:
        for use in uses:
            if use[0] == my:
                for z in uses:
                    if use[1] > z[1] : result = use
    
    print(result)
    return result

my = 1
while True:
    road = find_road(my)
    print(road)
    my = goto(my, road)
    print(my)
    if my == end : break
print(time)
