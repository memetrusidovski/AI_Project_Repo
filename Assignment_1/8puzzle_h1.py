from puzzles import Puzzle24
from copy import copy, deepcopy
from queue import PriorityQueue
'''
h1 = the number of misplaced tiles. For Figure 3.28, 
all of the eight tiles are out of position, so the 
start state would have h1 = 8. h1 is an admissible 
heuristic because it is clear that any tile that is 
out of place must be moved at least once.
'''
"""
puzzles =[]
for i in range(100):
    puzzles.append(Puzzle8.Puzzle())
"""

pzl = [
[1, 2, 3, 4, 5, 6, 7, 9, 14, 0, 11, 13, 8, 15, 10, 16, 12, 17, 19, 23, 21, 22, 18, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 12, 14, 19, 10, 16, 18, 23, 22, 15, 20, 21, 17, 13, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 24, 15, 22, 17, 20, 0, 16, 21, 18, 23, 19],
[1, 2, 3, 4, 5, 6, 7, 9, 14, 10, 11, 16, 17, 8, 15, 21, 12, 18, 23, 24, 20, 22, 13, 0, 19],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 13, 14, 15, 21, 11, 22, 18, 19, 20, 17, 23, 0, 24],
[1, 8, 4, 9, 5, 7, 3, 13, 2, 10, 6, 11, 14, 0, 15, 17, 21, 18, 12, 24, 16, 22, 23, 20, 19],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 24, 15, 17, 18, 20, 19, 16, 21, 22, 23, 0],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 21, 17, 14, 0, 22, 11, 12, 18, 15, 20, 16, 23, 19, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 20, 16, 17, 14, 18, 24, 21, 22, 0, 19, 23],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 21, 18, 19, 0, 20, 16, 22, 23, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 14, 22, 15, 21, 11, 13, 24, 23, 20, 18, 17, 0, 19],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 11, 12, 17, 0, 13, 23, 18, 20, 15, 21, 22, 19, 14, 24],
[1, 2, 3, 4, 0, 6, 7, 8, 9, 5, 11, 12, 13, 14, 10, 16, 23, 22, 18, 15, 20, 21, 17, 19, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 20, 21, 16, 18, 19, 24, 17, 0, 22, 14, 23],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 21, 17, 13, 15, 22, 11, 14, 20, 24, 12, 16, 0, 18, 23],
[1, 2, 3, 4, 0, 6, 7, 8, 9, 5, 11, 12, 13, 14, 10, 17, 21, 23, 24, 15, 20, 16, 22, 18, 19],
[1, 7, 2, 3, 5, 6, 9, 14, 8, 10, 11, 13, 4, 24, 0, 16, 12, 17, 19, 15, 21, 22, 18, 23, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 16, 22, 17, 19, 15, 20, 21, 18, 23, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 15, 0, 16, 17, 14, 13, 24, 21, 22, 18, 23, 19],
[1, 2, 3, 4, 5, 6, 12, 7, 9, 10, 16, 11, 8, 13, 15, 21, 17, 19, 14, 24, 20, 22, 18, 0, 23],
[1, 2, 3, 4, 5, 11, 6, 14, 8, 10, 17, 16, 9, 18, 0, 7, 13, 23, 20, 15, 21, 22, 19, 12, 24],
[1, 2, 3, 4, 0, 6, 7, 8, 9, 5, 11, 12, 13, 14, 10, 22, 21, 18, 24, 15, 20, 16, 19, 17, 23],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 21, 16, 23, 24, 15, 20, 17, 22, 18, 19],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 17, 12, 14, 10, 21, 16, 13, 24, 15, 20, 22, 23, 18, 19],
[1, 2, 3, 4, 5, 11, 6, 8, 9, 0, 7, 12, 13, 14, 10, 22, 21, 17, 18, 15, 20, 16, 23, 19, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 12, 13, 14, 10, 15, 22, 17, 20, 24, 16, 21, 18, 23, 19],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 19, 13, 15, 16, 22, 17, 23, 0, 20, 21, 18, 24, 14],
[1, 2, 3, 4, 5, 7, 11, 8, 9, 10, 6, 16, 22, 13, 15, 20, 21, 12, 18, 0, 24, 23, 17, 14, 19],
[1, 2, 3, 4, 5, 6, 7, 9, 13, 10, 11, 12, 19, 8, 15, 16, 18, 22, 14, 0, 20, 21, 17, 23, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 12, 14, 18, 10, 16, 22, 17, 13, 15, 20, 21, 23, 19, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 18, 17, 24, 0, 20, 16, 23, 22, 19],
[1, 2, 3, 4, 5, 6, 7, 9, 13, 10, 11, 19, 8, 15, 20, 22, 12, 0, 17, 24, 16, 21, 18, 14, 23],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 21, 0, 19, 24, 20, 16, 22, 23, 18],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17, 15, 0, 16, 12, 19, 14, 20, 21, 22, 18, 23, 24],
]


def cpy(obj):
    t = Puzzle24.Puzzle(shuffle=False, manhat=True)
    t.puzzle = copy(obj.puzzle)
    t._dist = (obj._dist)
    t._globalCost = obj._globalCost
    t._index = obj._index
    return t


costList = []

a1 = []
a2 = []
a3 = []

for u in pzl:
    q = PriorityQueue()
    explored = {""}
    cost = 0
    y = Puzzle24.Puzzle()
    z = Puzzle24.Puzzle(shuffle=False, manhat=True)

    x = y

    if True:
        x.puzzle = u

        x.distCheck()
        x.findIndex()

    explored.add(str(x.puzzle))

    while x._dist != 0 and cost <= 800000:
        up = cpy(x)
        down = cpy(x)
        left = cpy(x)
        right = cpy(x)

        x1 = up.up()
        x2 = down.down()
        x3 = left.left()
        x4 = right.right()

        if x1 and str(up.puzzle) not in explored:
            q.put(up)
            explored.add(str(up.puzzle))
            up.parent_node = x
            #print(up, up.puzzle, "up", up._index, "index", "\n\n")
        if x2 and str(down.puzzle) not in explored:
            q.put(down)
            explored.add(str(down.puzzle))
            down.parent_node = x
            #print(down, down.puzzle, "down", down._index, "index", "\n\n\n")
        if x3 and str(left.puzzle) not in explored:
            q.put(left)
            explored.add(str(left.puzzle))
            left.parent_node = x
            #print(left, left.puzzle, "left", left._index, "index", "\n\n\n")
        if x4 and str(right.puzzle) not in explored:
            q.put(right)
            explored.add(str(right.puzzle))
            right.parent_node = x
            #print(right, right.puzzle, "right", right._index, "index", "\n\n")

        x = q.get()
        x._globalCost += 1

        if cost % 800000 == 0 and cost != 0:
            print("too long><")
        #print(x._dist, " -------", x._globalCost)
        cost += 1

    """temp = x
    lst = []
    while temp.parent_node != None:
        lst.append(temp)
        temp = temp.parent_node

    for i in lst:
        print(i)"""

    if cost != 800001 and cost > 2000:
        print(u, "\t", cost, "\t", x._globalCost)
        a1.append(u)
        a2.append(cost)
        a3.append(x._globalCost)
        costList.append(cost)
    else:
        costList.append("too long")

print("\n\n\n\n<><><>\n")
for f in costList:
    print(f)

print("\n\n\n\n<><><>\n")
for f in a1:
    print(f)

print("\n\n\n\n<><><>\n")
for f in a2:
    print(f)

print("\n\n\n\n<><><>\n")
for f in a3:
    print(f)
