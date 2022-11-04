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

pzl = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 19, 20, 21, 22, 23, 24, 18],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 19, 20, 21, 22, 23, 24, 18],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 19, 20, 21, 22, 23, 24, 18],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 20, 21, 0, 22, 23],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 22, 0, 20, 24, 21, 18, 17, 23, 19],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 14, 23, 15, 16, 13, 24, 19, 20, 21, 22, 18, 0, 12],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 22, 17, 20, 0, 24, 21, 18, 23, 19],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 24, 16, 21, 22, 0, 23],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 16, 17, 19, 20, 15, 21, 22, 18, 23, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 0, 16, 17, 18, 14, 20, 21, 22, 23, 19, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 0, 16, 17, 13, 18, 20, 21, 22, 23, 19, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 19, 20, 21, 22, 23, 24, 18],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 12, 13, 0, 21, 16, 14, 20, 15, 22, 23, 18, 19, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 19, 20, 21, 22, 23, 24, 18],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 19, 20, 21, 22, 23, 24, 18],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 10, 0, 11, 12, 13, 9, 15, 16, 17, 18, 14, 20, 21, 22, 23, 19, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 19, 20, 21, 22, 23, 24, 18],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 0, 23, 24, 22],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 10],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 20, 16, 17, 18, 14, 0, 21, 22, 23, 19, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 0, 23, 24, 22],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 17, 18, 22, 20, 15, 16, 21, 23, 19, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 19, 20, 21, 22, 23, 24, 18],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 19, 20, 21, 22, 23, 24, 18],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 19, 20, 21, 22, 23, 24, 18],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 19, 20, 21, 22, 23, 24, 18],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 0, 16, 17, 18, 14, 20, 21, 22, 23, 19, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 10],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 0, 16, 17, 18, 14, 20, 21, 22, 23, 19, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 12, 13, 14, 10, 16, 17, 19, 24, 15, 20, 21, 18, 22, 23],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 19, 20, 21, 22, 23, 24, 18],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 23, 20, 21, 22, 19, 0],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 0, 6, 7, 8, 9, 5, 11, 12, 13, 14, 10, 16, 17, 19, 20, 15, 21, 22, 18, 23, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 0, 23, 24, 22],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 0, 23, 24, 22],
[1, 2, 3, 4, 0, 6, 7, 8, 9, 5, 11, 12, 13, 15, 10, 16, 17, 18, 14, 20, 21, 22, 23, 19, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 8, 2, 9, 5, 6, 4, 12, 3, 10, 11, 17, 7, 14, 15, 21, 16, 13, 19, 20, 22, 0, 18, 23, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 0, 6, 7, 8, 9, 5, 11, 13, 14, 15, 10, 16, 12, 17, 18, 20, 21, 22, 23, 19, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 0, 6, 7, 9, 10, 5, 11, 12, 8, 14, 15, 16, 17, 13, 18, 20, 21, 22, 23, 19, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 10],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 21, 22, 23, 24, 20],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 24],
]

def cpy(obj):
    t = Puzzle24.Puzzle(shuffle=False, manhat=True)
    t.puzzle = copy(obj.puzzle)
    t._dist = (obj._dist)
    t._globalCost =  obj._globalCost
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
        a1.append(u )
        a2.append(cost)
        a3.append(x._globalCost)
        costList.append(cost)
    else:
        costList.append("too long")
        a3.append("too long")

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

