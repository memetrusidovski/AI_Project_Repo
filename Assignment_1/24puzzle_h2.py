from puzzles import Puzzle24
from copy import deepcopy
from queue import PriorityQueue
'''
h1 = the number of misplaced tiles. For Figure 3.28, 
all of the eight tiles are out of position, so the 
start state would have h1 = 8. h1 is an admissible 
heuristic because it is clear that any tile that is 
out of place must be moved at least once.
'''
"""
x =[]
for i in range(100):
    x.append(Puzzle8.Puzzle())

print(x[0])
"""


q = PriorityQueue()
explored = {''}
cost = 0
y = Puzzle24.Puzzle()

x = y

# [1,2,3,4,5,6,7,8,9,10,11,12,0,13,14,15]
# [2,7,1,13,14,5,15,10,11,3,0,4,12,9,6,8]
# [2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
if False:
    x.puzzle = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 0, 23, 24]
    x.distCheck()
    x.findIndex()

explored.add(str(x.puzzle))

while x._dist != 0 and cost < 2000000:
    up = deepcopy(x)
    down = deepcopy(x)
    left = deepcopy(x)
    right = deepcopy(x)

    x1 = up.up()
    x2 = down.down()
    x3 = left.left()
    x4 = right.right()

    if x1 and str(up.puzzle) not in explored:
        q.put(up)
        explored.add(str(up.puzzle))
        up.parent_node = x

    if x2 and str(down.puzzle) not in explored:
        q.put(down)
        explored.add(str(down.puzzle))
        down.parent_node = x

    if x3 and str(left.puzzle) not in explored:
        q.put(left)
        explored.add(str(left.puzzle))
        left.parent_node = x

    if x4 and str(right.puzzle) not in explored:
        q.put(right)
        explored.add(str(right.puzzle))
        right.parent_node = x

    x = q.get()
    x._globalCost += 1
    print(cost)

    cost += 1


temp = x
lst = []
while temp.parent_node != None:
    lst.append(temp)
    temp = temp.parent_node

for i in lst:
    print(i)
print(x)
