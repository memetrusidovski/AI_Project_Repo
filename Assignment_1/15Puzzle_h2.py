from puzzles import Puzzle15
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
explored = []
cost = 0
y = Puzzle15.Puzzle(manhat=False)



x = y

# [1,2,3,4,5,6,7,8,9,10,11,12,0,13,14,15]
# [2,7,1,13,14,5,15,10,11,3,0,4,12,9,6,8]
# [2, 1, 3, 4 ,5 ,6 ,7 ,8 ,9, 10, 11, 12, 13, 14, 15,0]
#[1, 2, 3, 4, 5, 6, 7, 8, 0, 10, 11, 12, 9, 13, 14, 15]
x.puzzle = [2, 7, 1, 13, 14, 5, 15, 10, 11, 3, 0, 4, 12, 9, 6, 8]
x.distCheck()
x.findIndex()

explored.append(x.puzzle)

while x._dist != 0 and cost < 20000000:
    up = deepcopy(x)
    down = deepcopy(x)
    left = deepcopy(x)
    right = deepcopy(x)

    x1 = up.up()
    x2 = down.down()
    x3 = left.left()
    x4 = right.right()


    if x1 and up.puzzle not in explored:
        q.put(up)
        explored.append(up.puzzle)
        up.parent_node = x
        
    if x2 and down.puzzle not in explored:
        q.put(down)
        explored.append(down.puzzle)
        down.parent_node = x
        
    if x3 and left.puzzle not in explored:
        q.put(left)
        explored.append(left.puzzle)
        left.parent_node = x
        
    if x4 and right.puzzle not in explored:
        q.put(right)
        explored.append(right.puzzle)
        right.parent_node = x
        

    x = q.get()
    x._globalCost += 1
    print(cost)
    #print(x._dist, " -------", x._globalCost)
    cost += 1


temp = x
lst = []
while temp.parent_node != None:
    lst.append(temp)
    temp = temp.parent_node

for i in lst:
    print(i)
    #print(i._index)
print(x._globalCost)
