from puzzles import Puzzle8
from copy import deepcopy
from queue import PriorityQueue
'''
 h2 = the sum of the distances of the tiles from 
their goal positions. Because tiles cannot move 
along diagonals, the distance we will count is 
the sum of the horizontal and vertical distances. 
 This is sometimes called the city block distance 
or Manhattan distance. h is also admissible because 
all any move can do is move one tile one step 2 
closer to the goal. Tiles 1 to 8 in the start state 
give a Manhattan distance of

h2 =3+1+2+2+2+3+3+2=18.


As expected, neither of these overestimates the true 
solution cost, which is 26.
'''

q = PriorityQueue()
explored = []
cost = 0
y = Puzzle8.Puzzle(manhat=True)


x = y
if True:
    x.puzzle = [8,6,7,2,5,4,3,0,1]#[5,1,4,6,3,8,0,7,2]#[3, 6, 2, 5, 0, 7, 4, 1, 8]
    x.distCheck()
    x.findIndex()

explored.append(x.puzzle)

while x._dist != 0 and cost < 20000:
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
        #print(up, up.puzzle, "up", up._index, "index", "\n\n")
    if x2 and down.puzzle not in explored:
        q.put(down)
        explored.append(down.puzzle)
        down.parent_node = x
        #print(down, down.puzzle, "down", down._index, "index", "\n\n\n")
    if x3 and left.puzzle not in explored:
        q.put(left)
        explored.append(left.puzzle)
        left.parent_node = x
        #print(left, left.puzzle, "left", left._index, "index", "\n\n\n")
    if x4 and right.puzzle not in explored:
        q.put(right)
        explored.append(right.puzzle)
        right.parent_node = x
        #print(right, right.puzzle, "right", right._index, "index", "\n\n")

    x = q.get()
    x._globalCost += 1
    cost += 1


temp = x
lst = []
while temp.parent_node != None:
    lst.append(temp)
    temp = temp.parent_node

for i in lst:
    print(i)

print(x._globalCost)
