from puzzles import Puzzle8
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
y = Puzzle8.Puzzle()
z = Puzzle8.Puzzle(shuffle=False)

x = y

while x._dist != 0:
    up = deepcopy(x)
    down = deepcopy(x)
    left = deepcopy(x)
    right = deepcopy(x)


    x1 = up.up()
    x2 = down.down()
    x3 = left.left()
    x4 = right.right()

    print(explored)

    up._globalCost = cost
    down._globalCost = cost
    left._globalCost = cost
    right._globalCost = cost

    if x1 and up not in explored:
        q.put(up)
        explored.append(up)
        up.parent_node = x
    if x2 and down not in explored:
        q.put(down)
        down.parent_node = x
    if x3 and left not in explored:
        q.put(left)
        left.parent_node = x
    if x4 and right not in explored:
        q.put(right)
        right.parent_node = x

    x = q.get()
    cost += 1



print(x, "   \n\n", x._globalCost)

while not q.empty():
    next_item = q.get()
    print(next_item._dist + next_item._globalCost, "\n~~~~~~~~~\n")



"""
nodes = [x,y,z]

print(x<z)
nodes.sort()

for i in nodes:
    print(i._dist)


q.put(x)
q.put(y)
q.put(z)

while not q.empty():
    next_item = q.get()
    print(next_item._dist)

first try, didnt work
def search(puz, prev, cost): 
    up = deepcopy(puz)
    down = deepcopy(puz)
    left = deepcopy(puz)
    right = deepcopy(puz)

    x1 = up.up()
    x2 = down.down()
    x3 = left.left()
    x4 = right.right()

    
    m = [up if x1 else z,  down if x2 else z,
         left if x3 else z,  right if x3 else z]
    

    #Swap sort to bring the best move to the front
    if m[3]._dist < m[2]._dist and m[3].puzzle != prev.puzzle:
        m[3], m[2] = m[2], m[3]

    if m[2]._dist < m[1]._dist and m[2].puzzle != prev.puzzle:
        m[2], m[1] = m[1], m[2]

    if m[1]._dist < m[0]._dist and m[1].puzzle != prev.puzzle:
        m[1], m[0] = m[0], m[1]
    
    if m[0] == prev:
        return True
        
    

    for i in m: 
        print(i)
    print("\n\n")

  
    if m[0]._dist != 0:
        search(m[0], puz)
    else:
        return True
    



search(y, z, 0)
    """

