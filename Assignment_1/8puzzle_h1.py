from puzzles import Puzzle8
from copy import deepcopy
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




x = Puzzle8.Puzzle()
y = Puzzle8.Puzzle()
z = Puzzle8.Puzzle()
z._dist = 10
moves = []

def search(puz, lst, prev): 
    up = deepcopy(puz)
    down = deepcopy(puz)
    left = deepcopy(puz)
    right = deepcopy(puz)

    print(up)
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
        m[1], m[0] = m[0], m[1]

    print(puz)
    print(m[0])
    
    if m[0]._dist != 0:
        search(m[0], moves.append(deepcopy(m[0])), puz)
    else:
        return True


search(y, moves, z)

for i in moves:
    print(i)


#print(y.up())


#for i in y:
#    print(i)
