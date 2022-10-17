from pickle import dumps, loads
from typing import List, Set
from puzzles import Puzzle15
from copy import deepcopy, copy
from queue import PriorityQueue
from heapq import heappop, heappush
import numpy as np
from pympler import asizeof
import threading
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
explored: Set[str] = {""}
cost = 0
y = Puzzle15.Puzzle()

h = []
a = heappush(h, y)
x = y

print(asizeof.asizeof(x))

# [1,2,3,4,5,6,7,8,9,10,11,12,0,13,14,15]
# [2,7,1,13,14,5,15,10,11,3,0,4,12,9,6,8]
# [2, 1, 3, 4 ,5 ,6 ,7 ,8 ,9, 10, 11, 12, 13, 14, 15,0]
#[1, 2, 3, 4, 5, 6, 7, 8, 0, 10, 11, 12, 9, 13, 14, 15]
if False:
    x.puzzle = [2, 7, 1, 13, 14, 5, 15, 10, 11, 3, 0, 4, 12, 9, 6, 8]
    x.distCheck()
    x.findIndex()

print(x.puzzle)
explored.add(str(x.puzzle))

up = None
down = None 
left = None 
right = None 


def cpyU(obj):
    print(obj)
    t = Puzzle15.Puzzle(shuffle=False)
    t.puzzle = loads(dumps(x.puzzle))
    t._dist = (x._dist)
    t._globalCost = (x._globalCost)
    t._index = (x._index)
    up = t
    return t


def cpyD(obj):
    print(obj)
    t = Puzzle15.Puzzle(shuffle=False)
    t.puzzle = loads(dumps(x.puzzle))
    t._dist = (x._dist)
    t._globalCost = (x._globalCost)
    t._index = (x._index)
    down = t
    return t


def cpyL(obj):
    print(obj)
    t = Puzzle15.Puzzle(shuffle=False)
    t.puzzle = loads(dumps(x.puzzle))
    t._dist = (x._dist)
    t._globalCost = (x._globalCost)
    t._index = (x._index)
    left = t
    return t


def cpyR(obj):
    print(obj)
    t = Puzzle15.Puzzle(shuffle=False)
    t.puzzle = loads(dumps(x.puzzle))
    t._dist = (x._dist)
    t._globalCost = (x._globalCost)
    t._index = (x._index)
    right = t
    return t

while x._dist != 0 and cost < 20000000:

    thr1 = threading.Thread(target=cpyU, args=(x,))
    thr2 = threading.Thread(target=cpyR, args=(x,))
    thr3 = threading.Thread(target=cpyL, args=(x,))
    thr4 = threading.Thread(target=cpyR, args=(x,))

    thr1.start()
    thr2.start()
    thr3.start()
    thr4.start()

    thr1.join()
    thr2.join()
    thr3.join()
    thr4.join()
    
    print(up)

    x1 = up.up()
    x2 = down.down()
    x3 = left.left()
    x4 = right.right()

    if x1 and str(up.puzzle) not in explored:
        #q.put(up)
        heappush(h, up)
        explored.add(str(up.puzzle))
        #up.parent_node = x
    else:
        del up

    if x2 and str(down.puzzle) not in explored:
        #q.put(down)
        heappush(h, down)
        explored.add(str(down.puzzle))
        #down.parent_node = x
    else:
        del down

    if x3 and str(left.puzzle) not in explored:
        #q.put(left)
        heappush(h, left)
        explored.add(str(left.puzzle))
        #left.parent_node = x
    else:
        del left 

    if x4 and str(right.puzzle) not in explored:
        #q.put(right)
        heappush(h, right)
        explored.add(str(right.puzzle))
        #right.parent_node = x
    else:
        del right

    #x = q.get()
    x = heappop(h)
    x._globalCost += 1
    if(cost % 1000 == 0):
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
