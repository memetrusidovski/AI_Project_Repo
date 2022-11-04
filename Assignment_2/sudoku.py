import numpy as np
import math
from copy import copy, deepcopy
from queue import PriorityQueue
from func import *
from queue import Queue

board = [
    [7, 8, 0, 4, 0, 0, 1, 2, 0],
    [6, 0, 0, 0, 7, 5, 0, 0, 9],
    [0, 0, 0, 6, 0, 1, 0, 7, 8],
    [0, 0, 7, 0, 4, 0, 2, 6, 0],
    [0, 0, 1, 0, 5, 0, 9, 3, 0],
    [9, 0, 4, 0, 6, 0, 0, 0, 5],
    [0, 7, 0, 3, 0, 0, 0, 1, 2],
    [1, 2, 0, 0, 0, 7, 4, 0, 0],
    [0, 4, 9, 2, 0, 6, 0, 0, 7]
]
arc = Queue()
domain = {}

# Variables - All zero's, Constraints - Rules of Game, Domains - All possible scenario's

#print_board(board)
createDomain(board, domain)
#printDomain(domain)
createArcQueue(domain, arc)
r = arc.qsize()



AC3(arc, domain, board)
printDomain(domain)


#printArc(arc)