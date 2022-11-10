import ast
from func import *
from queue import Queue

"""
------------------------------------------------------------------------------------------
File:    sudoku.py
Project: AI_Project_Repo
Purpose: 
==========================================================================================

Program Description:
  This program solves a sudoku puzzle giving in the format
  of a 2d array. The program reads the first puzzle from the 
  txt file and attempts to solve it.
------------------------------------------------------------------------------------------
Group:   14
Email:   rusi1550@mylaurier.ca
Version  2022-11-09
-------------------------------------
"""

#Default Sudoku Board
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


with open('/Users/schoolaccount/Documents/GitHub/AI_Project_Repo/Assignment_2/sudoku.txt') as f:
    temp = []
    x = f.readlines()
    for lines in x:
        if lines != '\n' and lines[0] != '#':
            w = ast.literal_eval(lines)
            w = [int(x) for x in w]
            temp.append(w)
        else:
            break

    print(print_board(temp))
    board = temp

         
# Variables - All zero's, Constraints - Rules of Game, Domains - All possible scenario's
arc = Queue()
domain = {}

# Populate domain and arc queue
createDomain(board, domain)
#printDomain(domain)
createArcQueue(domain, arc)


AC3(arc, domain, board)

# If the queue is empty the puzzle is solved 
if arc.qsize() == 0:
    print_board(board)
    print("Arc Queue Size: ",arc.qsize())
    print("SOLVED")
else:
    # Finish solving the board 
    backtrack(board, 0, 0)
    printDomain(domain)

    print_board(board)

    print("WAS NOT SOLVED - Backtracking...")

#printArc(arc)
printDomain(domain)