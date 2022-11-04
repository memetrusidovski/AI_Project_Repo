from copy import deepcopy
from queue import Queue


def print_board(b):
    for i in range(len(b)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")

        for j in range(len(b[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                print(b[i][j])
            else:
                print(str(b[i][j]) + " ", end="")

def createDomain(board, q):
    abc = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    for x, letter in zip(board, abc):
        count = 0
        for y in x:
            if y == 0:
                q[f"{letter}{count}"] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            else:
                q[f"{letter}{count}"] = "x"
            count += 1

def createArcQueue(domain, arc):
    abc = {'a': 0, 'b': 1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8}

    for i in domain:
        # Check for arcs only in empty squares
        if domain[i] != 'x':
            # Row Consistency
            row = abc[i[0]]
            for x in range(9):
                if x != int(i[1]):
                    arc.put([i, (row, x)])

            # Colum Consistency
            col = int(i[1])
            for y in range(9):
                if y != abc[i[0]]:
                    arc.put([i, (y, col)])

            # Box Consistency
            quadY = abc[i[0]] - abc[i[0]] % 3
            quadX = int(i[1]) - int(i[1]) % 3

            for y in range(3):
                for x in range(3):
                    if (quadX + x) != int(i[1]) or (quadY + y) != abc[i[0]]:
                        arc.put([i, (quadY + y, quadX + x)])

    return

def AC3(arc, domain, board):
    B = True
    for i in range(1032):
    #while not arc.empty():

        revise(arc, domain, board)

    return B


def revise(arc, domain, board):
    B = True
    temp = arc.get()

    print(arc.qsize())

    if board[temp[1][0]][temp[1][1]] in domain[temp[0]]:
        domain[temp[0]].remove(board[temp[1][0]][temp[1][1]])
        addArc(domain, arc)





    if len(domain[temp[0]]) == 0:
        B = False

    return B

def addArc(domain, arc):
    row = abc[i[0]]
    for x in range(9):
        if x != int(i[1]):
            arc.put([i, (row, x)])

    # Colum Consistency
    col = int(i[1])
    for y in range(9):
        if y != abc[i[0]]:
            arc.put([i, (y, col)])

    # Box Consistency
    quadY = abc[i[0]] - abc[i[0]] % 3
    quadX = int(i[1]) - int(i[1]) % 3

    for y in range(3):
        for x in range(3):
            if (quadX + x) != int(i[1]) or (quadY + y) != abc[i[0]]:
                arc.put([i, (quadY + y, quadX + x)])



def printDomain(domain):
    for i in domain:
        print(i, "-",domain[i])

def printArc(arc):
    tmp = arc
    while not tmp.empty():
        print(tmp.get())
