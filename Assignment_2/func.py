from copy import deepcopy
from queue import Queue

abc = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}

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
    c = 0
    #for i in range(arc.qsize()):
    while not arc.empty() and c < 10000:
        revise(arc, domain, board)
        c += 1

    return B


def revise(arc, domain, board):
    B = True
    temp = arc.get()

    print(arc.qsize())

    if board[temp[1][0]][temp[1][1]] in domain[temp[0]]:
        domain[temp[0]].remove(board[temp[1][0]][temp[1][1]])
        #Add Changed Arcs
        if len(domain[temp[0]]) > 1:
            pass#arc.put( temp )





    if len(domain[temp[0]]) == 0:
        B = False
    
    if len(domain[temp[0]]) == 1:
        s = temp[0]
        
        board[abc[s[0]]][int(s[1])] = domain[temp[0]][0]
        #print(board[abc[s[0]]][int(s[1])])
        #domain[temp[0]] = 'x'
    if len(domain[temp[0]]) >= 2:
        arc.put(temp)

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


def backtrack(grid, row, col):
    if (row == 8 and col == 9):
        return True

    if col == 9:
        row += 1
        col = 0

    if grid[row][col] > 0:
        return backtrack(grid, row, col + 1)

    for num in range(1, 10):
        if isSafe(grid, row, col, num):
            grid[row][col] = num

            if backtrack(grid, row, col + 1):
                return True

        grid[row][col] = 0
    return False


def isSafe(grid, row, col, num):

    for x in range(9):
        if grid[row][x] == num:
            return False

    for x in range(9):
        if grid[x][col] == num:
            return False

    startRow = row - row % 3
    startCol = col - col % 3

    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False

    return True

def printDomain(domain):
    for i in domain:
        print(i, "-",domain[i])

def printArc(arc):
    tmp = arc
    while not tmp.empty():
        print(tmp.get())
