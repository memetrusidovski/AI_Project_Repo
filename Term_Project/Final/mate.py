import moveGen

#then you make board
board = moveGen.Board('8/8/6p1/7k/3r2NP/B5PK/2br1R2/8 w - - 0 1')
    #'r5rk/5p1p/5R2/4B3/8/8/7P/7K w KQ - 1 26')
#then you have list of moves with following line


moves = list(board.legal_moves)
print(board.unicode())
#board.push_san('g1h3')
'''
for i in moves:
    print(i)
print(board.fen())
print(board.is_checkmate())
moves.count
'''

moved = set()
def search(b, m, d, who):
    #Search too deep
    if d > 5:
        return False

    #Create the new node
    temp = b.copy()

    #Do a move from the list
    s = str(m.pop())
    
    temp.push_san(s)

    if temp.is_checkmate() and who == 0:
        print(temp, "<--\n")
        return temp
    elif len(m) > 0:
        #print(temp)
        search(b, m, d, who)

    
    moves = list(temp.legal_moves)
    
    if temp.is_check() and who == 0: 
        #print(moves)
        search(temp, moves, d + 1, 1)
    elif who == 1:
        search(temp, moves, d + 1, 0)

    return False


search(board, list(board.legal_moves), 0, 0)


 
'''brds = []
brds2 = []
brds3 = []


for j in moves:
    temp = board.copy()
    temp.push_san(str(j))
    if temp.is_check():
        brds.append(temp.fen())

for i in brds:
    board = moveGen.Board(i)
    moves = list(board.legal_moves)
    for j in moves:
        temp = board.copy()
        temp.push_san(str(j))
        brds2.append(temp.fen())

for i in brds2:
    board = moveGen.Board(i)
    moves = list(board.legal_moves)
    for j in moves:
        temp = board.copy()
        temp.push_san(str(j))
        if temp.is_checkmate():
            print("YES")
            brds3.append(temp.fen())

for i in brds3:
    print(i)
#print(f"{bP:b}")

print(moveGen.Board('r5rk/5p2/7R/4B2p/7P/8/8/7K b - - 1 27').unicode())
 
'''
