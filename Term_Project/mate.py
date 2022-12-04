import moveGen
#then you make board
board = moveGen.Board(
    'r5rk/5p1p/5R2/4B3/8/8/7P/7K w KQ - 1 26')
#then you have list of moves with following line


moves = list(board.legal_moves)
#board.push_san('g1h3')

for i in moves:
    print(i)
print(board.fen())
print(board.is_checkmate())
bP = 0b0000_0000_0000_0000_1111_1111_0000_0000

#x |= 0b0000_0001_1000_0001_1000_0001_1000_0001
y = 0b0000_0000_0000_0000_1111_1111_0000_0000
x = bP | y


brds = []
brds2 = []
brds3 = []


for j in moves:
    temp = board.copy()
    temp.push_san(str(j))
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
