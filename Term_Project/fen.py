import chess
#then you make board
board = chess.Board()
#then you have list of moves with following line
moves = list(board.legal_moves)

print(moves)