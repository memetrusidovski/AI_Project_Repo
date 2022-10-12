from puzzles import Puzzle8

'''
h1 = the number of misplaced tiles. For Figure 3.28, 
all of the eight tiles are out of position, so the 
start state would have h1 = 8. h1 is an admissible 
heuristic because it is clear that any tile that is 
out of place must be moved at least once.
'''
x =[]
for i in range(100):
    x.append(Puzzle8.Puzzle())

print(x[0])



y = Puzzle8.Puzzle()

print(y.isSolved())
