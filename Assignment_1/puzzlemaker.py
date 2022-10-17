from random import randint
from puzzles import Puzzle15


puzzles = []
for i in range(50):
    t = Puzzle15.Puzzle(shuffle=False)
    t.distCheck()
    t.findIndex()
    puzzles.append(t)

x = Puzzle15.Puzzle(shuffle=False)
x.distCheck()
x.findIndex()

for j in puzzles:
    for i in range(100):
        r = randint(0,3)
        if r == 0:
            j.up()
        elif r == 1:
            j.down()
        elif r == 2:
            j.left()
        elif r == 3:
            j.right()

for i in puzzles:
    print(i.puzzle)
