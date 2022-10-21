from random import randint
from puzzles import Puzzle24


puzzles = []
for i in range(1000):
    t = Puzzle24.Puzzle(shuffle=False)
    t.distCheck()
    t.findIndex()
    puzzles.append(t)

x = Puzzle24.Puzzle(shuffle=False)
x.distCheck()
x.findIndex()

op = 2

for j in puzzles:
    for i in range(randint(20,30)):
        moved = False
        while moved == False:
            r = randint(0,1)
            if r == 0:
                if j.up():
                    moved = True
                    op = 2
            elif r == 1:
                if j.down():
                    moved = True

            r = randint(0, 1)
            if r == 0:
                if j.left(): 
                    moved = True
            elif r == 1:
                if j.right():
                    moved = True

for i in puzzles:
    print(i.puzzle)
