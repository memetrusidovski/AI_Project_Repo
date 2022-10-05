import math
import random

'''
Zero is the place holder for the empty square.
the matrix is divided equally so,
[1,2,3,4,5,6,7,8, 0] =>
    __________
    |1 | 2| 3|
    |4 | 5| 6|
    |7 | 8| 0|
    ~~~~~~~~~~
'''

class Puzzle:
    
    def __init__(self, size, shuffle=True):
        self.size = size    
        self.puzzle = []#[1, 2, 3, 4, 5, 6, 7, 8, 0]
        self.createPuz(size)
        if(shuffle):
            self.scramble()

    def createPuz(self, size):
        for x in range(1,size*size):
            self.puzzle.append(x)
        self.puzzle.append(0)

    def __str__(self):
        return "{}".format(self.puzzle)

    def scramble(self):
        random.shuffle(self.puzzle)




x = Puzzle(5)
print(x)