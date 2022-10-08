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
    
    def __init__(self, size=3, shuffle=True):
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
        return "_____________\n| {0} | {1} | {2} |\n" \
            "| {3} | {4} | {5} |\n| {6} | {7} | {8} |\n~~~~~~~~~~~~~".format(*self.puzzle)
        


    def scramble(self):
        random.shuffle(self.puzzle)

    def up(self):
        if(0 in self.puzzle[0:self.size]):
            print("in top")
            

    def down(self):
        if(0 in self.puzzle[((self.size ** 2)-self.size):] ):
            print("in bottom")

    def right(self):
        pass

    def left(self):
        pass




x = Puzzle()
print(x)
x.up()
x.down()
