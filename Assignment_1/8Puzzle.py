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
        self.puzzle = []  # [1, 2, 3, 4, 5, 6, 7, 8, 0]
        self.createPuz(size)
        self._index = 8
        if(shuffle):
            self.scramble()

    def createPuz(self, size):
        for x in range(1, size*size):
            self.puzzle.append(x)
        self.puzzle.append(0)

    def __str__(self):
        return "_____________\n| {0} | {1} | {2} |\n" \
            "| {3} | {4} | {5} |\n| {6} | {7} | {8} |\n~~~~~~~~~~~~~".format(
                *self.puzzle)

    def findIndex(self):
        i = 0
        for x in range(8):
            if self.puzzle[x] == 0:
                i = x
                self._index = i

        return i

    def scramble(self):
        random.shuffle(self.puzzle)
        self.findIndex()

    def up(self):
        if(0 in self.puzzle[((self.size ** 2)-self.size):]):
            print("in bottom: invalid")
        else:
            self.puzzle[self._index], self.puzzle[self._index +
                                                  3] = self.puzzle[self._index + 3], self.puzzle[self._index]

    def down(self):
        if(0 in self.puzzle[0:self.size]):
            print("in top: invalid")
        else:
            self.puzzle[self._index], self.puzzle[self._index -
                                                  3] = self.puzzle[self._index - 3], self.puzzle[self._index]

    def right(self):
        if (self._index != 0 and self._index != 3 and self._index != 6):
            #swap the index to the left 
            self.puzzle[self._index], self.puzzle[self._index -
                                                  1] = self.puzzle[self._index - 1], self.puzzle[self._index]
        else:
            print("Invalid Move")

    def left(self):
        if (self._index != 2 and self._index != 5 and self._index != 8):
            #swap the index to the right
            self.puzzle[self._index], self.puzzle[self._index +
                                                  1] = self.puzzle[self._index + 1], self.puzzle[self._index]
        else:
            print("Invalid Move")


x = Puzzle()


print(x.findIndex())

print(x)
x.down()
print(x)
