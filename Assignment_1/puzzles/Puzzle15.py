import math
from math import sqrt
import random
from typing import List
import numpy as np

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

    def __init__(self, size=4, shuffle=True, manhat=False, ecd=False):
        self.size = size
        self.puzzle: List[int] = []  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
        self.createPuz(size)
        self._index = 16
        self._dist = 0
        
        self._globalCost = 0
        self.parent_node = None
        self._manhat = manhat
        self._ecd = ecd

        if(shuffle):
            self.scramble()
            self.distCheck()

    def createPuz(self, size):
        for x in range(1, size*size):
            self.puzzle.append(x)
        self.puzzle.append(0)

    def __str__(self):
        return "_____________________\n| {0:2d} | {1:2d} | {2:2d} | {3:2d} |\n" \
            "| {4:2d} | {5:2d} | {6:2d} | {7:2d} |\n| {8:2d} | {9:2d} | {10:2d} | {11:2d} |\n" \
            "| {12:2d} | {13:2d} | {14:2d} | {15:2d} |\n~~~~~~~~~~~~~~~~~~~~~".format(
                *self.puzzle)

    def findIndex(self):
        i = 0
        for x in range(16):
            if self.puzzle[x] == 0:
                i = x
                self._index = i
            #print(self.puzzle[x], "{\}", end="")

        return i

    def scramble(self):
        random.shuffle(self.puzzle)
        self.findIndex()

    def distCheck(self):
        dist = 0
        if self._manhat:
            g1 = np.asarray(self.puzzle).reshape(4, 4)
            g2 = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 0]).reshape(4, 4)

            for i in range(15):
                a, b = np.where(g1 == i+1)
                x, y = np.where(g2 == i+1)
                dist += abs((a-x)[0])+abs((b-y)[0])

        if self._ecd:
            g1 = np.asarray(self.puzzle).reshape(4, 4)
            g2 = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 0]).reshape(4, 4)

            for i in range(15):
                a, b = np.where(g1 == i+1)
                x, y = np.where(g2 == i+1)
                dist += sqrt((abs((a-x)[0]) ** 2) + (abs((b-y)[0]) ** 2))
        else:
            for i, j in zip(self.puzzle, range(16)):
                if i != (j + 1) and (i != 0):
                    dist += 1

        self._dist = dist
        return dist

    def up(self):
        if(self.puzzle[15] == 0 or self.puzzle[14] == 0 or self.puzzle[13] == 0 or self.puzzle[12] == 0):
            #print("in bottom: invalid")
            return False
        else:
            self.puzzle[self._index], self.puzzle[self._index +
                                                  4] = self.puzzle[self._index + 4], self.puzzle[self._index]
            self.distCheck()
            self.findIndex()
            #print(self._index,"......")
            return True

    def down(self):
        if(self.puzzle[0] == 0 or self.puzzle[1] == 0 or self.puzzle[2] == 0 or self.puzzle[3] == 0):
            #print("in top: invalid")
            return False
        else:
            self.puzzle[self._index], self.puzzle[self._index -
                                                  4] = self.puzzle[self._index - 4], self.puzzle[self._index]
            self.distCheck()
            self.findIndex()
            return True

    def right(self):
        if (self._index != 0 and self._index != 4 and self._index != 8 and self._index != 12):
            #swap the index to the left
            self.puzzle[self._index], self.puzzle[self._index -
                                                  1] = self.puzzle[self._index - 1], self.puzzle[self._index]
            self.distCheck()
            self.findIndex()
            return True
        else:
            #print("Invalid Move")
            return False

    def left(self):
        if (self._index != 3 and self._index != 7 and self._index != 11 and self._index != 15):
            #swap the index to the right
            self.puzzle[self._index], self.puzzle[self._index +
                                                  1] = self.puzzle[self._index + 1], self.puzzle[self._index]
            self.distCheck()
            self.findIndex()
            return True
        else:
            #print("Invalid Move")
            return False

    def __iter__(self):
        for v in self.puzzle:
            yield v

    def __lt__(self, obj):
        return (self._dist + self._globalCost) < (obj._dist + obj._globalCost)

    """def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result"""


"""
#Testing
x = Puzzle()


print(x.findIndex())

print(x)
x.down()
print(x)

"""
