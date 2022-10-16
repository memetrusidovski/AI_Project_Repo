import math
import random
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

    def __init__(self, size=5, shuffle=True, manhat=False):
        self.size = size
        self.puzzle = [] 
        self.createPuz(size)
        self._index = 25
        self._dist = 0
        self._solved = False
        self._globalCost = 0
        self.parent_node = None
        self._manhat = manhat

        if(shuffle):
            self.scramble()
            self.distCheck()

    def createPuz(self, size):
        for x in range(1, size*size):
            self.puzzle.append(x)
        self.puzzle.append(0)

    def __str__(self):
        return "__________________________\n| {0:2d} | {1:2d} | {2:2d} | {3:2d} | {4:2d} |\n" \
            "| {5:2d} | {6:2d} | {7:2d} | {8:2d} | {9:2d} |\n| {10:2d} | {11:2d} | {12:2d} | {13:2d} | {14:2d} |\n" \
            "| {15:2d} | {16:2d} | {17:2d} | {18:2d} | {19:2d} |\n" \
            "| {20:2d} | {21:2d} | {22:2d} | {23:2d} | {24:2d} |\n~~~~~~~~~~~~~~~~~~~~~~~~~~".format(
                *self.puzzle)

    def findIndex(self):
        i = 0
        for x in range(25):
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
            g1 = np.asarray(self.puzzle).reshape(5, 5)
            g2 = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 16,17,18,19,20,21,22,23,24, 0]).reshape(5, 5)

            for i in range(24):
                a, b = np.where(g1 == i+1)
                x, y = np.where(g2 == i+1)
                dist += abs((a-x)[0])+abs((b-y)[0])

        else:
            for i, j in zip(self.puzzle, range(24)):
                if i != (j + 1) and (i != 0):
                    dist += 1

        self._dist = dist
        return dist

    def up(self):
        if(0 in self.puzzle[((self.size ** 2)-self.size):]):
            #print("in bottom: invalid")
            return False
        else:
            self.puzzle[self._index], self.puzzle[self._index +
                                                  5] = self.puzzle[self._index + 5], self.puzzle[self._index]
            self.distCheck()
            self.findIndex()
            #print(self._index,"......")
            return True

    def down(self):
        if(0 in self.puzzle[0:self.size]):
            #print("in top: invalid")
            return False
        else:
            self.puzzle[self._index], self.puzzle[self._index -
                                                  5] = self.puzzle[self._index - 5], self.puzzle[self._index]
            self.distCheck()
            self.findIndex()
            return True

    def right(self):
        if (self._index != 0 and self._index != 4 and self._index != 9 and self._index != 14 and self._index != 19):
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
        if (self._index != 4 and self._index != 9 and self._index != 14 and self._index != 19 and self._index != 24):
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



#Testing
x = Puzzle()


print(x.findIndex())

print(x)
x.down()
x.up()
x.left()
x.right()
print(x)


