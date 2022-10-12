from puzzles import Puzzle8

'''
 h2 = the sum of the distances of the tiles from 
their goal positions. Because tiles cannot move 
along diagonals, the distance we will count is 
the sum of the horizontal and vertical distances. 
 This is sometimes called the city block distance 
or Manhattan distance. h is also admissible because 
all any move can do is move one tile one step 2 
closer to the goal. Tiles 1 to 8 in the start state 
give a Manhattan distance of

h2 =3+1+2+2+2+3+3+2=18.


As expected, neither of these overestimates the true 
solution cost, which is 26.
'''

x = Puzzle8.Puzzle()
print(x)
