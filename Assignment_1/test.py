import numpy as np

x = [3, 6, 2, 5, 0, 7, 4, 1, 8]
y = [1, 2, 3, 4, 5, 6, 7, 8, 0]
d = 0

g1 = np.asarray(x).reshape(3, 3)
g2 = np.asarray(y).reshape(3, 3)

for i in range(8):
    a, b = np.where(g1 == i+1)
    x, y = np.where(g2 == i+1)
    d = d + abs((a-x)[0])+abs((b-y)[0])
    
print(d)

for q in range(15):
    print(q)