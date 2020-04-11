import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

x = [1, 2, 3]
y = [1, 10, 1]

x2 = [3, 4, 5]
y2 = [1, 10, 1]

x3 = [5, 6, 7]
y3 = [1, 10, 1]

x4=[1,2,3,4,5,6,7]
y4=[1,1,1,1,1,1,1]

y5=[10,10,10,10,10,10,10]

# can plot specifically, after just showing the defaults:
plt.plot(x, y, linewidth=5)
plt.plot(x2, y2, linewidth=5)
plt.plot(x3, y3, linewidth=5)
plt.plot(x4, y4, linewidth=5)
plt.plot(x4, y5, linewidth=5)

plt.title('Triangle schedule')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.savefig('zigzag.png')
plt.show()
