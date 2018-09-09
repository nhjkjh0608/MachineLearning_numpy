import matplotlib.pyplot as plt

a = [1,2,3,4,5,6,20]
plt.boxplot(a,0,vert=0)
plt.savefig('aa.png')
plt.gcf().clear()
b = [1,2,3,4,5,6,6,6,6,6,6,20,30,40]
plt.boxplot(b, 0, vert = 0)
plt.savefig('aa2.png')
