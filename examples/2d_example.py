"""
This contains code that shows how the emulator
would handle input where the domain is multidimensional.
"""
import numpy as np
import emulator as Emulator
np.random.seed(0)

#Try emulating on some periodic data
N = 200
x = np.vstack((10*np.random.rand(N),10*np.random.rand(N)))
yerr = 0.2 * np.ones(N)
y = np.cos(x[0]) + np.cos(x[1]) + yerr*np.random.randn(N)
#y = np.sin(x[0]) + np.cos(x[1]) + yerr*np.random.randn(N)
#TODO: figure out why the emulator can't handle one odd and one even function...

#Declare an emulator, train it, and predict with it.
emu = Emulator.Emulator(name="2d",xdata=x.T,ydata=y,yerr=yerr)
emu.train()
print "Best parameters = ",emu.Ls,emu.k0

Nd = 50
out = 10
X0 = np.linspace(np.min(x)-out, np.max(x)+out, Nd)
X1 = np.linspace(np.min(x)-out, np.max(x)+out, Nd)
D0, D1 = np.meshgrid(X0, X1)
ystar = np.ones_like(D0)
for i in range(len(X0)):
    for j in range(len(X1)):
        Xij = np.atleast_2d([X0[i], X1[j]])
        ystarij, var = emu.predict(Xij)
        ystar[i,j] = ystarij

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[0], x[1], y, c='k')
ax.plot_wireframe(D0, D1, ystar, alpha=0.5)
fig.savefig("../figures/2d_example.png")
plt.show()
