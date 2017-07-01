import emulator
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', size=20)

#Create random training data
np.random.seed(0)
Nx = 40
x = 10*np.sort(np.random.rand(Nx))
yerr = 0.2 * np.ones_like(x)
y = np.cos(x) + 1 + yerr * np.random.randn(len(x))

#Train
emu = emulator.Emulator(name="example_emulator",xdata=x,ydata=y,yerr=yerr)
emu.train()

#Create some data to predict on
xstar = np.linspace(min(x)-3,max(x)+3,500)
ystar,ystarvar = emu.predict(xstar)
ystarerr = np.sqrt(ystarvar)

#Plot the training data
plt.errorbar(x,y,np.fabs(yerr),ls='',marker='o',color='k',ms=8,label="f")
#Plot the mean prediction
plt.plot(xstar,ystar,ls='-',c='r')
#Plot the errorbars
plt.fill_between(xstar, ystar+ystarerr, ystar-ystarerr, alpha=0.2, color='b')
#Labels
plt.xlabel(r"$x$",fontsize=24)
plt.ylabel(r"$y$",fontsize=24)
plt.xlim(min(xstar), max(xstar))
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.gcf().savefig("../figures/cosine_example.png")
plt.show()
