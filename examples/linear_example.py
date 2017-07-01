import emulator as emulator
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', size=20)

#Create random training data
np.random.seed(0)
Nx = 10
x = 10*np.sort(np.random.rand(Nx))
yerr = 0.4 * np.ones_like(x)
slope = 1.333
intercept = 1
y = slope*x + intercept + yerr * np.random.randn(len(x))

#Train
emu = emulator.Emulator(name="example_emulator",xdata=x,ydata=y,yerr=np.fabs(yerr))
emu.train()

#Create some data to predict on
xstar = np.linspace(min(x)-3,max(x)+9,500)
ystar,ystarvar = emu.predict(xstar)
ystarerr = np.sqrt(ystarvar)

#Plot the training data
plt.errorbar(x,y,np.fabs(yerr),ls='',marker='.',color='k',ms=8,label="f")
#Plot the mean prediction
plt.plot(xstar,ystar,ls='-',c='r')
#Plot the errorbars
plt.fill_between(xstar, ystar+ystarerr, ystar-ystarerr, alpha=0.2, color='b')
plt.xlim(min(xstar), max(xstar))
#Labels
plt.xlabel(r"$x$",fontsize=24)
plt.ylabel(r"$y$",fontsize=24)
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.show()
