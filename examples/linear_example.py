import emulator as emulator
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', size=20)

#Create random training data
np.random.seed(12345)
x = np.linspace(0,10,num=10)
yerr = 0.05+0.5 * np.random.rand(len(x))
slope = 1.333
y = slope*x + yerr - 1

#Train
emu = emulator.Emulator(name="example_emulator",xdata=x,ydata=y,yerr=np.fabs(yerr))
emu.train()

#Create some data to predict on
xstar = np.linspace(min(x)-3,max(x)+9,500)
ystar,ystarvar = emu.predict(xstar)
ystarerr = np.sqrt(ystarvar)

#Plot the training data
plt.errorbar(x,y,np.fabs(yerr),ls='',marker='o',color='k',ms=8,label="f")
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
