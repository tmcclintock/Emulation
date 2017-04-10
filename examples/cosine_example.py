import emulator
import numpy as np

#Create random training data
np.random.seed(85719)
x = np.linspace(0,10,num=10)
yerr = 0.05+0.5 * np.random.rand(len(x))
y = np.cos(x) + yerr + 1

#Train
emu = emulator.Emulator(name="example_emulator",xdata=x,ydata=y,yerr=np.fabs(yerr))
emu.train()

#Create some data to predict on
xstar = np.linspace(min(x)-3,max(x)+3,500)
ystar,ystarvar = emu.predict(xstar)
ystarerr = np.sqrt(ystarvar)

#Visualize
import matplotlib.pyplot as plt
plt.rc('text',usetex=True, fontsize=20)
#Plot the training data
plt.errorbar(x,y,np.fabs(yerr),ls='',marker='o',color='k',ms=8,label="f")
#Plot the mean prediction
plt.plot(xstar,ystar,ls='-',c='r')
#Plot the errorbars
plt.fill_between(xstar, ystar+ystarerr, ystar-ystarerr, alpha=0.2, color='b')
#Labels
plt.xlabel(r"$x$",fontsize=24)
plt.ylabel(r"$y$",fontsize=24)
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.gcf().savefig("../figures/cosine_example.png")
plt.show()
