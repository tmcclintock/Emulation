"""
This is identical to the cosine_example.py code, except that it
shows the relative size of L and k_0. The former corresponds to the
correlation length of the kernel or more generally the size of "feautures"
in the data, and the latter is the uncertainty in the emulator in 
regions it has to extrapolate.
"""
import emulator
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text',usetex=True)
plt.rc('font', size=20)

#Create random training data
np.random.seed(85719)
x = np.linspace(0,10,num=10)
yerr = 0.05+0.5 * np.random.rand(len(x))
y = np.cos(x) + yerr + 1

#Train
emu = emulator.Emulator(name="example_emulator",xdata=x,ydata=y,yerr=np.fabs(yerr))
emu.train()

#Create some data to predict on
xstar = np.linspace(min(x)-10,max(x)+10,500)
ystar,ystarvar = emu.predict(xstar)
ystarerr = np.sqrt(ystarvar)

#Get out the hyperparameters
Ls, k0 = emu.Ls[0], emu.k0

#Plot the training data
plt.errorbar(x,y,np.fabs(yerr),ls='',marker='o',color='k',ms=8)
#Plot the mean prediction
plt.plot(xstar,ystar,ls='-',c='r')
#Plot the errorbars
plt.fill_between(xstar, ystar+ystarerr, ystar-ystarerr, alpha=0.2, color='b')
#Plot the kernel length region
minx = x[np.argmin(y)]
ylims = plt.gca().get_ylim()
plt.fill_between([minx-Ls/2., minx+Ls/2.], ylims[0], ylims[1], facecolor="purple", alpha=0.3, zorder=-1, label=r"$L$ region")
#Plot the kernel amplitude region
ymean = np.mean(y)
err = np.sqrt(k0)
plt.fill_between([min(xstar),max(xstar)], ymean-err, ymean+err, facecolor="green", alpha=0.3, zorder=-2, label=r"$k_0$ region")
#Labels
plt.xlabel(r"$x$",fontsize=24)
plt.ylabel(r"$y$",fontsize=24)
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.ylim(ylims)
plt.xlim(min(xstar), max(xstar))
plt.legend(fontsize=12)
plt.show()
