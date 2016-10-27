import Emulator as Emulator
import numpy as np

#Create random training data
np.random.seed(85719)
x = np.linspace(0,10,num=10)
yerr = 0.05+0.5 * np.random.rand(len(x))
y = np.cos(x) + yerr

#Train
emu = Emulator.Emulator(name="example_emulator",xdata=x,ydata=y,yerr=np.fabs(yerr))
emu.train()

#Create some data to predict on
xstar = np.linspace(min(x)-1,max(x)+1,500)
ystar,ystarvar = emu.predict(xstar)
ystarerr = np.sqrt(ystarvar)

#Visualize
import matplotlib.pyplot as plt
plt.rc('text',usetex=True, fontsize=20)
#Plot the training data
plt.errorbar(x,y,np.fabs(yerr),ls='',marker='o',ms=2,label="f")
#Plot the mean prediction
plt.plot(xstar,ystar,ls='-',c='r')
#Plot the errorbars
plt.plot(xstar,ystar+ystarerr,ls='-',c='g')
plt.plot(xstar,ystar-ystarerr,ls='-',c='g')
#Labels
plt.xlabel("x")
plt.ylabel("y")
plt.show()
