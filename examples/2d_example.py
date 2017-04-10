"""
This contains code that shows how the emulator
would handle input where the domain is multidimensional.
"""
import numpy as np
import emulator as Emulator
#Create some junk data to emulate
Nx = 25 #number of x points

#Try emulating on some periodic data
np.random.seed(85719)
x1 = np.linspace(0.,10.,num=Nx)
x2 = np.linspace(0.,10.,num=Nx)
yerr = 0.05 + 0.5*np.random.rand(Nx)
y = np.sin(x1) + 1 + np.cos(x2)
x = np.vstack((x1,x2)).T

#Declare an emulator, train it, and predict with it.
print x.shape, y.shape
emu = Emulator.Emulator(name="Dev_emulator",xdata=x,ydata=y,yerr=np.fabs(yerr))#,kernel_exponent=1)
emu.train()
print "Best parameters = ",emu.lengths_best,emu.amplitude_best

N = 100
x1star = np.linspace(np.min(x1)-5,np.max(x1)+5,N)
x2star = np.linspace(np.min(x2)-5,np.max(x2)+5,N)
xstar = np.vstack((x1star,x2star)).T

ystar,ystarvar = emu.predict(xstar)
ystarerr = np.sqrt(ystarvar)

import matplotlib.pyplot as plt
plt.errorbar(x1,y,np.fabs(yerr),ls='',marker='o')
plt.plot(x1star,ystar,ls='-',c='r')
plt.fill_between(x1star, ystar+ystarerr, ystar-ystarerr, alpha=0.2, color='b')
plt.show()
