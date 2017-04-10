"""
Here is an example with some real-world data.

The data is found in ./reaction_rate.txt
and is the nuclear reaction rate for the
reaction D(p,g)3He, or a deuterium+proton
into a photon and helium-3.

There are four columns in their data file,
T9, the temperature, the reaction rate
mean, and the upper and lower 1-sigma values.
"""
import emulator
import numpy as np
import matplotlib.pyplot as plt

#Get the data
data = np.genfromtxt("reaction_rate.txt")
T9 = data[:,0]
R = data[:,1]
low  = R-data[:,2]
high = data[:,3]-R
err = np.mean([low,high],0)

#Things are happening in log space, so use that
lT9 = np.log(T9)
lR = np.log(R)
lerr = err/R

#Make an emulator and train
emu = emulator.Emulator(name="RR",xdata=lT9,ydata=lR,yerr=lerr)
emu.train()

#Create some prediction
lT9_pred = np.linspace(min(lT9)*0.9,max(lT9)*1.1)
lR_pred,lR_predvar = emu.predict(lT9_pred)
lR_prederr = np.sqrt(lR_predvar)
T9_pred = np.exp(lT9_pred)
R_pred = np.exp(lR_pred)
R_prederr = lR_prederr*R_pred

#Plot everything
#Note: the emuator preduction is very tight, so the
#error bars are almost imperceptible
plt.errorbar(T9,R,yerr=err,c='k',marker='.',ls='')
plt.fill_between(T9_pred, R_pred+R_prederr, R_pred-R_prederr, alpha=0.2, color='b')
plt.plot(T9_pred,R_pred,c='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"${\rm Temperature}\ [{\rm K}\times10^9]$",fontsize=24)
plt.ylabel(r"${\rm Rate}\ [{\rm cm^3/mol/sec}]$",fontsize=24)
plt.title(r"$D(p,\gamma) ^3\!He$",fontsize=24)
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.gcf().savefig("../figures/reaction_rate.png")
plt.show()
