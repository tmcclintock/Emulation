"""

This is a class for our emulators. It is a simple GP implementation
that takes in some data, errorbars on that data, and an initial guess
for hyperparameters for the GP and then emulates it. This is for
1D data or a vector input for x.

In order to use it, you create an Emulator() object with
xdata, ydata, and yerr. Then you call Emulator.train()
and then you are allowed to call Emulator.predict(xstar)
on some new input xstar.

"""

import numpy as np
import scipy.optimize as op

class Emulator(object):
    def __init__(self,xdata,ydata,yerr,name="",kernel_exponent=2):
        self.name = name
        self.kernel_exponent = kernel_exponent
        self.xdata = xdata
        self.ydata = ydata
        self.yerr = yerr
        self.Kxx = 0
        self.Kinv = 0
        self.Kxxstar = 0
        self.Kxstarxstar = 0
        self.length_best = 0
        self.amplitude_best = 0
        self.trained = False

    def __str__(self):
        return self.name

    def Corr(self,x1,x2,length,amplitude):
        return amplitude*np.exp(-0.5*np.sum(np.fabs(x1-x2)**self.kernel_exponent)/length)

    def make_Kxx(self,length,amplitude):
        x,y,yerr = self.xdata,self.ydata,self.yerr
        N = len(x)
        Kxx = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                Kxx[i,j] = self.Corr(x[i],x[j],length,amplitude)
            Kxx[i,i] += yerr[i]**2
        self.Kxx = Kxx
        return Kxx
        
    def make_Kinv(self):
        Kxx = self.Kxx
        self.Kinv = np.linalg.inv(Kxx)
        return self.Kinv

    def make_Kxxstar(self,xs):
        x,length,amplitude = self.xdata,self.length_best,self.amplitude_best
        Kxxs = np.zeros_like(x)
        for i in range(len(x)):
            Kxxs[i] = self.Corr(x[i],xs,length,amplitude)
        self.Kxxstar = Kxxs
        return Kxxs

    def make_Kxstarxstar(self,xs):
        length,amplitude = self.length_best,self.amplitude_best
        self.Kxstarxstar = self.Corr(xs,xs,length,amplitude)
        return self.Kxstarxstar

    def lnp(self,params):
        length,amplitude = np.exp(params[0]),np.exp(params[1])
        K = self.make_Kxx(length,amplitude)
        Kinv = np.linalg.inv(K)
        return -0.5*np.dot(y,np.dot(Kinv,y))\
            - 0.5*np.log(np.linalg.det(2*np.pi*K))

    def train(self):
        nll = lambda *args: -self.lnp(*args)
        guesses = (1.0,1.0)
        lb,ab = op.minimize(nll,guesses)['x']
        lb,ab = np.exp(lb),np.exp(ab)
        self.length_best,self.amplitude_best = lb,ab
        self.make_Kxx(lb,ab)
        self.make_Kinv()
        self.trained = True
        return

    def predict_one_point(self,xs):
        if not self.trained:
            raise Exception("Emulator is not yet trained")
        self.make_Kxxstar(xs)
        self.make_Kxstarxstar(xs)
        Kxx,Kinv,Kxxs,Kxsxs, = self.Kxx,self.Kinv,\
                               self.Kxxstar,self.Kxstarxstar
        return (np.dot(Kxxs,np.dot(Kinv,self.ydata)),\
                np.sqrt(Kxsxs - np.dot(Kxxs,np.dot(Kinv,Kxxs))))

    def predict(self,xs):
        if not self.trained:
            raise Exception("Emulator is not yet trained")
        ystar = []
        for xsi in xs:
            ystar.append(self.predict_one_point(xsi))
        ystar = np.array(ystar)
        return ystar.T

"""
Here is a unit test for the emulator.
"""
if __name__ == '__main__':
    #Create some junk data to emulate
    Nx = 10 #number of x points

    #Try emulating on some periodic data
    np.random.seed(85719)
    x = np.linspace(0,10,num=10)#10 * np.sort(np.random.rand(Nx))
    yerr = 0.05+0.5 * np.random.rand(len(x))
    y = np.sin(x) + yerr

    #Declare an emulator, train it, and predict with it.
    emu = Emulator(name="Test emulator",xdata=x,ydata=y,yerr=np.fabs(yerr))#,kernel_exponent=1)
    emu.train()
    print "Best parameters = ",emu.length_best,emu.amplitude_best
    xstar = np.linspace(np.min(x)-1,np.max(x)+1,500)
    ystar,ystarerr = emu.predict(xstar)

    import matplotlib.pyplot as plt
    plt.errorbar(x,y,np.fabs(yerr),ls='',marker='o')
    plt.plot(xstar,ystar,ls='-',c='r')
    plt.plot(xstar,ystar+ystarerr,ls='-',c='g')
    plt.plot(xstar,ystar-ystarerr,ls='-',c='g')
    plt.show()
