"""

This is a class for our emulators. It is a simple GP implementation
that takes in some data, errorbars on that data, and an initial guess
for hyperparameters for the GP and then emulates it. This is for
1D data or a vector input for x.

In order to use it, you create an Emulator() object with
xdata, ydata, and yerr. Then you call Emulator.train()
and then you are allowed to call Emulator.predict(xstar)
on some new input xstar.

This has a different kriging length for each element in x.
"""

import numpy as np
import scipy.optimize as op

class Emulator(object):
    def __init__(self,xdata,ydata,yerr,name="",kernel_exponent=2):
        self.name = name
        self.kernel_exponent = kernel_exponent
        self.xdata = xdata.T
        self.ydata = ydata
        self.yerr = yerr
        self.Kxx = None
        self.Kinv = None
        self.Kxxstar = None
        self.Kxstarxstar = None
        self.lengths_best = np.ones(np.atleast_2d(xdata).shape[0])
        self.amplitude_best = 0
        self.trained = False

    def __str__(self):
        return self.name

    """
    This is the kriging kernel. You can change it's form either
    by hard-coding in something different or by just
    changing the kernel exponent.
    """
    def Corr(self,x1,x2,length,amplitude):
        result = amplitude*np.exp(-0.5*np.sum(np.fabs(x1-x2)**self.kernel_exponent/length))
        return result

    """
    This makes the Kxx array.
    """
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
        
    """
    This inverts the Kxx array.
    """
    def make_Kinv(self):
        Kxx = self.Kxx
        self.Kinv = np.linalg.inv(Kxx)
        return self.Kinv

    """
    This creates the new row/column extensions K_x,xstar.
    Used for predicting ystar at xstar.
    """
    def make_Kxxstar(self,xs):
        x,length,amplitude = self.xdata,self.length_best,self.amplitude_best
        N = len(x)
        Kxxs = np.zeros(N)
        for i in range(len(x)):
            Kxxs[i] = self.Corr(x[i],xs,length,amplitude)
        self.Kxxstar = Kxxs
        return Kxxs

    """
    This creates the new element K_xstarxstar.
    Used for predicting ystar at xstar.
    """
    def make_Kxstarxstar(self,xs):
        length,amplitude = self.length_best,self.amplitude_best
        self.Kxstarxstar = self.Corr(xs,xs,length,amplitude)
        return self.Kxstarxstar

    """
    This is the log probability used for training
    to find the best amplitude and kriging length.
    """
    def lnp(self,params):
        length,amplitude = np.exp(params[:-1]),np.exp(params[-1])
        K = self.make_Kxx(length,amplitude)
        Kinv = np.linalg.inv(K)
        return -0.5*np.dot(y,np.dot(Kinv,y))\
            - 0.5*np.log(np.linalg.det(2*np.pi*K))

    """
    This initiates the training process and
    remembers the length, amplitude, and Kxx array.
    """
    def train(self):
        #Before actually training, figure out how many
        #kriging lengths we need here
        ##################
        nll = lambda *args: -self.lnp(*args)
        length_guesses = np.ones_like(self.lengths_best)
        guesses = np.concatenate([length_guesses,np.array([1.0])])
        result = op.minimize(nll,guesses)['x']
        lb,ab = np.exp(result[:-1]),np.exp(result[-1])
        self.length_best,self.amplitude_best = lb,ab
        self.make_Kxx(lb,ab)
        self.make_Kinv()
        self.trained = True
        return

    """
    This predicts a single ystar given an xstar.
    """
    def predict_one_point(self,xs):
        if not self.trained:
            raise Exception("Emulator is not yet trained")
        self.make_Kxxstar(xs)
        self.make_Kxstarxstar(xs)
        Kxx,Kinv,Kxxs,Kxsxs, = self.Kxx,self.Kinv,\
                               self.Kxxstar,self.Kxstarxstar
        return (np.dot(Kxxs,np.dot(Kinv,self.ydata)),\
                np.sqrt(Kxsxs - np.dot(Kxxs,np.dot(Kinv,Kxxs))))

    """
    This is a wrapper for predict_one_point so that
    we can work with a list of xstars that are predicted
    one at a time.
    """
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
    x1 = np.linspace(0,10,num=10)#10 * np.sort(np.random.rand(Nx))
    x2 = -np.linspace(0.,10.,num=10) * 0.5
    yerr = 0.05+0.5 * np.random.rand(len(x1))
    y = np.sin(x1) + yerr + np.cos(x2)
    x = np.array([x1,x2])

    #Declare an emulator, train it, and predict with it.
    emu = Emulator(name="Test emulator",xdata=x,ydata=y,yerr=np.fabs(yerr))#,kernel_exponent=1)
    emu.train()
    print "Best parameters = ",emu.length_best,emu.amplitude_best
    xstar = np.array([np.linspace(np.min(x1)-1,np.max(x1)+1,500),\
                          np.ones(500)*np.mean(x2)]).T
    
    ystar,ystarerr = emu.predict(xstar)

    import matplotlib.pyplot as plt
    xplot = x1
    xsplot = np.linspace(np.min(x1)-1,np.max(x1)+1,500)
    plt.errorbar(xplot,y,np.fabs(yerr),ls='',marker='o')
    plt.plot(xsplot,ystar,ls='-',c='r')
    plt.plot(xsplot,ystar+ystarerr,ls='-',c='g')
    plt.plot(xsplot,ystar-ystarerr,ls='-',c='g')
    plt.show()
