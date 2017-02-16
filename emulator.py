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
import pickle, sys

class Emulator(object):
    """The emulator class.

    """

    def __init__(self,xdata,ydata,yerr,name="",kernel_exponent=2):
        """The __init__ method for the emulator class.

        Args:
            xdata: An array of floats or an nd.array of floats. Represents
                the location of the training data in domain space.
            ydata: An array of floats. The value or targets of the training data.
            yerr: An array of floats. The standard deviations or error bars
                on the ydata.
            kernel_exponent (optional): The exponent used in the kernel function.

        """
        if len(xdata) != len(ydata):raise ValueError("xdata and ydata must be the same length.")
        if len(yerr) != len(ydata):raise ValueError("ydata and yerr must be the same length.")
        self.name = name
        self.kernel_exponent = kernel_exponent
        self.xdata = np.array(xdata)
        self.Nx = len(self.xdata)
        self.ymean = np.mean(ydata)
        self.ydata_true = ydata
        self.ydata = ydata - self.ymean #Take off the mean
        self.yerr = yerr
        self.yvar = self.yerr**2
        self.Kxx = None
        self.Kinv = None
        self.Kxxstar = None
        self.Kxstarxstar = None
        self.lengths_best = np.ones(len(np.atleast_1d(xdata[0])))
        self.amplitude_best = 1.0
        Kernel = [] # Create an array of the differences between all X values
        for i in range(len(xdata)):
            Kernel.append([-0.5*np.fabs(xdata[i]-xdataj)**self.kernel_exponent for xdataj in xdata])
        self.Kernel = np.array(Kernel)
        self.Kernel = self.Kernel.reshape(len(xdata),len(xdata),len(np.atleast_1d(xdata[0])))
        self.trained = False

    def __str__(self):
        return self.name

    """
    These functions save all parts of the emulator, so that it can be
    loaded up again. This could save the hassle of
    training, if that happens to take a long time.
    """
    def save(self,path=None):
        if path == None: pickle.dump(self,open("%s.p"%(self.name),"wb"))
        else: pickle.dump(self,open("%s.p"%path,"wb"))
        return

    def load(self,fname):
        emu_in = pickle.load(open("%s.p"%(fname),"rb"))
        self.name = emu_in.name
        self.kernel_exponent = emu_in.kernel_exponent
        self.xdata = emu_in.xdata
        self.Nx = emu_in.Nx
        self.ydata_true = emu.ydata
        self.ydata = emu_in.ydata
        self.ymean = emu.ymean
        self.yerr = emu_in.yerr
        self.Kxx = emu_in.Kxx
        self.Kinv = emu_in.Kinv
        self.Kxxstar = emu_in.Kxxstar
        self.Kxstarxstar = emu_in.Kxstarxstar
        self.lengths_best = emu_in.lengths_best
        self.amplitude_best = emu_in.amplitude_best
        self.Kernel = emu.Kernel
        self.trained = emu_in.trained
        return

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
        Kxx = np.zeros((self.Nx,self.Nx))
        Kxx = amplitude*np.exp(np.sum(self.Kernel[:,:]/length,-1))
        Kxx += np.diag(self.yvar)
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
        x,length,amplitude = self.xdata,self.lengths_best,self.amplitude_best
        N = len(x)
        Kxxs = np.zeros(N)
        for i in range(len(x)): Kxxs[i] = self.Corr(x[i],xs,length,amplitude)
        self.Kxxstar = Kxxs
        return Kxxs

    """
    This creates the new element K_xstarxstar.
    Used for predicting ystar at xstar.
    """
    def make_Kxstarxstar(self,xs):
        length,amplitude = self.lengths_best,self.amplitude_best
        self.Kxstarxstar = self.Corr(xs,xs,length,amplitude)
        return self.Kxstarxstar

    """
    This is the log probability used for training
    to find the best amplitude and kriging length.
    """
    def lnp(self,params):
        y = self.ydata
        lengths,amplitude = params[:-1],params[-1]
        if amplitude < 0.0: return -np.inf
        Kxx = self.make_Kxx(lengths,amplitude)
        Kdet = 2*np.pi*np.linalg.det(Kxx)
        if Kdet < 0: return -np.inf
        Kinv = self.make_Kinv()
        return -0.5*np.dot(y,np.dot(Kinv,y)) - 0.5*np.log(Kdet)

    """
    This initiates the training process and
    remembers the length, amplitude, and Kxx array.
    """
    def train(self):
        nll = lambda *args: -self.lnp(*args)
        lengths_guesses = np.ones_like(self.lengths_best)
        guesses = np.concatenate([lengths_guesses,np.array([self.amplitude_best])])
        result = op.minimize(nll,guesses)['x']
        self.lengths_best,self.amplitude_best = result[:-1],result[-1]
        self.make_Kxx(self.lengths_best,self.amplitude_best)
        self.make_Kinv()
        self.trained = True
        return

    """
    This predicts a single ystar given an xstar.

    The output is ystar and the variance of ystar.
    """
    def predict_one_point(self,xs):
        if not self.trained: raise Exception("Emulator is not yet trained")
        self.make_Kxxstar(xs)
        self.make_Kxstarxstar(xs)
        Kxx,Kinv,Kxxs,Kxsxs, = self.Kxx,self.Kinv,\
                               self.Kxxstar,self.Kxstarxstar
        return (np.dot(Kxxs,np.dot(Kinv,self.ydata))+self.ymean, Kxsxs - np.dot(Kxxs,np.dot(Kinv,Kxxs)))

    """
    This is a wrapper for predict_one_point so that
    we can work with a list of xstars that are predicted
    one at a time.

    The output is ystar and the variance of ystar.
    """
    def predict(self,xs):
        if not self.trained: raise Exception("Emulator is not yet trained")
        ystar,ystarvar = np.array([self.predict_one_point(xsi) for xsi in xs]).T
        return ystar,ystarvar

"""
Here is a unit test for the emulator.
"""
if __name__ == '__main__':
    #Create some junk data to emulate
    Nx = 25 #number of x points

    #Try emulating on some periodic data
    np.random.seed(85719)
    x = np.linspace(0.,10.,num=Nx)
    yerr = 0.05 + 0.5*np.random.rand(Nx)
    y = np.sin(x) + 1

    #Declare an emulator, train it, and predict with it.
    emu = Emulator(name="Dev_emulator",xdata=x,ydata=y,yerr=np.fabs(yerr))#,kernel_exponent=1)
    emu.train()
    #emu.save("pickled_files/test_emulator")
    #emu.load("pickled_files/test_emulator")

    N = 100
    xstar = np.linspace(np.min(x)-5,np.max(x)+5,N)

    ystar,ystarvar = emu.predict(xstar)
    ystarerr = np.sqrt(ystarvar)

    import matplotlib.pyplot as plt
    plt.errorbar(x,y,np.fabs(yerr),ls='',marker='o')
    plt.plot(xstar,ystar,ls='-',c='r')
    plt.plot(xstar,ystar+ystarerr,ls='-',c='g')
    plt.plot(xstar,ystar-ystarerr,ls='-',c='g')
    plt.show()
