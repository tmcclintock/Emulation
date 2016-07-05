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
import pickle

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

    """
    These functions save all parts of the emulator, so that it can be
    loaded up again. This could save the hassle of
    training, if that happens to take a long time.
    """
    def save(self):
        pickle.dump(self,open("%s.p"%(self.name),"wb"))
        return

    def load(self,fname):
        emu_in = pickle.load(open("%s.p"%(fname),"rb"))
        self.name = emu_in.name
        self.kernel_exponent = emu_in.kernel_exponent
        self.xdata = emu_in.xdata
        self.ydata = emu_in.ydata
        self.yerr = emu_in.yerr
        self.Kxx = emu_in.Kxx
        self.Kinv = emu_in.Kinv
        self.Kxxstar = emu_in.Kxxstar
        self.Kxstarxstar = emu_in.Kxstarxstar
        self.length_best = emu_in.length_best
        self.amplitude_best = emu_in.amplitude_best
        self.trained = emu_in.trained
        return

    """
    This is the kriging kernel. You can change it's form either
    by hard-coding in something different or by just
    changing the kernel exponent.
    """
    def Corr(self,x1,x2,length,amplitude):
        return amplitude*np.exp(-0.5*np.sum(np.fabs(x1-x2)**self.kernel_exponent)/length)

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
        Kxxs = np.zeros_like(x)
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
        y = self.ydata
        length,amplitude = np.exp(params[0]),np.exp(params[1])
        K = self.make_Kxx(length,amplitude)
        Kinv = np.linalg.inv(K)
        return -0.5*np.dot(y,np.dot(Kinv,y))\
            - 0.5*np.log(np.linalg.det(2*np.pi*K))

    """
    This initiates the training process and
    remembers the length, amplitude, and Kxx array.
    """
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
    #x = np.linspace(0,10,num=10)#10 * np.sort(np.random.rand(Nx))
    #yerr = 0.05+0.5 * np.random.rand(len(x))
    #y = np.sin(x) + yerr
    x = np.loadtxt("test_data/scale_factors.txt")
    print x
    y = np.loadtxt("test_data/farr.txt")
    yerr = np.loadtxt("test_data/ferr.txt")

    #Declare an emulator, train it, and predict with it.
    emu = Emulator(name="Test_emulator",xdata=x,ydata=y,yerr=np.fabs(yerr))
    emu.train()
    emu.save()
    emu.load("Test_emulator")
    print "Best parameters = ",emu.length_best,emu.amplitude_best

    xstar = np.linspace(np.min(x)-0.1,np.max(x)+1,500)
    ystar,ystarerr = emu.predict(xstar)

    import matplotlib.pyplot as plt
    plt.errorbar(x,y,np.fabs(yerr),ls='',marker='o',ms=2,label="f")
    plt.plot(xstar,ystar,ls='-',c='r')
    plt.plot(xstar,ystar+ystarerr,ls='-',c='g')
    plt.plot(xstar,ystar-ystarerr,ls='-',c='g')
    #plt.show()


    x2 = np.loadtxt("test_data/scale_factors.txt")
    y2 = np.loadtxt("test_data/garr.txt")
    yerr2 = np.loadtxt("test_data/gerr.txt")
    #Declare an emulator, train it, and predict with it.
    emu2 = Emulator(name="Test_emulator2",xdata=x2,ydata=y2,yerr=np.fabs(yerr2))
    emu2.train()
    emu2.save()
    emu2.load("Test_emulator2")
    print "Best parameters = ",emu2.length_best,emu2.amplitude_best
    xstar2 = np.linspace(np.min(x2)-0.1,np.max(x2)+1,500)
    ystar2,ystarerr2 = emu2.predict(xstar2)
    plt.errorbar(x2,y2,np.fabs(yerr2),ls='',marker='o',ms=2,label="g")
    plt.plot(xstar2,ystar2,ls='-',c='r')
    plt.plot(xstar2,ystar2+ystarerr2,ls='-',c='g')
    plt.plot(xstar2,ystar2-ystarerr2,ls='-',c='g')
    plt.ylabel("Tinker08 parameters",fontsize=28)
    plt.xlabel("scale factor",fontsize=28)
    leg = plt.legend(loc="lower right",fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.subplots_adjust(bottom=0.15)
    plt.xlim(0.15,1.1)
    plt.ylim(0.4,1.3)
    plt.show()
