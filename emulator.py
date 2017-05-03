"""
This is a class for our emulators. It is a simple GP implementation
that takes in some data, errorbars on that data, and an initial guess
for hyperparameters for the GP and then emulates it. This is for
1D data or a vector input for x.

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
            xdata (array_like): Location of the training data in domain space.
            ydata (array_like): The value, or targets, of the training data.
            yerr (array_like): The standard deviations or error bars on the ydata.
            kernel_exponent (float; optional): The exponent used in the kernel function. Defaults is 2, or squared exponential.

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


    def save(self,path=None):
        """The save method. Uses pickle to save
        everything about the emulator.

        Args:
            path (:obj:`str`, optional): The path of where to save the emulator. Otherwise it saves it at './'.

        """
        if path == None: pickle.dump(self,open("./%s.p"%(self.name),"wb"))
        else: pickle.dump(self,open("%s/%s.p"%(path,self.name),"wb"))
        return


    def load(self,input_path):
        """The load method. Uses pickle to load in
        a saved emulator and overwrite the attributes
        in self with those found in the loaded emulator.

        Args:
            input_path (:obj:`str`): the path to the emulator to be loaded.

        """
        emu_in = pickle.load(open("%s.p"%(input_path),"rb"))
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

    def Corr(self,x1,x2,length,amplitude):
        """The kriging kernel, or correlation function. It uses the squared exponential by default.

        Args:
            x1 (float or array_like): First data point.
            x2 (float or array_like): Second data point.
            length (float or array_like): Kernel length for each dimension in x.
            amplitude (float): Correlation amplitude.

        Returns:
            result (float): Correlation between x1 and x2.

        """
        return amplitude*np.exp(-0.5*np.sum(np.fabs(x1-x2)**self.kernel_exponent/length))

    def make_Kxx(self,length,amplitude):
        """Creates the Kxx array. This is the represents the connectivity between all of the training data, x.

        Note: Sets self.Kxx, Kxx_det, and Kxx_inv.

        Args:
            length (float or array_like): Kernel length for each dimension in x.
            amplitude (float): Correlation amplitude.

        """
        self.Kxx = amplitude*np.exp(np.sum(self.Kernel[:,:]/length,-1))
        self.Kxx += np.diag(self.yvar)
        self.Kxx_det = np.linalg.det(self.Kxx)
        self.Kxx_inv = np.linalg.inv(self.Kxx)
        return self.Kxx
    

    def make_Kxxstar(self,xstar):
        """Computes the next row/column that extends K_xx. This is the covariance of xstar with the training points, xdata.

        Args:
            xstar (float or array_like): The new point to predict at.

        Returns:
            K(x,xstar) (array_like): The row/column that is the covariance of xstar with xdata.

        """
        length,amplitude = self.lengths_best,self.amplitude_best
        self.Kxxstar = np.array([self.Corr(xi,xstar,length,amplitude) for xi in self.xdata])
        return self.Kxxstar

    def make_Kxstarxstar(self,xstar):
        """Computes the next diagonal element that would extend K_xx to the bottom right. Represents the variance of x_star.

        Args:
            xstar (float or array_like): The new point to predict at.

        Returns:
            K(xstar,xstar) (float): Variance of xstar.

        """
        length,amplitude = self.lengths_best,self.amplitude_best
        self.Kxstarxstar = self.Corr(xstar,xstar,length,amplitude)
        return self.Kxstarxstar

    def lnp(self,params):
        """The log of the probability distribution being sampled. This is used to find the optimal amplitude and kriging length(s).

        Args:
            params (array_like): Contains the length(s) and the amplitude.

        Returns:
            log_probability (float): Natural log of the probability of these parameters.

        """
        y = self.ydata
        lengths,amplitude = params[:-1],params[-1]
        if amplitude < 0.0: return -np.inf
        Kxx = self.make_Kxx(lengths,amplitude)
        if self.Kxx_det < 0: return -np.inf
        Kinv = self.Kxx_inv
        return -0.5*np.dot(y,np.dot(Kinv,y)) - 0.5*np.log(2*np.pi*self.Kxx_det)

    def train(self):
        """Train the emulator on the given x and y data.

        Note: this is necessary before making predictions.

        """
        nll = lambda *args: -self.lnp(*args)
        lengths_guesses = np.ones_like(self.lengths_best)
        guesses = np.concatenate([lengths_guesses,np.array([self.amplitude_best])])
        result = op.minimize(nll, guesses, method='BFGS')['x']
        self.lengths_best,self.amplitude_best = result[:-1],result[-1]
        self.make_Kxx(self.lengths_best,self.amplitude_best)
        self.trained = True
        return

    def predict_one_point(self,xstar):
        """Predicts a single data point given some point in the domain, xstar.

        Args:
            xstar (float or array_like): Single location in the domain.

        Returns:
            ystar (float): Predicted data point.
            ystar_variance (float): Variance of ystar.

        """
        if not self.trained: raise Exception("Emulator is not yet trained")
        self.make_Kxxstar(xstar)
        self.make_Kxstarxstar(xstar)
        Kxx,Kinv,Kxxs,Kxsxs, = self.Kxx,self.Kxx_inv,\
                               self.Kxxstar,self.Kxstarxstar
        return (np.dot(Kxxs,np.dot(Kinv,self.ydata))+self.ymean, Kxsxs - np.dot(Kxxs,np.dot(Kinv,Kxxs)))

    def predict(self,xstar):
        """Predicts an array of data points given an array of points in the domain, xstar.

        Args:
            xstar (array_like): List of points in the domain.

        Returns:
            ystar (array_like): List of predicted data points.
            ystar_variance (array_like): List of variances of ystar.

        """
        if not self.trained: raise Exception("Emulator is not yet trained")
        return np.array([self.predict_one_point(xsi) for xsi in xstar]).T

"""
Here is an example of how to use the emulator.
"""
if __name__ == '__main__':
    #Create some junk data to emulate
    Nx = 25 #number of x points

    #Emulate some periodic data
    np.random.seed(85719)
    x = np.linspace(0.,10.,num=Nx)
    yerr = 0.05 + 0.5*np.random.rand(Nx)
    y = np.sin(x) + 1

    #Create an emulator and train it
    emu = Emulator(name="Dev_emulator",xdata=x,ydata=y,yerr=np.fabs(yerr))
    emu.train()

    #Predict with it
    N = 100
    xstar = np.linspace(np.min(x)-5,np.max(x)+5,N)
    ystar,ystarvar = emu.predict(xstar)
    ystarerr = np.sqrt(ystarvar)

    #Plot everything
    import matplotlib.pyplot as plt
    plt.errorbar(x,y,np.fabs(yerr),ls='',marker='o')
    plt.plot(xstar,ystar,ls='-',c='r')
    plt.fill_between(xstar, ystar+ystarerr, ystar-ystarerr, alpha=0.2, color='b')
    plt.show()
