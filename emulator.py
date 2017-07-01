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

    def __init__(self, xdata, ydata, yerr, name=""):
        """The __init__ method for the emulator class.

        Args:
            xdata (array_like): Location of the training data in domain space.
            ydata (array_like): The value, or targets, of the training data.
            yerr (array_like): The standard deviations or error bars on the ydata.
        """
        if len(xdata) != len(ydata):raise ValueError("xdata and ydata must be the same length.")
        if len(yerr) != len(ydata):raise ValueError("ydata and yerr must be the same length.")
        self.name = name
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
        self.Ls = np.ones(len(np.atleast_1d(xdata[0])))
        self.k0 = 1.0
        Kernel = [] # Create an array of the differences between all X values
        for i in range(len(xdata)):
            Kernel.append([-0.5*np.fabs(xdata[i]-xdataj)**2 for xdataj in xdata])
        self.Kernel = np.array(Kernel)
        self.Kernel = self.Kernel.reshape(len(xdata),len(xdata),len(np.atleast_1d(xdata[0])))
        self.make_Kxx(self.Ls, self.k0)

    def Corr(self,x1,x2,Ls,k0):
        """The kriging kernel, or correlation function. It uses the squared exponential by default.

        Args:
            x1 (float or array_like): First data point.
            x2 (float or array_like): Second data point.
            Ls (float or array_like): Kernel length for each dimension in x.
            k0 (float): Correlation amplitude.

        Returns:
            result (float): Correlation between x1 and x2.

        """
        return k0*np.exp(-0.5*np.sum(np.fabs(x1-x2)**2/Ls))

    def make_Kxx(self,Ls,k0):
        """Creates the Kxx array. This is the represents the connectivity between all of the training data, x.

        Note: Sets self.Kxx, Kxx_det, and Kxx_inv.

        Args:
            Ls (float or array_like): Kernel length for each dimension in x.
            k0 (float): Correlation amplitude.

        """
        self.Kxx = k0*np.exp(np.sum(self.Kernel[:,:]/Ls,-1))
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
        Ls,k0 = self.Ls,self.k0
        self.Kxxstar = np.array([self.Corr(xi,xstar,Ls,k0) for xi in self.xdata])
        return self.Kxxstar

    def make_Kxstarxstar(self,xstar):
        """Computes the next diagonal element that would extend K_xx to the bottom right. Represents the variance of x_star.

        Args:
            xstar (float or array_like): The new point to predict at.

        Returns:
            K(xstar,xstar) (float): Variance of xstar.

        """
        Ls,k0 = self.Ls,self.k0
        self.Kxstarxstar = self.Corr(xstar,xstar,Ls,k0)
        return self.Kxstarxstar

    def lnp(self,params):
        """The log of the probability distribution being sampled. This is used to find the optimal amplitude and kriging length(s).

        Args:
            params (array_like): Contains the length(s) and the amplitude.

        Returns:
            log_probability (float): Natural log of the probability of these parameters.

        """
        y = self.ydata
        Lss,k0 = params[:-1],params[-1]
        if k0 < 0.0: return -np.inf
        Kxx = self.make_Kxx(Lss,k0)
        if self.Kxx_det < 0: return -np.inf
        Kinv = self.Kxx_inv
        return -0.5*np.dot(y,np.dot(Kinv,y)) - 0.5*np.log(2*np.pi*self.Kxx_det)

    def train(self):
        """Train the emulator on the given x and y data.

        Note: this is necessary before making predictions.

        """
        nll = lambda *args: -self.lnp(*args)
        Ls_guesses = np.ones_like(self.Ls)
        guesses = np.concatenate([Ls_guesses,np.array([self.k0])])
        result = op.minimize(nll, guesses, method='BFGS')['x']
        self.Ls,self.k0 = result[:-1],result[-1]
        self.make_Kxx(self.Ls,self.k0)
        return

    def predict_one_point(self,xstar):
        """Predicts a single data point given some point in the domain, xstar.

        Args:
            xstar (float or array_like): Single location in the domain.

        Returns:
            ystar (float): Predicted data point.
            ystar_variance (float): Variance of ystar.

        """
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
