import numpy as np
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats
from yank.experiment import ExperimentBuilder
from yank.analyze import ExperimentAnalyzer

#############################################
#
# Gaussian bayesian bandit
#
#############################################

class Bandit(object):
    """ Gaussian bayesian bandit
    Notes
    -----
    A bayesian bandit with a likelihood of unknown expectation and variance with a normal-inverse gamma conjugate prior
    References
    ----------
    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    Examples
    --------
    """
    def __init__(self,name,fe,error,index,color=np.random.rand(3)):
        """Construct a gaussian bayesian bandit
        Parameters
        ----------
        name : ligand name 
        fe : experimental free energy 
        error : associated experimental error (sigma) 
        index : index of ligand in input file
        color : color used to plot the ligand
        nsteps : number of times bandit has been pulled
        sum_value : sum used in updating of b hyper-parameter
                  : \sum^{n}_{i=1} (x_i - <x>)^2
        lamb_0 : prior lambda 
        a_0 : prior alpha
        b_0 : prior beta
        nu_0 : prior nu (expectation of prior)
             : initiated from short simulation
        lamb : posterior lambda 
        a : posterior alpha
        b : posterior beta
        nu : posterior nu 
        Notes
        -----
        Velocities should be set to zero before using this integrator.
        """
        self.name = name.replace(' ','')
        self.fe = fe #will be removed when sampling likelihood properly
        self.error = error # will be also taken out
        self.index = index
        self.color = color 

        self.nsteps = 0
        self.sum_value = 0.

        # prior parameters
        self.lamb_0 = self.lamb = 1.
        self.a_0 = self.a = 2.
        self.b_0 = self.b = 1.
        self.nu_0 = self.nu = 0.

        # obersevable/reward
        self.x = None  # average
        self.x_n = None # of current step



    def sample(self):
        # sampling sigma**2
        sig_squared = scipy.stats.invgamma.rvs(self.a,scale = self.lamb)
        std_dev = (sig_squared/self.lamb)**0.5
        # sampling x
        return np.random.normal(self.nu,std_dev)

    def calc_sigma(self):
        self.sigma = (self.b/(self.a - 1.))**0.5
        return

    def pull(self):
        self.nsteps += 1.
        self.x_n = np.random.normal(self.fe,self.error)
        return

    def update(self):
        if self.x is not None:
            self.x = (self.x*(self.nsteps-1.) + self.x_n)/(self.nsteps)
        else:
            self.x = self.x_n
        self.lamb = self.lamb_0 + self.nsteps
        self.nu = (self.lamb_0 + self.nsteps * np.mean(self.x))/(self.lamb)
        self.a = self.a_0 + 0.5 * self.nsteps
        self.sum_value += (self.x_n - self.x)**2
        self.b = self.b_0 + 0.5 * (self.sum_value + ((self.nsteps*self.lamb_0)/(self.lamb))*(np.mean(self.x) - self.nu_0)**2)
        return

    def plot(self):
        # plotting bandit
        self.calc_sigma()
        x = np.linspace(self.nu - 3 * self.sigma, self.nu + 3 * self.sigma, 100)
        y = mlab.normpdf(x, self.nu, self.sigma)
        plt.plot(x, y,color=self.color,label=self.name,alpha=0.5)
        # plotting experimental 
        x = np.linspace(self.fe - 3 * self.error, self.fe + 3 * self.error, 100)
        y = mlab.normpdf(x, self.fe, self.error)
        plt.plot(x, y,color=self.color,linestyle=':')
        return

    def print_bandit(self,steps):
        print('Ligand:{}'.format(self.name))
        print('Hydration FE: {0:.2f} kcal/mol'.format(self.nu))
        self.calc_sigma()
        print('Error: {0:.2f} kcal/mol'.format(self.sigma))
        print('Total time: {0:.2f}%'.format(self.nsteps*(100./steps)))
        print('')
        return


class SimulationBandit(Bandit):
    def __init__(self,name,fe,error,index,simulation,color=np.random.rand(3),init_steps=2000,pull_steps=1000):
        super().__init__(name,fe,error,index,color)
        self.contents = simulation.format(self.name,'resume_simulation: {}',self.index,'{}')
        self.init_steps = init_steps
        self.pull_steps = pull_steps
        yaml_builder = ExperimentBuilder(self.contents.format('no',self.init_steps))
        yaml_builder.run_experiments()
        exp_analyzer = ExperimentAnalyzer(str(self.name)+'/experiments')
        analysis_data = exp_analyzer.auto_analyze()
        self.nu_0 = analysis_data['free_energy']['free_energy_diff']

    def pull(self):
        self.nsteps += 1.
        yaml_builder = ExperimentBuilder(self.contents.format('yes',self.init_steps+self.nsteps*self.pull_steps))
        yaml_builder.run_experiments()
        exp_analyzer = ExperimentAnalyzer(str(self.name)+'/experiments')
        analysis_data = exp_analyzer.auto_analyze()
        # the following is the free_energy_diff of the WHOLE simulation, need the average of the last pull_steps
        self.x_n = analysis_data['free_energy']['free_energy_diff']
        return
