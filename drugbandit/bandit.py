import numpy as np
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats
import scipy.special
from yank.experiment import ExperimentBuilder
from yank.analyze import ExperimentAnalyzer
import itertools

class Node(object):
    def __init__(self,name,fe,error,index):
        self.name = name
        self.fe = fe
        self.error = error
        self.index = index
        self.probability = 0.5
        self.a = 1
        self.b = 1

    def plot(self):
        x = np.linspace(0.,1.,100)
        plt.plot(x,scipy.stats.beta.pdf(x,self.a,self.b))
        return

    def sample(self):
        return np.random.beta(self.a,self.b)

    def update(self,success):
        self.a = self.a + success
        self.b = self.b + 1 - success

    def pull(self,lig_ids,edges,benchmark_id=0,n_steps=3):
        paths = self.find_all_paths(lig_ids,benchmark_id, n_steps)
        corner_fe, corner_error = self.combine_all_paths(paths,edges)
        new_prob = self.get_probability(corner_fe,corner_error)
        if np.isnan(new_prob):
            self.probability = 0.5
        else:
            self.probability = new_prob
        return scipy.stats.bernoulli.rvs(self.probability)


    def find_all_paths(self,lig_ids,benchmark_id,n_steps):
        # finding all the paths between this corner and the benchmark
        all_paths = []
        all_steps = [x for x in itertools.combinations(lig_ids,2)]
        for n_step in range(1,n_steps+1):
            for path in itertools.combinations(all_steps,n_step):
                endpoints = [[benchmark_id,self.index],[self.index,benchmark_id]] #only want paths that are from this corner to baseline corner
                if [path[0][0],path[-1][-1]] in endpoints: # check that the combination of steps goes between what we are interested in
                    cont = self.test_path_continuous(path) # see that the steps make up a continuous route
                    if cont:
                        if cont not in [cont[1] for cont in all_paths]: # check it's not a duplicate path
                            if cont[::-1] not in [cont[1] for cont in all_paths]: # check it's not in there backwards...
                                # saving the bookends, the continuous path and the path of steps, very redundant
                                all_paths.append([(cont[0], cont[-1]), cont, path])
#                    # TODO this can check for cycle closures in future
#                if path[0][0] == self.index and path[-1][-1] == self.index:
#                    test_path_continuous(path)
        return all_paths

    def combine_all_paths(self,paths,edges):
        all_fes = []
        all_variance = []
        for path in paths:
            route = path[2]
            path_fe = 0.
            path_variance = 0.
            for edge in route:
                contributor = [e for e in edges if e.indexes == list(edge)]
                if len(contributor) == 1:
                    contributor[0].update()
                    path_fe += contributor[0].mu  # forward
                if len(contributor) == 0:
                    contributor = [e for e in edges if e.indexes[::-1] == list(edge)]
                    path_fe -= contributor[0].mu  # backward
                if len(contributor) > 1:
                    print('ERROR')
                    quit()
                path_variance += (contributor[0].sigma) ** 2
            all_fes.append(path_fe)
            all_variance.append(path_variance)
        return np.mean(all_fes), np.mean(all_variance) ** 0.5

    def get_probability(self,mu,sigma):
        frac_better = 1 - 0.5 * (1 + scipy.special.erf((-mu) / (sigma * np.sqrt(2))))
        return frac_better



    def test_path_continuous(self,path):
        previous = [x for x in path[0]]
        continuous = [x for x in path[0]]
        for step in path[1:]:
            if step[0] != previous[1]:
                return None  # non-continuous path
            elif step[1] in continuous:
                return None  # self-looping path
            else:
                continuous.append(step[1])
                previous = [x for x in step]
        return continuous

class Edges(object):
    def __init__(self,liga,ligb):
        self.mu = 0.
        self.sigma = 1.
        self.name = liga.name+'-'+ligb.name
        self.bandits = []
        self.indexes = [liga.index, ligb.index]
        self.fe = liga.fe - ligb.fe
        self.error = np.sqrt((liga.error **2 + ligb.error**2))
        self.color = np.random.rand(3)
        for i in [1,2,5]:
            self.bandits.append(Bandit(self.name,self.fe,self.error*i,color=self.color))

    def update(self):
        for band in self.bandits:
            band.calc_sigma()
        self.mu = np.sum([band.nu*band.nsteps for band in self.bandits])/np.sum([band.nsteps for band in self.bandits])
        variance = (np.sum([(band.sigma**2+band.nu**2)*band.nsteps for band in self.bandits]) / np.sum([band.nsteps for band in self.bandits]))
        self.sigma = np.sqrt(variance - self.mu**2)
        return

    def plot(self):
        '''
        Plots both the likelihood distribution and the posterior distribution of the banditS
        :return:
        '''
        x = np.linspace(self.mu - 3 * self.sigma, self.mu + 3 * self.sigma, 100)
        y = scipy.stats.norm.pdf(x, self.mu, self.sigma)
        plt.plot(x, y,'-.',color=self.color,label=self.name)
        # plotting experimental
        for band in self.bandits:
            band.plot()
        return

#############################################
#
# Gaussian bayesian bandit
#
#############################################

class Bandit(object):
    ''' Gaussian bayesian bandit
    Notes
    -----
    A bayesian bandit with a likelihood of unknown expectation and variance with a normal-inverse gamma conjugate prior
    References
    ----------
    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    Examples
    --------
    '''
    def __init__(self,name,fe,error,color=np.random.rand(3)):
        '''
        Bayesian bandit
        :param name: ligand name
        :param fe: experimental free energy
        :param error: associated experimental error (sigma)
        :param index: index of ligand in input file
        :param color: color used to plot ligand
        '''
        self.name = name.replace(' ','')
        self.fe = fe #will be removed when sampling likelihood properly
        self.error = error # will be also taken out
        self.color = color

        self.nsteps = 0
        self.sum_value = 0.

        # prior parameters
        self.lamb_0 = self.lamb = 1.
        self.a_0 = self.a = 2.
        self.b_0 = self.b = 1.
        self.nu_0 = self.nu = 0.

        # obserevable/reward
        self.x = None  # average
        self.x_n = None # of current step

        self.sigma_history = []

    def reset_nsteps(self,newval = 0):
        self.nsteps = newval

    def sample(self):
        '''
        :return: random variate from normal-inverse gamma function
        '''
        # sampling sigma**2
        sig_squared = scipy.stats.invgamma.rvs(self.a,scale = self.lamb)
        std_dev = (sig_squared/self.lamb)**0.5
        # sampling x
        return np.random.normal(self.nu,std_dev)

    def calc_sigma(self):
        '''
        :return: sigma of gaussian
        '''
        self.sigma = (self.b/(self.a - 1.))**0.5
        self.sigma_history.append(self.sigma)
        return

    def update(self,x_n):
        '''
        Updates posterior based on reward
        :param x_n: reward
        :return:
        '''
        if self.x is not None:
            self.x = (self.x*(self.nsteps-1.) + x_n)/(self.nsteps)
        else:
            self.x = x_n
        self.lamb = self.lamb_0 + self.nsteps
        self.nu = (self.lamb_0 + self.nsteps * np.mean(self.x))/(self.lamb)
        self.a = self.a_0 + 0.5 * self.nsteps
        self.sum_value += (x_n - self.x)**2
        self.b = self.b_0 + 0.5 * (self.sum_value + ((self.nsteps*self.lamb_0)/(self.lamb))*(np.mean(self.x) - self.nu_0)**2)
        return

    def plot(self):
        '''
        Plots both the likelihood distribution and the posterior distribution of the bandit
        :return:
        '''
        # plotting bandit
        self.calc_sigma()
        x = np.linspace(self.nu - 3 * self.sigma, self.nu + 3 * self.sigma, 100)
        y = scipy.stats.norm.pdf(x, self.nu, self.sigma)
        plt.plot(x, y,color=self.color,alpha=0.5)
        # plotting experimental 
        x = np.linspace(self.fe - 3 * self.error, self.fe + 3 * self.error, 100)
        y = scipy.stats.norm.pdf(x, self.fe, self.error)
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
        '''
        Bandit initiated with short Yank simulation
        :param name: ligand name
        :param fe:  experimental free energy
        :param error: associated experimental error (sigma)
        :param index: index of ligand in input file
        :param simulation:
        :param color: color used to plot ligand
        :param init_steps: number of steps for simulation to initiate bandit
        '''
        super().__init__(name,fe,error,index,color)
        self.contents = simulation.format(self.name,'resume_simulation: {}',self.index,'{}')
        self.init_steps = init_steps
        self.pull_steps = pull_steps
        yaml_builder = ExperimentBuilder(self.contents.format('no',self.init_steps))
        yaml_builder.run_experiments()
        exp_analyzer = ExperimentAnalyzer(str(self.name)+'/experiments')
        analysis_data = exp_analyzer.auto_analyze()
        self.nu_0 = analysis_data['free_energy']['free_energy_diff']

