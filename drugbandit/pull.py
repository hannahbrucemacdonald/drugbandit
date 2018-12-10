import numpy as np
from yank.experiment import ExperimentBuilder
from yank.analyze import ExperimentAnalyzer


#############################################
#
# Functions to 'pull' the arm of a bandit
#
#############################################

# TODO
# - add openmmtools simulation function


def Yankpull(arm):
    '''

    :param arm: Bandit to be pulled
    :return: reward, calculated from Yank simulation
    '''
    arm.nsteps += 1.
    yaml_builder = ExperimentBuilder(arm.contents.format('yes',arm.init_steps+arm.nsteps*arm.pull_steps))
    yaml_builder.run_experiments()
    exp_analyzer = ExperimentAnalyzer(str(arm.name)+'/experiments')
    analysis_data = exp_analyzer.auto_analyze()
    # the following is the free_energy_diff of the WHOLE simulation, need the average of the last pull_steps
    return analysis_data['free_energy']['free_energy_diff']


def Pull(arm):
    '''

    :param arm: Bandit to be pulled
    :return: reward, calculated from sampling from likelihood
    '''
    arm.nsteps += 1.
    return np.random.normal(arm.fe, arm.error)


def PullSample(arm):
    arm.nsteps += 1.
