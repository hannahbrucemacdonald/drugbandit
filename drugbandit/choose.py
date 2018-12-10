import numpy as np
import random

#############################################
#
# Sampling schemes for Bayesian bandits
#
#############################################


# TODO
# write pursuit bandits


def e_greedy(arms,e,reverse=False):
    '''

    :param arms: list of Bandits that are being chosen between
    :param e: greedy parameter, float between 0 (random choice) , 1 (fully greedy choice)
    :param reverse: select the lowest arm, rather than highest
    :return: single Bandit
    '''
    if e >1. or e < 0.:
        print('e_greedy algoritm requires a probability, e, between 0 (random) and 1 (greedy)')
        print('e here is : {}'.format(e))
        print('quitting')
        quit()
    if np.random.rand(1) > e:
        return random.choice(arms)
    else:
        if reverse == False:
            return arms[np.argmax([arm.sample() for arm in arms])]
        else:   # reverse selection, optimising for the minimum
            return arms[np.argmin([arm.sample() for arm in arms])]

def boltzmann_corners(corners,temp):
    expectations = [x.probability for x in corners]
    expectations = [np.exp(expect/temp) for expect in expectations]
    normalize = np.sum(expectations)
    probabilities = [(expect)/normalize for expect in expectations]
    return np.random.choice(corners,1, p=probabilities)[0]


def boltzmann(arms, temp,reverse=False):
    '''
    :param arms: list of Bandits that are being chosen between
    :param temp: temperature factor, controls how much disorder is in the choice of the bandit
    :param reverse: select the lowest arm, rather than highest
    :return: single Bandit
    '''
    expectations = [arm.sample() for arm in arms]
    if reverse:
        expectations = [np.exp(-expect/temp) for expect in expectations]
    else:
        expectations = [np.exp(expect/temp) for expect in expectations]
    normalize = np.sum(expectations)
    probabilities = [(expect)/normalize for expect in expectations]
    return np.random.choice(arms,1, p=probabilities)[0]

def boltzmann_ranking(arms,temp):
    '''
    :param arms: list of Bandits that are being chosen between
    :param temp: temperature factor, controls how much disorder is in the choice of the bandit
    :return: single Bandit
    '''
    for arm in arms:
        arm.calc_sigma()
    expectations = [arm.sigma for arm in arms]
    expectations = [np.exp(-expect/temp) for expect in expectations]
    normalize = np.sum(expectations)
    probabilities = [(expect)/normalize for expect in expectations]
    return np.random.choice(arms,1, p=probabilities)[0]

def greedy_ranking(arms,av):
    expectations = [arm.sample() for arm in arms]
    expectations = [np.abs(e - av) for e in expectations]
    normalize = np.sum(expectations)
    probabilities = [(expect)/normalize for expect in expectations]
    minimum = probabilities.index(min(probabilities))
    return arms[minimum]
