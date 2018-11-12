import drugbandits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random

# used for printing output
BOLD = '\033[1m'
END = '\033[0m'


bandits = []
colors = ['b','g','r','c','m','y','purple','orange']


with open('protocol.yaml', 'r') as myfile:
    simulation = myfile.read()

with open('set_of_eight.smiles','r') as f:
    index = 0
    for line in f:
        if line[0] != '#':
            details = line.split(';')
            name, exp_fe, exp_error = details[2], float(details[3]), float(details[4])
            bandits.append(drugbandits.bandit.SimulationBandit(name,exp_fe,exp_error,index,simulation,color=colors[index]))
            index += 1

print('Bandits have been initiated, now beginning muli-armed pulling protocol')

bandit_steps = 64
for step in range(1,bandit_steps+1):
    pick_bandit = drugbandit.choose.boltzmann(bandits,5,reverse=True)
    pick_bandit.pull()
    pick_bandit.update()
    
    print('{}Step {} {}'.format(BOLD,step,END))
    ax = plt.subplot(1, 1, 1)
    for bandit in bandits:
        bandit.plot()
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    plt.savefig('bayesplot{}.png'.format(step), bbox_extra_artists=(lgd,), bbox_inches='tight')

    for bandit in bandits:
        bandit.print_bandit(step)
