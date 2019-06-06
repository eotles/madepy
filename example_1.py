'''
Created in Fall 2014

@author: eotles
'''

import markovDecisionProblem as mdp

def main():
    N = float("inf")
    S = ['L','H']
    A = {'L': ['Send','DoNot'],
         'H': ['Send','DoNot']}
    r_t = {mdp.stateAndAction('L', 'Send'): 20-15,
           mdp.stateAndAction('L', 'DoNot'): 10,
           mdp.stateAndAction('H', 'Send'): 50-15,
           mdp.stateAndAction('H', 'DoNot'): 25}
    r_N = None
    p =   {mdp.stateAndAction('L', 'Send'):  {'L':0.3,'H':0.7},
           mdp.stateAndAction('L', 'DoNot'): {'L':0.5,'H':0.5},
           mdp.stateAndAction('H', 'Send'):  {'L':0.2,'H':0.8},
           mdp.stateAndAction('H', 'DoNot'): {'L':0.6,'H':0.4}}
    l = 0.9

    model = mdp.model(N, S, A, r_t, r_N, p, l)
    model.valueIteration(0.01)
    model.valueIteration(0.1)
    model.policyIteration()
    model.modifiedPolicyIteration(.01, 100)
    model.modifiedPolicyIteration(.1, 100)


if __name__ == '__main__':
    main()
