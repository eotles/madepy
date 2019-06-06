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
    model.arValueIteration(0.01)
    model.arLinearProgramming_Dual(True)
    model.arLinearProgramming_Primal(True)

    S = ['s1','s2','s3']
    A = {'s1': ['a11','a12'],
         's2': ['a21'],
         's3': ['a31']}
    r_t = {mdp.stateAndAction('s1', 'a11'): 0,
           mdp.stateAndAction('s1', 'a12'): 0,
           mdp.stateAndAction('s2', 'a21'): 3,
           mdp.stateAndAction('s3', 'a31'): 4}
    r_N = None
    p =   {mdp.stateAndAction('s1', 'a11'): {'s1':0.5,'s2':0.5,'s3':0},
           mdp.stateAndAction('s1', 'a12'): {'s1':2.0/3.0,'s3':1.0/3.0,'s3':0},
           mdp.stateAndAction('s2', 'a21'): {'s1':1,'s2':0,'s3':0},
           mdp.stateAndAction('s3', 'a31'): {'s3':1,'s2':0,'s3':0}}
    l = 0.9

    denardo = mdp.model(N, S, A, r_t, r_N, p, l)
    #denardo.arValueIteration(1)
    denardo.arLinearProgramming_Primal(False)
    denardo.arLinearProgramming_Dual(False)


if __name__ == '__main__':
    main()
