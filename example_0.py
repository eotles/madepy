'''
Markov Decision Processes
Homework #3
Created Oct. 2014

@author: eotles
'''
import collections
import math
import sys
stateAndAction = collections.namedtuple("stateAndAction", ['state', 'action'])


#Puterman 4.2 a,b
def p1():
    N = 3
    S = ['s_1','s_2']
    A = {'s_1': ['a_1,1','a_1,2'],
         's_2': ['a_2,1']}

    r_t = {stateAndAction('s_1', 'a_1,1'): 5,
           stateAndAction('s_1', 'a_1,2'): 10,
           stateAndAction('s_2', 'a_2,1'): -1}
    r_N = {'s_1': 0,
           's_2': 0}
    p =   {stateAndAction('s_1', 'a_1,1'): {'s_1': 0.5, 's_2': 0.5},
           stateAndAction('s_1', 'a_1,2'): {'s_1': 0,   's_2': 1},
           stateAndAction('s_2', 'a_2,1'): {'s_1': 0,   's_2': 1}}

    gamma = 0.2

    #step 1
    t = N
    u_star = [dict() for _ in range(N+1)]
    a_star = [dict() for _ in range(N)]
    u_star[N] = r_N
    #step 2 and 3
    while(t>1):
        t = t-1
        #for each s find maximizing a
        for s in S:
            maxReward = -sys.float_info.max
            for a in A.get(s):
                s_a = stateAndAction(s, a)
                #calculate reward
                reward = math.e**(r_t.get(s_a) * gamma)
                nsReward = 0
                for j in S:
                    nsReward += math.e**(p.get(s_a).get(j)*u_star[t+1].get(j) * gamma)
                reward *= nsReward
                if(reward > maxReward):
                    maxReward = reward
                    u_star[t].update({s: reward})
                    a_star[t].update({s: a})
    print('Problem 4.2\na.  See appended code for alg. implementation.  A BA '+
          'for problem when reward is f(x) of current state, current action, '+
          'subsequent state: the addition between the reward for the current '+
          'state being replaced with multiplication.  The alg. becomes '+
          'simplified if it we don\'t depend on subsequent state in step 2 '+
          'all that needs to be done for each step 2 is max{r(s,a)}\nb.  ')
    for t in xrange(1,3):
        print("    Time %s" %(t))
        for s in S:
            print("\t%s:\t%s\t%s" %(s, a_star[t].get(s), u_star[t].get(s)))


#Puterman 4.6
def p2():
    N = 4
    S = ['0_1', '1_2', '1_3', '1_4', '2_5', '2_6', '3_5', '3_6', '3_7', '4_7',
         '5_8', '6_8', '7_8']
    A = {'0_1': ['1_2', '1_3', '1_4'],
         '1_2': ['2_5', '2_6'], '1_3': ['3_5', '3_6', '3_7'], '1_4': ['4_7'],
         '2_5': ['5_8'], '2_6': ['6_8'], '3_5': ['5_8'], '3_6': ['6_8'], '3_7': ['7_8'], '4_7': ['7_8'],
         '5_8': [], '6_8': [], '7_8': []}
    r_t = {stateAndAction('0_1', '1_2'): 2,
           stateAndAction('0_1', '1_3'): 4,
           stateAndAction('0_1', '1_4'): 3,
           stateAndAction('1_2', '2_5'): 4,
           stateAndAction('1_2', '2_6'): 5,
           stateAndAction('1_3', '3_5'): 5,
           stateAndAction('1_3', '3_6'): 6,
           stateAndAction('1_3', '3_7'): 1,
           stateAndAction('1_4', '4_7'): 2,
           stateAndAction('2_5', '5_8'): 1,
           stateAndAction('2_6', '6_8'): 2,
           stateAndAction('3_5', '5_8'): 1,
           stateAndAction('3_6', '6_8'): 2,
           stateAndAction('3_7', '7_8'): 6,
           stateAndAction('4_7', '7_8'): 6}
    r_N = {'5_8': 0,
           '6_8': 0,
           '7_8': 0}
    p =   {stateAndAction('0_1', '1_2'): {'1_2': 0.6, '1_3': 0.2, '1_4': 0.2},
           stateAndAction('0_1', '1_3'): {'1_2': 0.2, '1_3': 0.6, '1_4': 0.2},
           stateAndAction('0_1', '1_4'): {'1_2': 0.2, '1_3': 0.2, '1_4': 0.6},
           stateAndAction('1_2', '2_5'): {'2_5': 0.7, '2_6': 0.3},
           stateAndAction('1_2', '2_6'): {'2_5': 0.3, '2_6': 0.7},
           stateAndAction('1_3', '3_5'): {'3_5': 0.6, '3_6': 0.2, '3_7': 0.2},
           stateAndAction('1_3', '3_6'): {'3_5': 0.2, '3_6': 0.6, '3_7': 0.2},
           stateAndAction('1_3', '3_7'): {'3_5': 0.2, '3_6': 0.2, '3_7': 0.6},
           stateAndAction('1_4', '4_7'): {'4_7': 1},
           stateAndAction('2_5', '5_8'): {'5_8': 1},
           stateAndAction('2_6', '6_8'): {'6_8': 1},
           stateAndAction('3_5', '5_8'): {'5_8': 1},
           stateAndAction('3_6', '6_8'): {'6_8': 1},
           stateAndAction('3_7', '7_8'): {'7_8': 1},
           stateAndAction('4_7', '7_8'): {'7_8': 1}}

    #step 1
    t = N
    u_star = [dict() for _ in range(N+1)]
    a_star = [dict() for _ in range(N)]
    u_star[N] = r_N
    #step 2 and 3
    while(t>1):
        t = t-1
        #for each s find maximizing a
        for s in S:
            maxReward = sys.float_info.max
            for a in A.get(s):
                s_a = stateAndAction(s, a)
                #calculate reward
                reward = r_t.get(s_a)
                for j in u_star[t+1]:
                    if(p.get(s_a) is None or p.get(s_a).get(j) is None):
                        prob = 0
                    else:
                        prob = p.get(s_a).get(j)
                    reward += prob*u_star[t+1].get(j)
                if(reward < maxReward):
                    maxReward = reward
                    u_star[t].update({s: reward})
                    a_star[t].update({s: a})
    print('Problem 4.6\n  See appended code for alg. implementation.  '+
          'Policy seems to be pick shortest path available.')

    for t in xrange(1,N):
        print("    Time %s" %(t))
        for s in S:
                print("\t%s:\t%s\t%s" %(s,a_star[t].get(s), u_star[t].get(s)))

def backprop(N, S, A, r_N, r_t, p):
    #step 1
    t = N
    u_star = [dict() for _ in range(N+1)]
    a_star = [dict() for _ in range(N)]
    u_star[N] = r_N
    #step 2 and 3
    while(t>1):
        t -= 1
        #for each s find maximizing a
        for s in S:
            maxReward = -sys.float_info.max
            for a in A.get(s):
                s_a = stateAndAction(s, a)
                #calculate reward
                reward = r_t.get(s_a)
                for j in S:
                    #print("%s %s %s" %(t,s_a, j))
                    if(p.get(s_a) is None or p.get(s_a).get(j) is None):
                        prob = 0
                    else:
                        prob = p.get(s_a).get(j)
                    if(u_star[t+1].get(j) is not None):
                        reward += prob*u_star[t+1].get(j)
                if(reward > maxReward):
                    maxReward = reward
                    u_star[t].update({s: reward})
                    a_star[t].update({s: a})
    #print(u_star)
    for t in xrange(1,N):
        print("    Time %s" %(t))
        for s in S:
                print("\t%s:\t%s\t%s" %(s,a_star[t].get(s), u_star[t].get(s)))


#Puterman 4.21
def p3():
    N = 3
    S = ['1_su', '1_st', '2_su', '2_st']
    A = {'1_su': ['1_su', '2_su'], '1_st': ['1_su', '2_su'], '2_su': ['1_su', '2_su', '1_st', '2_st'], '2_st': ['1_su', '2_su']}
    r_t = {stateAndAction('1_su', '1_su'): 1,
           stateAndAction('1_su', '2_su'): 1,
           stateAndAction('1_st', '1_su'): 1,
           stateAndAction('1_st', '2_su'): 1,
           stateAndAction('2_su', '1_su'): 2,
           stateAndAction('2_su', '2_su'): 2,
           stateAndAction('2_su', '1_st'): 2,
           stateAndAction('2_su', '2_st'): 2,
           stateAndAction('2_st', '1_su'): 0,
           stateAndAction('2_st', '2_su'): 0
            }
    r_N = {'1_su': 1,
            '1_st': 1,
            '2_su': 2,
            '2_st': 0}
    p =   {stateAndAction('1_su', '1_su'): {'1_su': 1},
            stateAndAction('1_su', '2_su'): {'2_su': 1},
            stateAndAction('1_st', '1_su'): {'1_su': 1},
            stateAndAction('1_st', '2_su'): {'2_su': 1},
            stateAndAction('2_su', '1_su'): {'1_su': 0.5, '1_st': 0.5},
            stateAndAction('2_su', '2_su'): {'2_su': 0.5, '2_st': 0.5},
            stateAndAction('2_su', '1_st'): {'1_su': 0.5, '1_st': 0.5},
            stateAndAction('2_su', '2_st'): {'2_su': 0.5, '2_st': 0.5},
            stateAndAction('2_st', '1_su'): {'1_su': 1},
            stateAndAction('2_st', '2_su'): {'2_su': 1}}

    print('Problem 4.21\n  See appended code for alg. implementation.  ')
    backprop(N, S, A, r_N, r_t, p)

def p4():
    def makePoss(s_prime, food, pFood, pPred):
        #died looking for food
        if(s_prime<4):
            return({'0': 1})
        else:
            #found food and survived
            s_eatSurv = str(min(int(s_prime) + food, 10))
            p_eatSurv = pFood*(1-pPred)
            #no food and survived
            s_surv = s_prime
            p_surv = (1-pFood)*(1-pPred)
            #killed
            s_dead = '0'
            p_dead = pPred
            return({s_eatSurv: p_eatSurv, s_surv: p_surv, s_dead: p_dead})
    N = 20
    S = ['0','4', '5', '6', '7', '8', '9', '10']
    actions = ['0', '1', '2', '3']
    A = dict()
    for s in S:
        A.update({s : actions}) #allow animal to choose suicide for ease of coding
    r_t = dict()
    r_N = dict()
    p = dict()
    for s in S:
        val = 0 if (s == '0') else 1
        for a in actions:
            s_a = stateAndAction(s,a)
            r_t.update({s_a: val})
            if(s == '0'):
                p.update({s_a: {'0': 1}})
            else:
                poss = dict()
                s_prime = int(s)-1
                #choose death
                if(a == '0'):
                    poss = {'0': 1}
                #choose patch 1
                if(a == '1'):
                    s_prime = '0' if (s_prime<4) else str(s_prime)
                    poss = {s_prime: 1}
                #chose patch 2
                if(a == '2'):
                    poss = makePoss(s_prime, 3, .4, .004)
                if(a == '3'):
                    poss = makePoss(s_prime, 5, .5, .02)
                p.update({s_a: poss})
        r_N.update({s: val})

    #for k,v in p.iteritems():
    #    print("%s %s" %(k,v))

    print('Problem 4.28\n  See appended code for alg. implementation.  ')
    backprop(N, S, A, r_N, r_t, p)




def main():
    p1()
    p2()
    p3()
    p4()

if __name__ == '__main__':
    main()
