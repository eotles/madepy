'''
Created in Fall 2014

@author: eotles

Based on MDP algorithms presented in Martin L. Putterman's 
Markov Decision Processes.
'''

import collections
import math
import numpy as np
import sys
#check if computer has gurobi installed
try:
    import gurobipy as guro
    HAVEGUROBI = True
except ImportError:
    HAVEGUROBI = False

stateAndAction = collections.namedtuple("stateAndAction", ['state', 'action'])

#badly formed data exception
class badlyFormedDataException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

#improper time horizon exception
class improperTimeHorizonException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class model(object):
    def __init__(self, N, S, A, r_t, r_N, p, l):
        self.N = N
        if(N == float("inf")):
            self.infHoriz = True
            self.l = l
        else:
            self.infHoriz = False
            self.r_N = r_N
        self.S = S
        self.A = A
        self.r_t = r_t
        self.p = p
        
    #TODO: Implement this!
    def backwardInduction(self):
        if self.infHoriz: 
            raise(improperTimeHorizonException("Try An Infinite Time Method"))
      
    def valueIteration(self, epsilon):
        if (not self.infHoriz): 
            raise(improperTimeHorizonException("Try A Finite Time Method"))
        #1.  select v_0 in V, specify E>0, set n=0
        v = dict()
        d = dict()
        n = 0        
        convergenceLimit = epsilon*(1-self.l)/(2*self.l)
        v.update({n: self._initV0()})
        
        doStep2 = True
        while(doStep2):
            n+=1
            #2.  for each s in S, compute v_n+1 = max_wrt_a_in_As(r(s,a) + 
            #                            sum_over_j_in_S(l*p(j|s,a)*v_n(j))
            tempV = dict()
            tempD = dict()
            for state in self.S:
                #find max reward action
                maxReward = sys.float_info.min
                argMaxReward = None                
                for action in self.A.get(state):
                    saPair = stateAndAction(state, action)
                    reward = self.r_t.get(saPair)
                    for nextState, transProb in self.p.get(saPair).iteritems():
                        reward += self.l * transProb * v.get(n-1).get(nextState)
                    if(reward > maxReward):
                        maxReward = reward
                        argMaxReward = action
                tempV.update({state: maxReward})
                tempD.update({state: argMaxReward})
            v.update({n: tempV})
            d.update({n: tempD})
                
            #3.  if ||v_n+1 - v_n|| < E(1-l)/2l goto: 4 else n++ goto 2
            l2Norm = self._l2Norm(v.get(n), v.get(n-1))
            if(l2Norm < convergenceLimit):
                doStep2 = False
        
        #4.  for_sInS choose d_E(s) belong to argmax_wrt_a_in_As(r(s,a) + 
        #                            sum_over_j_in_S(l*p(j|s,a)*v_n+1(j))
        print("Value Iteration (E: %s)" %(epsilon))
        self._printPolicy(d, v, n)
          
    def policyIteration(self):
        if (not self.infHoriz): 
            raise(improperTimeHorizonException("Try A Finite Time Method"))
        #1.  set n = 0 and select arbitrary decision rule d_0 in D
        v = dict()
        d = dict()
        n = 0
        
        tempD = dict()
        for state in self.S:
            tempD.update({state: self.A.get(state)[0]})
        d.update({n: tempD})
        
        improvePolicy = True
        while improvePolicy:
            #2.  (Policy evaluation) Obtain v_n by solving: (I-l*P_dn)*v = r_dn
            P_dn = dict()
            matrix_P_dn = list()
            vector_R_dn = list()
            for state in self.S:
                saPair = stateAndAction(state, d.get(n).get(state))
                row = list()
                for colState in self.S:
                    row.append(self.p.get(saPair).get(colState))
                P_dn.update({saPair: self.p.get(saPair)})
                matrix_P_dn.append(row)
                vector_R_dn.append(self.r_t.get(saPair))
            
            matrix_P_dn = np.matrix(matrix_P_dn)
            vector_R_dn = np.matrix(vector_R_dn).transpose()
            
            matrix_tempV=((matrix_P_dn**0 - self.l*matrix_P_dn)**-1)*vector_R_dn
            matrix_tempV= np.array(matrix_tempV).tolist()
            
            tempV2 = dict()
            for index, state in enumerate(self.S):
                tempV2.update({state: matrix_tempV[index][0]})
            
            v.update({n: tempV2})
                
            #3.  (Policy improvement) Choose d_n+1 to satisfy: d_n+1 belong to 
            #    argmax_d_in_D(r_d + l*P_d*v_n), setting d_n+1=d_n if possible
            piResults = self._policyImprovement(v, n)
            d.update({n+1: piResults[1]})      
            #4.  If d_n+1 = d_n and set d*=d_n, otherwise n++ and goto step 2
            if(d.get(n+1) == d.get(n)):
                improvePolicy = False
            else:
                n+=1
        
        print("Policy Iteration")
        self._printPolicy(d, v, n)
              
    def modifiedPolicyIteration(self, epsilon, m):
        if (not self.infHoriz): 
            raise(improperTimeHorizonException("Try A Finite Time Method"))        
        #1.  select v_0 in V, specify E>0, set n=0
        v = dict()
        d = dict()
        u = dict()
        n = 0
        v.update({n: self._initV0()})
        convergenceLimit = epsilon*(1-self.l)/(2*self.l)
        
        
        improvePolicy = True
        while(improvePolicy):
            #2.  (Policy Improvment_ choose d_n+1 to satisfy d_n+1 belong to 
            #    argmax_d_in_D(r_d + l*P_d*v_n), setting d_n+1=d_n if possible
            piResults = self._policyImprovement(v, n)
            d.update({n+1: piResults[1]}) 
            
            #3.  (Partial policy evaluation)
            #3.a set k=0 and u_0_n = max(rd + l*Pd*v_n)
            u.update({(n,0): piResults[0]})
            
            #3.b if ||u_0_n - v_n|| < E(1-l)/(2l) go to step 4, else go to c
            if(self._l2Norm(u.get((n,0)), v.get(n)) < convergenceLimit):
                improvePolicy = False
                break
            else:
                #3c. If k = m,, go to (e). Otherwise, compute u_n_k+l 
                #        by u_n_k+1 = r_dn+1 + l*P_dn+1 = L_dn+1 * u_k_n
                for k in range(0,m):
                    #3c.  Otherwise, compute u_n_k+l by 
                    #        u_n_k+1 = r_dn+1 + l*P_dn+1 = L_dn+1 * u_k_n
                    #3d. Increment k by 1 and return to (c).
                    tempU = dict()
                    for state in self.S:
                        saPair = stateAndAction(state, d.get(n+1).get(state))
                        u_n_k1 = self.r_t.get(saPair)
                        for nextState,transProb in self.p.get(saPair).iteritems():
                            u_n_k1+=self.l*transProb*u.get((n,k)).get(nextState)
                        tempU.update({state: u_n_k1})
                    u.update({(n,k+1): tempU})
                #e. Set v_n+1 = u_n_mn, increment n by 1, and go to step 2.
                v.update({n+1: u.get((n,m))})
                n+=1
        #4. Set d_E = d_n+1, and stop.
        print("Modified Policy Iteration (E: %s, m: %s)" %(epsilon, m))
        self._printPolicy(d, v, n)
                
    #LP Primal Model Provided by Putterman's Markov Decision Processes (p223)
    #    min a*v
    #    s.t.    v(s) - sum_jInS(l*p(j|s,a)*v(j)) >= r(s,a) for_aInA_sInS
    #            v(s) unconstrained
    #Parameters:
    #    outputFlag (boolean): print gurobi solver output
    #    alpha (dict(state:float)) :alpha vector
    #Return:
    #    None
    def linearProgramming_Primal(self, outputFlag=False, alpha=None):
        if (not self.infHoriz): 
            raise(improperTimeHorizonException("Try A Finite Time Method"))
        if (not HAVEGUROBI): raise(ImportError("Unable to import gurobipy"))
        alpha = self._checkAlpha(alpha)
        
        try:
            #create a new model for primal lp MDP
            plpMDP = guro.Model("MDP (Primal)")
            if(not outputFlag): plpMDP.setParam( 'OutputFlag', outputFlag)
            #create variables
            v = dict()
            for state in self.S:
                v.update({state: plpMDP.addVar()})
            #update
            plpMDP.update()
            #set objective
            plpMDP.setObjective(guro.quicksum(plpMDP.getVars()), guro.GRB.MINIMIZE)
            #add constraints
            plpCons = dict()
            for state in self.S:
                for action in self.A.get(state):
                    saPair = stateAndAction(state, action)
                    tempLinExp = guro.LinExpr(v.get(state))
                    for nextState, transProb in self.p.get(saPair).iteritems():
                        tempLinExp.addTerms(-self.l * transProb, v.get(nextState))
                    plpCons.update({saPair: plpMDP.addConstr(tempLinExp, guro.GRB.GREATER_EQUAL, self.r_t.get(saPair))})      
            #solve
            if(outputFlag): print("Solving MDP (Primal Form)...")
            plpMDP.optimize()
    
            #get policy (d)
            d = dict()
            for state in self.S:
                for action in self.A.get(state):
                    saPair = stateAndAction(state, action)
                    if(plpCons.get(saPair).slack >= 0):
                        break
                d.update({state: action})
            
            print("Linear Programming (Primal Form)")
            for state in self.S:
                print("State: %s, Decision: %s, Value: %f" %(state, d.get(state), v.get(state).x))
            print('')
            
        except guro.GurobiError:
            print('Gurobi Reported an error')

    #TODO: Action not in Action Space for State Protection
    #LP Dual Model Provided by Putterman's Markov Decision Processes (p224)
    #    max r*x
    #    s.t.    sum_aInA(x(j,a)) - sum_sInS_aInA(l*p(j|s,a)*x(s,a)) = a(j) for_jInS
    #            x(s,a) >= 0
    #Parameters:
    #    outputFlag (boolean): print gurobi solver output
    #    alpha (dict(state:float)) :alpha vector
    #Return:
    #    None
    def linearProgramming_Dual(self, outputFlag, alpha=None):
        if (not self.infHoriz): 
            raise(improperTimeHorizonException("Try A Finite Time Method"))
        if (not HAVEGUROBI): raise(ImportError("Unable to import gurobipy"))
        alpha = self._checkAlpha(alpha)
        
        try:
            #create a new model for dual lp MDP
            dlpMDP = guro.Model("MDP (Dual)")
            if(not outputFlag): dlpMDP.setParam( 'OutputFlag', outputFlag)
            #create variables
            x = dict()
            for state in self.S:
                for action in self.A.get(state):
                    saPair = stateAndAction(state, action)
                    x.update({saPair: dlpMDP.addVar()})
            dlpMDP.update()
            #set objective
            tempLinExp = guro.LinExpr()
            for state in self.S:
                for action in self.A.get(state):
                    saPair = stateAndAction(state,action)
                    tempLinExp.addTerms(self.r_t.get(saPair), x.get(saPair))
            dlpMDP.setObjective(tempLinExp, guro.GRB.MAXIMIZE)
            #add constraints
            dlpCons = dict()
            for jstate in self.S:
                tempLinExp = guro.LinExpr()
                for action in self.A.get(jstate):
                    jaPair = stateAndAction(jstate, action)
                    tempLinExp.addTerms(1, x.get(jaPair))
                for state in self.S:
                    for action in self.A.get(state):
                        saPair = stateAndAction(state,action)
                        tempLinExp.addTerms(-self.l*self.p.get(saPair).get(jstate), x.get(saPair))
                dlpCons.update({jstate: dlpMDP.addConstr(tempLinExp, guro.GRB.EQUAL, alpha.get(jstate))})
            #solve
            if(outputFlag): print("Solving MDP (Dual Form)...")
            dlpMDP.optimize()
            
            #get policy (d) and values (v)
            d = dict()
            v = dict()
            for state in self.S:
                for action in self.A.get(state):
                    saPair = stateAndAction(state, action)
                    if(x.get(saPair).x >= 0): break
                d.update({state: action})
                v.update({state: dlpCons.get(state).Pi})
            
            print("Linear Programming (Dual Form)")
            for state in self.S:
                print("State: %s, Decision: %s, Value: %f" %(state, d.get(state), v.get(state)))
            print('')
        
        except guro.GurobiError:
            print('Gurobi Reported an error')
                
                  
    def _policyImprovement(self, v, n):
        tempV = dict()
        tempD = dict()
        for state in self.S:
            #find max reward action
            maxReward = sys.float_info.min
            argMaxReward = None  
            for action in self.A.get(state):
                saPair = stateAndAction(state, action)
                reward = self.r_t.get(saPair)
                for nextState, transProb in self.p.get(saPair).iteritems():
                        reward += self.l * transProb * v.get(n).get(nextState)
                if(reward > maxReward):
                    maxReward = reward
                    argMaxReward = action
            tempV.update({state: maxReward})
            tempD.update({state: argMaxReward})
        return((tempV,tempD))
    
    def _checkAlpha(self, alpha):
        if(alpha==None):
            alpha = dict()
            for state in self.S:
                alpha.update({state: 1})
        else:
            for state in self.S:
                try: float(alpha.get(state))
                except: 
                    raise(badlyFormedDataException("Alpha poorly formed.  Need a value for every state."))
        return(alpha)
    
    def _initV0(self):
        tempV = dict()
        for state in self.S:
            tempV.update({state: sys.float_info.min})
        return(tempV)
    
    def _l2Norm(self, vDict1, vDict2):
        distance = 0
        for state in self.S:
            distance += (vDict1.get(state)-vDict2.get(state))**2
        return(math.sqrt(distance))

    def _printPolicy(self, d, v, n):
        for state in self.S:
            print("State: %s, Decision: %s, Value: %f" %(state, d.get(n).get(state), v.get(n).get(state)))
        print('')
          
    def arValueIteration(self, epsilon):
        if (not self.infHoriz): 
            raise(improperTimeHorizonException("Try A Finite Time Method"))
        #1.  select v_0 in V, specify E>0, set n=0
        v = dict()
        d = dict()
        n = 0        
        #convergenceLimit = epsilon*(1-self.l)/(2*self.l)
        v.update({n: self._initV0()})
        
        doStep2 = True
        while(doStep2):
            n+=1
            #2.  for each s in S, compute v_n+1 = max_wrt_a_in_As(r(s,a) + 
            #                            sum_over_j_in_S(l*p(j|s,a)*v_n(j))
            tempV = dict()
            tempD = dict()
            for state in self.S:
                #find max reward action
                maxReward = sys.float_info.min
                argMaxReward = None                
                for action in self.A.get(state):
                    saPair = stateAndAction(state, action)
                    reward = self.r_t.get(saPair)
                    for nextState, transProb in self.p.get(saPair).iteritems():
                        reward += transProb * v.get(n-1).get(nextState)
                    if(reward > maxReward):
                        maxReward = reward
                        argMaxReward = action
                tempV.update({state: maxReward})
                tempD.update({state: argMaxReward})
            v.update({n: tempV})
            d.update({n: tempD})
                
            #3.  if sp(v_n+1 - v_n) < E goto: 4 else n++ goto 2
            span = self._spanOfDiff(v.get(n), v.get(n-1))
            if(span < epsilon):
                doStep2 = False
        
        #4.  for_sInS choose d_E(s) belong to argmax_wrt_a_in_As(r(s,a) + 
        #                            sum_over_j_in_S(l*p(j|s,a)*v_n+1(j))
        print("Average Reward Value Iteration (E: %s)" %(epsilon))
        self._printPolicy(d, v, n)
 
    #TODO: Explanations
    #LP Primal Model Provided by Putterman's Markov Decision Processes (p223)
    #    min a*v
    #    s.t.    v(s) - sum_jInS(l*p(j|s,a)*v(j)) >= r(s,a) for_aInA_sInS
    #            v(s) unconstrained
    #Parameters:
    #    outputFlag (boolean): print gurobi solver output
    #Return:
    #    None
    def arLinearProgramming_Primal(self, outputFlag=False):
        if (not self.infHoriz): 
            raise(improperTimeHorizonException("Try A Finite Time Method"))
        if (not HAVEGUROBI): raise(ImportError("Unable to import gurobipy"))
        
        try:
            #create a new model for primal lp MDP
            plpMDP = guro.Model("MDP (Primal)")
            if(not outputFlag): plpMDP.setParam( 'OutputFlag', outputFlag)
            #create variables
            g = plpMDP.addVar()
            h = dict()
            for state in self.S:
                h.update({state: plpMDP.addVar()})
            #update
            plpMDP.update()
            #set objective
            plpMDP.setObjective(g, guro.GRB.MINIMIZE)
            #add constraints
            plpCons = dict()
            for state in self.S:
                for action in self.A.get(state):
                    saPair = stateAndAction(state, action)
                    tempLinExp = guro.LinExpr()
                    tempLinExp.addTerms(1, g)
                    tempLinExp.addTerms(1, h.get(state))
                    for nextState, transProb in self.p.get(saPair).iteritems():
                        tempLinExp.addTerms(-transProb, h.get(nextState))
                    plpCons.update({saPair: plpMDP.addConstr(tempLinExp, guro.GRB.GREATER_EQUAL, self.r_t.get(saPair))})      
            #solve
            if(outputFlag): print("Solving AR MDP (Primal Form)...")
            plpMDP.optimize()
    
            #get policy (d)
            d = dict()
            for state in self.S:
                for action in self.A.get(state):
                    saPair = stateAndAction(state, action)
                    if(plpCons.get(saPair).slack >= 0):
                        break
                d.update({state: action})
            
            print("Average Reward Linear Programming (Primal Form)")
            print("Gain: %f" %(g.x))
            for state in self.S:
                print("State: %s, Decision: %s, Bias: %f" %(state, d.get(state), h.get(state).x))
            print('')
            
        except guro.GurobiError:
            print('Gurobi Reported an error')   
    
    #TODO: Explanations
    #LP Dual Model Provided by Putterman's Markov Decision Processes (p224)
    #    
    #Parameters:
    #    outputFlag (boolean): print gurobi solver output
    #Return:
    #    None
    def arLinearProgramming_Dual(self, outputFlag):
        if (not self.infHoriz): 
            raise(improperTimeHorizonException("Try A Finite Time Method"))
        if (not HAVEGUROBI): raise(ImportError("Unable to import gurobipy"))
        
        try:
            #create a new model for dual lp MDP
            dlpMDP = guro.Model("MDP (Dual)")
            if(not outputFlag): dlpMDP.setParam( 'OutputFlag', outputFlag)
            #create variables
            x = dict()
            for state in self.S:
                for action in self.A.get(state):
                    saPair = stateAndAction(state, action)
                    x.update({saPair: dlpMDP.addVar()})
            dlpMDP.update()
            #set objective
            tempLinExp = guro.LinExpr()
            for state in self.S:
                for action in self.A.get(state):
                    saPair = stateAndAction(state,action)
                    tempLinExp.addTerms(self.r_t.get(saPair), x.get(saPair))
            dlpMDP.setObjective(tempLinExp, guro.GRB.MAXIMIZE)
            #add constraints
            dlpCons = dict()
            for jstate in self.S:
                tempLinExp = guro.LinExpr()
                for action in self.A.get(jstate):
                    jaPair = stateAndAction(jstate, action)
                    tempLinExp.addTerms(1, x.get(jaPair))
                for state in self.S:
                    for action in self.A.get(state):
                        saPair = stateAndAction(state,action)
                        p = self.p.get(saPair).get(jstate) if(self.p.get(saPair).get(jstate) != None) else 0
                        tempLinExp.addTerms(-p, x.get(saPair))
                dlpCons.update({jstate: dlpMDP.addConstr(tempLinExp, guro.GRB.EQUAL, 0)})
            tempLinExp = guro.LinExpr()
            for state in self.S:
                for action in self.A.get(state):
                    saPair = stateAndAction(state, action)
                    tempLinExp.addTerms(1, x.get(saPair))
            dlpCons.update({'last': dlpMDP.addConstr(tempLinExp, guro.GRB.EQUAL, 1)})         
            
            #solve
            if(outputFlag): print("Solving AR MDP (Dual Form)...")
            dlpMDP.optimize()
            
            #get policy (d) and bias (h)
            g = dlpMDP.ObjVal
            d = dict()
            h = dict()
            for state in self.S:
                for action in self.A.get(state):
                    saPair = stateAndAction(state, action)
                    if(x.get(saPair).x >= 0): break
                d.update({state: action})
                h.update({state: dlpCons.get(state).Pi})
            
            #print(dlpCons)
            #print(x)
            
            print("Average Reward Linear Programming (Dual Form)")
            print("Gain: %f" %(g))
            for state in self.S:
                print("State: %s, Decision: %s, Bias: %f" %(state, d.get(state), h.get(state)))
            print('')
        
        except guro.GurobiError:
            print('Gurobi Reported an error')
            
            
    def _spanOfDiff(self, vDict1, vDict2):
        maxDiff = sys.float_info.min
        minDiff = sys.float_info.max
        for state in self.S:
            diff = vDict1.get(state)-vDict2.get(state)
            if(diff > maxDiff): 
                maxDiff = diff
            if(diff < minDiff): 
                minDiff = diff
        return(maxDiff-minDiff)