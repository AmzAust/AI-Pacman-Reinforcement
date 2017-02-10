# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
          
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(0, iterations):
          myCounter = util.Counter()
          for state in mdp.getStates():
            if not mdp.isTerminal(state):
                maxOfVal = float("-infinity")
                for action in mdp.getPossibleActions(state): # Obtain possible actions in state
                    aggregate = 0
                    for nextState, probability in mdp.getTransitionStatesAndProbs(state,action):
                        sumCalc = (self.values[nextState] * discount) + (mdp.getReward(state, action, nextState))
                        probabilityCalc = probability * sumCalc # Calculate probability with rewards and discounts
                        aggregate += probabilityCalc
                    maxOfVal = max(maxOfVal, aggregate) # Get max value
                    myCounter[state] = maxOfVal # Insert value into counter
            else:
              myCounter[state] = 0 # If it is a terminal state, insert 0 into counter
          self.values = myCounter

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        aggregate = 0
        for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
          sumCalc = (self.mdp.getReward(state, action, nextState) + (self.values[nextState] * self.discount))
          probabilityCalc = probability * sumCalc # Calculate probability from rewards and discounts
          aggregate += probabilityCalc
        return aggregate

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if not self.mdp.isTerminal(state):
            pass
        else:
          return None # Return none if in a terminal state
        val = float("-infinity")
        pol = None
        for action in self.mdp.getPossibleActions(state):
          tempQval = self.computeQValueFromValues(state, action) # Compute Q for state and action
          if tempQval < val: # If Q value is less than Q value, pass over
              pass
          else: # If Q value is greater or equal, store to value, and store action to policy
            pol = action
            val = tempQval
        return pol

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
