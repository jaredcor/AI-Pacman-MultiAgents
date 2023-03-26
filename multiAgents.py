# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        # Few things that you might need
        newFood = newFood.asList() # list of all foods
        GhostPosition = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates] # list of all ghost positions
        closestFoodDistance = sorted(newFood, key=lambda fDist: manhattanDistance(fDist, newPos)) # Distance to the clostest food 
        closestGhostDistance = sorted(GhostPosition, key=lambda gDist: manhattanDistance(gDist, newPos)) # Distance to the clostest ghost
        
        "*** CS3368 YOUR CODE HERE ***"
        "Decribe your function:"
        """ What to do: 
        1. check for failure: return -1 if the ghosts are not scared and you have failed
        2. check if you have eaten a food, return 1 in this case
        3. Use the distance to closest dot distance and the  distance to the closest ghost to estimate an evaluation function
        """
        ghostScared = min(newScaredTimes) > 0
        # 1. check for failure
        if not ghostScared and (newPos in GhostPosition): # return -1 if the ghosts are not scared and pacman has failed
            return -1
        # 2. check if pac has eaten a food
        if newPos in currentGameState.getFood().asList():
            return 1
        
        # 3. use closestFoodDistance and closestGhostDistance to estimate an evaluation func
        # manhattan distance from closest food/ghost to current pac position
        # return util.manhattanDistance(closestFoodDistance[0], newPos) - util.manhattanDistance(closestGhostDistance[0], newPos)
        # trying the reciprocal of these values rather than just the values themselves
        return 1.0/manhattanDistance(closestFoodDistance[0], newPos) - 1.0/manhattanDistance(closestGhostDistance[0], newPos)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** CS3368 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"
        '''
        recursive implimentation 
        If it is a terminal return it value using self.evaluationFunction(state)
        If a min, called your implimented min function
        If a max, called your implimented max function
        '''
        # get all ghost index numbers
        # start at 1 bc pacman is agent 0
        ghostId = [i for i in range(1, gameState.getNumAgents())]
        
        # terminal conditions
        def value(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        # min function
        # search for ghosts' actions that cause Pacman minimum score 
        def min_function(state, depth, ghost):
            if value(state, depth):
                return self.evaluationFunction(state)
            
            val = float("inf")
            for action in state.getLegalActions(ghost):
                # if current ghost is the last one
                if ghost == ghostId[-1]:
                    # search for pacman action to generate the max value
                    maxVal = max_function(state.generateSuccessor(ghost, action), depth + 1)
                    val = min(val, maxVal)
                else:
                    # else find the minimum
                    minVal = min_function(state.generateSuccessor(ghost, action), depth, ghost + 1)
                    val = min(val, minVal)
            
            return val
        
        # max function
        # search for Pacmans max value action
        def max_function(state, depth):
            if value(state, depth):
                return self.evaluationFunction(state)
            
            val = float("-inf")
            for action in state.getLegalActions(0):
                minVal = min_function(state.generateSuccessor(0, action), depth, ghostId[0])
                val = max(val, minVal)
            
            return val
        
        pacActions = gameState.getLegalActions(0)
        # pair action and value from that action -> (action, value)
        results = [(action, min_function(gameState.generateSuccessor(0, action), 0, ghostId[0])) for action in pacActions]
        # sort by value 
        results.sort(key=lambda k: k[1])
        # return maximum value
        return results[-1][0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** CS5368 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"
        # follow alpha-beta pseudocode in class slides
        
        currVal = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        nextPacAction = Directions.STOP
        
        actions = gameState.getLegalActions(0).copy()
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            nextVal = self.value(nextState, 0, 1, alpha, beta)
            
            if nextVal > currVal:
                currVal = nextVal
                nextPacAction = action
                
            alpha = max(alpha, currVal)
        return nextPacAction    
    
    def value(self, gameState, depth = 0, agentId = 0, alpha = float("-inf"), beta = float("inf")):
        # using value function to choose best action
        maxList = [0, ]
        minList = list(range(1, gameState.getNumAgents()))
        # if terminal            
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        elif agentId in maxList:
            return self.max_function(gameState, depth, agentId, alpha, beta)
        elif agentId in minList:                
            return self.min_function(gameState, depth, agentId, alpha, beta)
        
    # alpha part: search for maximums 
    # def max-value(state, alpha, beta) from slides
    def max_function(self, gameState, depth, agentId, alpha = float("-inf"), beta = float("inf")):
        # initialize v = -inf
        val = float("-inf")
        actions = gameState.getLegalActions(agentId)
        # for each successor of state
        for i, action in enumerate(actions):
            nextVal = self.value(gameState.generateSuccessor(agentId, action), depth, agentId + 1, alpha, beta)
            # v = max(v, value(successor, alpha, beta))
            val = max(val, nextVal)
            # if v >= beta return v
            if val > beta:
                return val
            # alpha = max(alpha ,v)
            alpha = max(alpha, val)
        # return v
        return val
            
    # beta part: search for minimums
    # def min-value(state, alpha, beta)
    def min_function(self, gameState, depth, agentId, alpha = float("-inf"), beta = float("inf")):
        # initialize v = +inf
        val = float("inf")
        actions = gameState.getLegalActions(agentId)
        # for each successor of state
        for i, action in enumerate(actions):
            # if last agent
            if agentId == gameState.getNumAgents() - 1:
                nextVal = self.value(gameState.generateSuccessor(agentId, action), depth + 1, 0, alpha, beta)
                # v = min(v, value(successor, alpha, beta))
                val = min(val, nextVal)
                # if v <= alpha return v
                if val < alpha:
                    return val
            else:
                nextVal = self.value(gameState.generateSuccessor(agentId,action), depth, agentId + 1, alpha, beta)
                val = min(val, nextVal)
                if val < alpha:
                    return val
            beta = min(beta, val)
        return val

