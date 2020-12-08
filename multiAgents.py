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
        
        
        "*** YOUR CODE HERE ***"
        score = 0.
        for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
            distance = manhattanDistance(newPos, ghost.getPosition())
            if (distance < scaredTime):
                score += 5. / (1 + distance * distance)
            elif(scaredTime <= 1 and distance <= 2):
                score -= 5. / (1 + distance * distance)
        foods = newFood.asList()
        for food in foods:
            distance = manhattanDistance(newPos, food)
            score += 1. / (1. + distance)
        score -= len(foods)
        score -= len(newGhostStates) * 10
        if (manhattanDistance(currentGameState.getPacmanPosition(), newPos) > 0):
            score += 1
        return score + random.random()/10.

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
        "*** YOUR CODE HERE ***"
        best, bestAct = None, 'STOP'
        for action in gameState.getLegalActions():
            nextState = gameState.generateSuccessor(0, action)
            score = self.minimax(nextState, 1, self.depth)
            if (best == None or best < score):
                best = score
                bestAct = action
                
        return bestAct
        
    def minimax(self, gameState, agent, depth):
        """
        Helper function for minimax agent.
        
        """
        best = None
        if (gameState.isLose() or gameState.isWin() or (depth<=0)):
            return self.evaluationFunction(gameState)
        
        numAgents = gameState.getNumAgents()
        if (agent >= numAgents - 1):
            depth -= 1
            agent = numAgents - 1
            
        for action in gameState.getLegalActions(agent):
            nextState = gameState.generateSuccessor(agent, action)
            score = self.minimax(nextState, (agent+1)%numAgents, depth)
            if (best == None):
                best = score
            if (agent == 0):
                best = max(best, score)
            else:
                best = min(best, score)
        
        return (best)    
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        best = None
        a = -1e9
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            score = self.alphabeta(nextState, 1 % nextState.getNumAgents(),
                self.depth, a, 1e9)
                
            if (a <= score):
                best = action
                a = score
                
        return best
        
    def alphabeta(self, gameState, agent, depth, a, b):
        
        if (gameState.isLose() or gameState.isWin() or (depth<=0)):
            return self.evaluationFunction(gameState)
        
        value = 0
        numAgents = gameState.getNumAgents()
        if (agent == 0):
            value = -1e9
            for action in gameState.getLegalActions(agent):
                nextState = gameState.generateSuccessor(agent, action)
                value = max(value, self.alphabeta(nextState, 
                    (agent+1)%numAgents,
                    depth, a, b))
                a = max(a, value)
                if (a > b):
                    break
        else:
            value = 1e9
            for action in gameState.getLegalActions(agent):
                nextState = gameState.generateSuccessor(agent, action)
                value = min(value, self.alphabeta(nextState,
                    (agent+1)%numAgents,
                    depth - int(agent+1 == numAgents),
                    a, b))
                b = min(b, value)
                if (b < a):
                    break
        
        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        best, bestAct = None, 'STOP'
        for action in gameState.getLegalActions():
            nextState = gameState.generateSuccessor(0, action)
            score = self.expectimax(nextState, 1, self.depth)
            if (best == None or best < score):
                best = score
                bestAct = action
                
        return bestAct

    def expectimax(self, gameState, agent, depth):
        """
        Helper function for expectimax agent.
        
        """
        if (gameState.isLose() or gameState.isWin() or (depth<=0)):
            return self.evaluationFunction(gameState)
        
        numAgents = gameState.getNumAgents()
            
        value, best = 0., -1e9
        n = 0
        for action in gameState.getLegalActions(agent):
            nextState = gameState.generateSuccessor(agent, action)
            score = self.expectimax(nextState, 
                (agent+1)%numAgents, 
                depth - int(agent+1 == numAgents))
                
            value += score
            best = max(best, score)
            n += 1
        
        return (best) if (agent == 0) else value / n

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacman = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostStates()
    foods  = currentGameState.getFood().asList()
    scared = [ghost.scaredTimer for ghost in ghosts]
    capsules = currentGameState.getCapsules()
    
    score = 0.
    for ghost, scaredTime in zip(ghosts, scared):
        distance = manhattanDistance(pacman, ghost.getPosition())
        if (scaredTime > 1):
            score += 1. / (1 + distance)
        elif(scaredTime < 1 and distance <= 2):
            score -= 4
        elif(scaredTime <= 1):
            score += 2

    for food in foods:
        distance = manhattanDistance(pacman, food)
        score += 1. / (1. + distance)
        
    for capsule in capsules:
        distance = manhattanDistance(pacman, capsule)
        score += 2. / (1. + distance)
        
    score -= len(foods)
    score -= len(capsules)*3
    return score + random.random()/20.

# Abbreviation
better = betterEvaluationFunction
