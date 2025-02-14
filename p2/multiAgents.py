"""multiAgents.py - Multi-Agent Search Algorithms for Pacman
===========================================================

This module implements various multi-agent search algorithms for the Pacman game,
including reflex agents, minimax, alpha-beta pruning, and expectimax search.

The module provides agent classes that:
- Make decisions based on state evaluation functions
- Implement adversarial search algorithms
- Model both deterministic and probabilistic opponent behavior
- Search to configurable depths using evaluation heuristics

Key Classes:
    ReflexAgent: Makes decisions using state evaluation heuristics
    MinimaxAgent: Implements minimax search algorithm
    AlphaBetaAgent: Implements alpha-beta pruning search
    ExpectimaxAgent: Implements expectimax probabilistic search

Usage:
    This module is used by the Pacman game to create AI agents. Agents can be
    selected and configured via command line arguments.

Author: George Rudolph
Date: 14 Nov 2024
Major Changes:
1. Added type hints throughout the codebase for better code clarity and IDE support
2. Improved docstrings with detailed descriptions and Args/Returns sections
3. Enhanced code organization with better function and variable naming

This code runs on Python 3.13

Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""

import random, math
import util
from util import manhattanDistance
from game import Agent, Directions
from typing import List, Tuple, Any
from pacman import GameState

class ReflexAgent(Agent):
    """A reflex agent that chooses actions by examining alternatives via a state evaluation function.
    
    This agent evaluates each possible action using a heuristic evaluation function and selects
    among the best options. The evaluation considers factors like:
    - Distance to ghosts (avoiding them)
    - Score improvements
    - Distance to food
    - Maintaining movement direction
    """

    def getAction(self, gameState: GameState) -> str:
        """Choose among the best actions according to the evaluation function.
        
        Args:
            gameState: The current game state
            
        Returns:
            str: A direction from Directions.{North, South, West, East, Stop}
            
        The method collects legal moves, scores them using the evaluation function,
        and randomly selects among those with the best score.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
        """Evaluate the desirability of a game state after taking an action.
        
        Args:
            currentGameState: The current game state
            action: The proposed action
            
        Returns:
            float: A score where higher numbers are better, using values 8,4,2,1,0
            that are bitwise orthogonal (powers of 2)
            
        The function evaluates states based on:
        - Avoiding ghosts (returns 0 if too close)
        - Score improvements (returns 8)
        - Getting closer to food (returns 4) 
        - Maintaining direction (returns 2)
        - Default case (returns 1)
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        Food=newFood.asList()
        gPos=successorGameState.getGhostPositions()
        FoodDist=[]
        GhostDist=[]

        for food in Food:
            FoodDist.append(manhattanDistance(food,newPos))
        for ghost in gPos:
            GhostDist.append(manhattanDistance(ghost,newPos))

        if currentGameState.getPacmanPosition()==newPos:
            return(-(float("inf")))

        for dist in GhostDist:
            if dist<2:
                return (-(float("inf")))
        if len(FoodDist)==0:
            return float("inf")

        return 1000/sum(FoodDist) +10000/len(FoodDist)
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState) -> float:
    """Return the score of the state for use with adversarial search agents.
    
    Args:
        currentGameState: The game state to evaluate
        
    Returns:
        float: The score displayed in the Pacman GUI
        
    This is the default evaluation function for adversarial search agents.
    Not intended for use with reflex agents.
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """Base class for adversarial search agents (minimax, alpha-beta, expectimax).
    
    This abstract class provides common functionality for multi-agent searchers.
    It should not be instantiated directly, but rather extended by concrete
    agent implementations.
    
    Attributes:
        index: Agent index (0 for Pacman)
        evaluationFunction: Function used to evaluate game states
        depth: Maximum depth of search tree
    """

    def __init__(self, evalFn: str = 'scoreEvaluationFunction', depth: str = '2') -> None:
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """Minimax agent that implements adversarial search.
    
    This agent uses minimax search to determine the optimal action by considering
    the worst case scenario at each level.
    """

    def getAction(self, gameState: GameState) -> str:
        """Return the minimax action from the current gameState.
        
        Args:
            gameState: The current game state
            
        Returns:
            str: The optimal action according to minimax search
            
        Uses self.depth and self.evaluationFunction to determine the best action
        by considering the worst-case scenario at each level.
        """

        "*** YOUR CODE HERE ***"
        def max_value(gameState,depth):
            Actions=gameState.getLegalActions(0)
            if len(Actions)==0 or gameState.isWin() or gameState.isLose() or depth==self.depth:
                return(self.evaluationFunction(gameState),None)
            w=-(float("inf"))
            Act=None
            for action in Actions:
                sucsValue=min_value(gameState.generateSuccessor(0,action),1,depth)
                sucsValue=sucsValue[0]
                if(sucsValue>w):
                    w,Act=sucsValue,action
            return(w,Act)

        def min_value(gameState,agentID,depth):
            Actions=gameState.getLegalActions(agentID)
            if len(Actions) == 0:
                return(self.evaluationFunction(gameState),None)
            l=float("inf")
            Act=None
            for action in Actions:
                if(agentID==gameState.getNumAgents() -1):
                    sucsValue=max_value(gameState.generateSuccessor(agentID,action),depth + 1)
                else:
                    sucsValue=min_value(gameState.generateSuccessor(agentID,action),agentID+1,depth)
                sucsValue=sucsValue[0]
                if(sucsValue<l):
                    l,Act=sucsValue,action
            return(l,Act)
        max_value=max_value(gameState,0)[1]
        return max_value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """Minimax agent with alpha-beta pruning optimization.
    
    This agent implements minimax search with alpha-beta pruning to more efficiently
    explore the game tree by pruning branches that cannot affect the final decision.
    """

    def getAction(self, gameState: GameState) -> str:
        """Return the minimax action using alpha-beta pruning.
        
        Args:
            gameState: The current game state
            
        Returns:
            str: The optimal action according to alpha-beta pruning
            
        Pacman is always the max agent, ghosts are always min agents.
        At depth 0, max_value returns an action. At other depths, it returns a value.
        """

        "*** YOUR CODE HERE ***"
        def max_value(gameState,depth,a,b):
            Actions = gameState.getLegalActions(0) # Get actions of pacman
            if len(Actions)==0 or gameState.isWin() or gameState.isLose() or depth==self.depth:
                return (self.evaluationFunction(gameState), None)

            w=-(float("inf"))
            Act=None
            for action in Actions:
                sucsValue=min_value(gameState.generateSuccessor(0,action),1,depth,a,b)
                sucsValue=sucsValue[0]
                if w<sucsValue:
                    w,Act=sucsValue,action
                if w>b:
                    return (w,Act)
                a=max(a,w)
            return (w,Act)

        def min_value(gameState,agentID,depth,a,b):
            Actions=gameState.getLegalActions(agentID)
            if len(Actions) == 0:
                return (self.evaluationFunction(gameState),None)
            l = float("inf")
            Act = None
            for action in Actions:
                if (agentID == gameState.getNumAgents() - 1):
                    sucsValue = max_value(gameState.generateSuccessor(agentID,action),depth + 1,a,b)
                else:
                    sucsValue = min_value(gameState.generateSuccessor(agentID,action),agentID + 1,depth,a,b)
                sucsValue=sucsValue[0]
                if (sucsValue<l):
                    l,Act=sucsValue,action

                if (l<a):
                    return (l,Act)

                b=min(b,l)

            return(l,Act)

        a=-(float("inf"))
        b=float("inf")
        max_value=max_value(gameState,0,a,b)[1]
        return max_value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """An agent that uses expectimax search to make decisions.
    
    This agent models ghosts as choosing uniformly at random from their legal moves.
    It uses expectimax search to find optimal actions against probabilistic opponents.
    
    The agent searches to a fixed depth using a supplied evaluation function.
    """

    def getAction(self, gameState: GameState) -> str:
        """Return the expectimax action using self.depth and self.evaluationFunction.
        
        Args:
            gameState: The current game state
            
        Returns:
            str: The selected action (one of Directions.{North,South,East,West,Stop})
            
        All ghosts are modeled as choosing uniformly at random from their legal moves.
        """
        
        "*** YOUR CODE HERE ***"
        def max_value(gameState,depth):
            Actions=gameState.getLegalActions(0)
            if len(Actions)==0 or gameState.isWin() or gameState.isLose() or depth==self.depth:
                return (self.evaluationFunction(gameState),None)

            w=-(float("inf"))
            Act=None

            for action in Actions:
                sucsValue=exp_value(gameState.generateSuccessor(0,action),1,depth)
                sucsValue=sucsValue[0]
                if(w<sucsValue):
                    w,Act=sucsValue,action
            return(w,Act)

        def exp_value(gameState,agentID,depth):
            Actions=gameState.getLegalActions(agentID)
            if len(Actions)==0:
                return (self.evaluationFunction(gameState),None)

            l=0
            Act=None
            for action in Actions:
                if(agentID==gameState.getNumAgents() -1):
                    sucsValue=max_value(gameState.generateSuccessor(agentID,action),depth+1)
                else:
                    sucsValue=exp_value(gameState.generateSuccessor(agentID,action),agentID+1,depth)
                sucsValue=sucsValue[0]
                prob=sucsValue/len(Actions)
                l+=prob
            return(l,Act)

        max_value=max_value(gameState,0)[1]
        return max_value

def betterEvaluationFunction(currentGameState: GameState) -> float:
    """A more sophisticated evaluation function for Pacman game states.
    
    This function evaluates states by combining the game score with a penalty
    based on distance to the closest food pellet. The penalty uses the reciprocal
    of the distance to give higher penalties to food that is farther away.
    
    Args:
        game_state: The game state to evaluate
        
    Returns:
        float: The evaluation score where higher values are better
    """
    
    "*** YOUR CODE HERE ***"
    pacPosition=currentGameState.getPacmanPosition()
    gList=currentGameState.getGhostStates()
    Food=currentGameState.getFood()
    Capsules=currentGameState.getCapsules()

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    foodDistList=[]
    for food in Food.asList():
        foodDistList+=[util.manhattanDistance(food,pacPosition)]
    minFDist=min(foodDistList)
    GhDistList=[]
    ScGhDistList=[]
    for ghost in gList:
        if ghost.scaredTimer==0:
            GhDistList+=[util.manhattanDistance(pacPosition,ghost.getPosition())]
        elif ghost.scaredTimer>0:
            ScGhDistList+=[util.manhattanDistance(pacPosition,ghost.getPosition())]
    minGhDist=-1
    if len(GhDistList) > 0:
        minGhDist=min(GhDistList)
    minScGhDist=-1
    if len(ScGhDistList)>0:
        minScGhDist=min(ScGhDistList)
    score=scoreEvaluationFunction(currentGameState)
    score-= 1.5 * minFDist + 2 * (1.0/minGhDist) + 2 * minScGhDist + 20 * len(Capsules) + 4 * len(Food.asList())
    return score
    
# Abbreviation
better = betterEvaluationFunction

