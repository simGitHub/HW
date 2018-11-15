import numpy as np 
import matplotlib.pyplot as plt

# no boundary is assumed.
def PerformRandomWalk(agentMatrix, d):
	updatedAgentMatrix = agentMatrix
	for iRow in range(0,latticeSize):
		for jCol in range(0,latticeSize):
			for kState in range(0,2):
				nbrOfAgentsAtCurrentStateSite = agentMatrix[kState,iRow,jCol]
				for iAgentOnCurrentStateSite in range(0,nbrOfAgentsAtSite):
					if(np.random.rand(1) < d):
						updatedAgentMatrix[kState,iRow,jCol] -= 1
						if(0 < np.random.rand(1) < 0.25):
							updatedAgentMatrix[kState, (1 + iRow) % latticeSize,jCol] += 1
						else if(0.25 < np.random.rand(1) < 0.5):
							updatedAgentMatrix[kState, (iRow - 1) % latticeSize,jCol] += 1
						else if(0.5 < np.random.rand(1) < 0.75):
							updatedAgentMatrix[kState, iRow, (jCol + 1) % latticeSize] += 1
						else if(0.75 < np.random.rand(1) < 1):
							updatedAgentMatrix[kState, iRow, (jCol - 1) % latticeSize] += 1	
	return updatedAgentMatrix


def PerformInfectionAndRecovery(agentMatrix, beta, gamma):
	updatedAgentMatrix = agentMatrix
	for iRow in range(0,latticeSize):
		for jCol in range(0,latticeSize):
			nbrOfInfectedAtCurrentSite = agentMatrix[1,iRow,jCol]
			for iInfectedAtCurrentSite in range(0,nbrOfInfectedAtCurrentSite):
				if(np.random.rand(1) < beta):
					updatedAgentMatrix[0,iRow,jCol] = 0
					updatedAgentMatrix[1,iRow,jCol] += agentMatrix[0,iRow,jCol]
				else if(np.random.rand(1) < gamma):
					updatedAgentMatrix[1,iRow,jCol] -= 1
					updatedAgentMatrix[2,iRow,jCol] += 1
	return updatedAgentMatrix


def InitializeAgentMatrix(numberOfAgents, latticeSize):
	agentMatrix = np.zeros((3,latticeSize,latticeSize)) # 0: susc, 1: inf, 2: rec
	for iAgent in range(0,numberOfAgents):
		x = np.random.randint(latticeSize)
		y = np.random.randint(latticeSize)
		infectionLength = np.sqrt((1 - ratio)) * latticeSize
		if(latticeSize / 2 - infectionLength / 2 <= x <= latticeSize / 2 + infectionLength / 2 and
			latticeSize / 2 - infectionLength / 2 <= y <= latticeSize / 2 + infectionLength / 2):
			agentMatrix[1, y, x] += 1
		else:
			agentMatrix[0, y, x] += 1
	return agentMatrix

d = 0.8; beta = 0.6; gamma = 0.01
latticeSize = 100; numberOfAgents = 1000
ratio = 0.9
