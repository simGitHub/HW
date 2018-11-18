import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation



# no boundary is assumed.
def PerformRandomWalk(agentMatrix, d):
	updatedAgentMatrix = agentMatrix.copy()
	for iRow in range(0,latticeSize):
		for jCol in range(0,latticeSize):
			for kState in range(0,3):
				nbrOfAgentsAtCurrentStateSite = agentMatrix[iRow,jCol,kState]
				for iAgentOnCurrentStateSite in range(0,int(nbrOfAgentsAtCurrentStateSite)):
					if(np.random.rand(1) < d):
						updatedAgentMatrix[iRow,jCol,kState] -= 1
						r = np.random.rand(1)
						if(0 < r < 0.25):
							updatedAgentMatrix[(1 + iRow) % latticeSize,jCol,kState] += 1
						elif(0.25 < r < 0.5):
							updatedAgentMatrix[(iRow - 1) % latticeSize,jCol,kState] += 1
						elif(0.5 < r < 0.75):
							updatedAgentMatrix[iRow, (jCol + 1) % latticeSize,kState] += 1
						elif(0.75 < r < 1):
							updatedAgentMatrix[iRow, (jCol - 1) % latticeSize,kState] += 1	
	return updatedAgentMatrix

def PerformInfectionAndRecovery(agentMatrix, beta, gamma):
	updatedAgentMatrix = agentMatrix.copy()
	for iRow in range(0,latticeSize):
		for jCol in range(0,latticeSize):
			nbrOfInfectedAtCurrentSite = agentMatrix[iRow,jCol,1]
			for iInfectedAtCurrentSite in range(0,int(nbrOfInfectedAtCurrentSite)):
				if(np.random.rand(1) < beta):
					updatedAgentMatrix[iRow,jCol,0] = 0
					updatedAgentMatrix[iRow,jCol,1] += agentMatrix[iRow,jCol,0]
				if(np.random.rand(1) < gamma):
					updatedAgentMatrix[iRow,jCol,1] -= 1
					updatedAgentMatrix[iRow,jCol,2] += 1
	return updatedAgentMatrix


def InitializeAgentMatrix(numberOfAgents, latticeSize):
	agentMatrix = np.zeros((latticeSize,latticeSize,3)) # 0: susc, 1: inf, 2: rec
	for iAgent in range(0,numberOfAgents):
		x = np.random.randint(latticeSize)
		y = np.random.randint(latticeSize)
		infectionLength = np.sqrt((1 - ratio)) * latticeSize
		if(latticeSize / 2 - infectionLength / 2 <= x <= latticeSize / 2 + infectionLength / 2 and
			latticeSize / 2 - infectionLength / 2 <= y <= latticeSize / 2 + infectionLength / 2):
			agentMatrix[y, x, 1] += 1
		else:
			agentMatrix[y, x, 0] += 1
	return agentMatrix



fig, ax = plt.subplots(figsize=(5, 8))
def update(t):
    ax.imshow(storedAgentMatrix[:,:,:,t])
    ax.set_title('Random walk')


runRandomWalk = False; runMainSimulation = True

d = 0.8; beta = 0.6; gamma = 0.01
latticeSize = 10; numberOfAgents = 1; ratio = 0.9
timeStepsForRandomWalk = 40
storedAgentMatrix = np.zeros((latticeSize, latticeSize, 3 , timeStepsForRandomWalk))
t = 0
if(runRandomWalk == True):
	agentMatrix = InitializeAgentMatrix(numberOfAgents, latticeSize)
	for t in range(0,timeStepsForRandomWalk):
		agentMatrix = PerformRandomWalk(agentMatrix,d)
		storedAgentMatrix[:,:,:,t] = agentMatrix.copy()
	# create gif for random walk
	anim = FuncAnimation(fig, update, frames=np.arange(0, timeStepsForRandomWalk), interval=50)
	anim.save('randomWalk.gif', dpi=80, writer='imagemagick')
	plt.close()



d = 0.8; nRuns = 1
latticeSize = 100; numberOfAgents = 1000; ratio = 0.99
gammaRuns = np.array([0.3, 0.2,0.1,0.05,0.03,0.02,0.017,0.015,0.013,0.010])
betaRuns = np.array([0.8])
if(runMainSimulation == True):
	R_storedRuns = np.zeros(( nRuns, np.size(betaRuns) , np.size(gammaRuns) ))
	for iRun in range(0,nRuns):
		print 'iRun = %i' %iRun
		iBeta = 0
		for beta in betaRuns:
			iGamma = 0
			for gamma in gammaRuns:
				print 'iGamma = %i' %iGamma
				agentMatrix = InitializeAgentMatrix(numberOfAgents, latticeSize)
				SnbrSusceptible = np.array([])
				SnbrInfected = np.array([])
				SnbrRecovered = np.array([])
				agentMatrixSaved = np.array([])
				t = 0; nbrInfected = np.sum(agentMatrix[:,:,1]); showDynamics = False; tSaveMatrix = 100
				saveMatrix = False; matrixSaved = False; plotPopulationDynamics = False; plotR = True
				while(nbrInfected != 0 and nbrInfected < numberOfAgents):
					agentMatrix = PerformInfectionAndRecovery(agentMatrix, beta, gamma)
					agentMatrix = PerformRandomWalk(agentMatrix,d)

					nbrSusceptible = np.sum(agentMatrix[:,:,0])
					nbrInfected = np.sum(agentMatrix[:,:,1])
					nbrRecovered = np.sum(agentMatrix[:,:,2])
					SnbrSusceptible = np.append(SnbrSusceptible,nbrSusceptible)
					SnbrInfected = np.append(SnbrInfected,nbrInfected)
					SnbrRecovered = np.append(SnbrRecovered,nbrRecovered)
					
					if(showDynamics == True): 
						im1 = plt.imshow(agentMatrix)	
						plt.title('Time step: %i' %t)
						plt.ion()
						plt.show()
						plt.pause(0.1)

					#if((t % 100) == 0):
					#	print t
					if(t == tSaveMatrix and saveMatrix):
						agentMatrixSaved = agentMatrix.copy()
						matrixSaved = True

					t += 1	
				R_storedRuns[iRun,iBeta,iGamma] = nbrRecovered
				iGamma += 1

				if(plotPopulationDynamics == True):
					St = np.arange(np.size(SnbrInfected))
					plt.figure(1)
					plt.plot(St, SnbrInfected,'r')
					plt.plot(St, SnbrRecovered,'g')
					plt.plot(St, SnbrSusceptible,'b')

					plt.title('d = %(d)f, beta = %(beta)f and gamma = %(gamma)f' %{'d' :d, 'beta': beta, 'gamma' : gamma})
					plt.legend(['Infected','Recovered','Susceptible'])
					plt.xlabel('Time step')
					plt.ylabel('Number of agents')
					plt.show()

				if(matrixSaved == True):
					plt.figure(2)
					im1 = plt.imshow(agentMatrixSaved)
					plt.title('Time step: %i' %tSaveMatrix)

			iBeta += 1
R = np.average(R_storedRuns,0)

plt.figure(1)

for i in range(0,np.size(betaRuns)):
	plt.scatter(betaRuns[i] / gammaRuns, R[i,:])


plt.ylabel('R_inf')
plt.xlabel('beta / gamma')
plt.legend([betaRuns[0]])#, betaRuns[1]])
plt.show()




	



	