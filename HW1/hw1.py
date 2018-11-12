import numpy as np 
import matplotlib.pyplot as plt

def PerformRandomWalk(agentMatrix):
	end

latticeSize = 100;
numberOfAgents = 1000;
agentMatrix = np.zeros((3,latticeSize,latticeSize)) # first column represents state of agent, 1: susc, 2: inf, 3: rec. second (firt other wise) represents row size, third then is column size.
