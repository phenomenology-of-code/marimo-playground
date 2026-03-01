from math import isclose
from tracemalloc import start
import numpy as np #Check this works - if not, speak to me and I can chat about how to get it installed :) if reading this outside session, google "how to install numpy from command line" and follow online instructions
import matplotlib.pyplot as plt
# The statespace
# Main political parties in Britain, with additional states for dead and not voting (alive)
states = ["Conservative","Labour","LibDem", "Reform", "Green", "Other", "NVA", "YP", "Dead"]

# transitionName = [["SS","SR","SI"],["RS","RR","RI"],["IS","IR","II"]]
startingProbs = np.array([0.237,0.337,0.122,0.143,0.067])
startingProbs = np.append(startingProbs,1-sum(startingProbs))#The "other" vote
startingProbs = [x*0.597 for x in startingProbs]#turnout
startingProbs = np.append(startingProbs,1-0.597)#Didn't vote, alive
startingProbs = np.append(startingProbs,[0.0, 1-sum(startingProbs)])#Dead voters starts at 0, YP didn't exist in 2024
# Probabilities matrix (transition matrix)
transitionMatrix = np.array([[0.48,0.02,0.04,0.23,0.01,0.01,0.10,0.01,0.1],#tories
                    [0.02,0.50,0.09,0.07,0.09,0.02,0.14,0.02,0.05],#labour
                    [0.04,0.07,0.62,0.05,0.06,0.01,0.10,0.005,0.045],#libdem
                    [0.02,0.005,0.005,0.71,0.005,0.0,0.1,0.005,0.15],#reform
                    [0.02,0.05,0.05,0.02,0.68,0.02,0.1,0.03,0.03],#green
                    [0.01,0.01,0.02,0.12,0.21,0.48,0.11,0.01,0.03],#other
                    [0.0,0.0,0.0,0.25,0.15,0.0,0.55,0.0,0.05],#NVA
                    [0.0,0.04,0.01,0.09,0.18,0.05,0.05,0.54,0.04],#YP
                    [0,0,0,0,0,0,0,0,1.0]])#dead
currentParty = "" #Can set this determinstically if we'd like rather than assigning initial probabilities
# Starting Probabilities
print("Starting proportions:")
for party, proportion in zip(states, startingProbs):
    print(f"{party}\t{proportion}")
print(sum(startingProbs))
totalProbs = list(map(sum, transitionMatrix))   # You don't need `list()` in Py2
print(totalProbs)

#Exercise 1: Make a python def that takes in the startingProbs and transitionMatrix as arguments, and returns the next state of probabilities
#Solution:

#Extension: what's the long-term behaviour of this model? Can you work this out in two different ways, for example through lots of transitions or via finding stationary solution?
#HINT: for lots of transitions, think about a for loop.
#HINT: for linear algebra solution, np.linalg.eig(matrix) returns eigenvalues and eigenvectors of the matrix. Note that xP=x can be transposed to get P^T x^T = x^T
#we also note that the complex eigenvalues aren't possible, and additionally, if abs(eigenvalue)<1, this just goes to 0 as n approaches infinity, so we're searching for the one where eigenvalue =1
#
#Solution, modelled:

#Solution, linear algebra

#Exercise 2: Make a python def that takes in the startingProbs (or cumulative starting probs, either works for this example) and gives the index in the states, or currentPub as a name, based on monte carlo method
#HINT: google monte carlo, or chat about it, if you're unsure that is - briefly, it's getting a random number between 0 and 1, which represents a probability that we can use to sample
# Make it so that this def works for the starting probabilities or the probabilities in 2029

#Solution:

#EXTENSION: plot these on a histogram, or a graph showing proportion by party

#Exercise 3 - Find the entries, for tories and labour, of the generator matrix that models the 25% decrease of their voter base to the other party every 5 years
#Solution:
k = 0.001#Needed for next part - calculate this yourself, or use solutions; 0.01 is NOT correct
#EXTENSION: why is it that the survival function is exp(-kt)? How can we derive that? Think about what P(X>s+t) is using the memoryless markov proprty, and then think about what functions are such that f(x+y)=f(x)f(y), maybe google cauchy functional equations
dt = 0.0001#Infinitesimal time-step
# states = ["Conservative","Labour","LibDem", "Reform", "Green", "Other", "NVA", "YP", "Dead"]
generatorMatrix = np.array([[-k,0.25*k,0.1*k,0.42*k,0.01*k,0.02*k,0.2*k,0.0*k,0.0],#tories
                           [0.2*k,-k,0.1*k,0.2*k,0.4*k,0.02*k,0.05*k,0.03*k,0.0],#labour
                           [0.05*k,0.1*k,-0.25*k,0.01*k,0.02*k,0.02*k,0.05*k,0.0,0.0],#libdems
                           [0.08*k,0.01*k,0.0,-0.25*k,0.05*k,0.01*k,0.1*k,0.0,0.0],#reform
                           [0.0,0.05*k,0.05*k,0.0,-0.2*k,0.03*k,0.02*k,0.05*k,0.0],#green
                           [0.01*k,0.01*k,0.01*k,0.1*k,0.1*k,-0.23*k,0.0,0.0,0.0],#other
                           [0.01*k,0.01*k,0.01*k,0.1*k,0.1*k,0.01*k,-0.24*k,0.0,0.0],#NVA
                           [0.0,0.01*k,0.0,0.01*k,0.7*k,0.01*k,0.3*k,-1.03*k,0.0],#YP
                           [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]#Dead - why have I put dead as 0 for all parties?
                           ])#Do these have to be constant? What sort of functions, analytical or numerical, could we replace them with, if any? What behaviour would you want to implement?
lastColumn = generatorMatrix.shape[1]-1
for i in range(generatorMatrix.shape[0]):
    generatorMatrix[i,lastColumn] = -sum(generatorMatrix[i,0:lastColumn])
totalGenerator = list(map(sum, generatorMatrix))
print("Check that these are all zero:")
print(totalGenerator)
nextTransitionMatrix = transitionMatrix.copy()

#Exercise 4: Given the generator matrix above and our time steps, make a python def that takes in time (in years) as an input, the starting probabilities, and the generator matrix as arguments
#And then returns the new distribution
#HINT: use numerical integration for the timesteps between the times; for example, if t=3, we do n = t/dt=3000 steps of recalculating P using our forward Kolmogorov equation, and then multiply our initial distribution by this
#CONSIDERl: because we're numerically integrating, what could happen to our transition probabilities, specifically the requirement that the sum of rows is 1? How do we avoid that?
#Solution:

print("Make sure these are all 1.0:")
#print(totalProbsNext)
#newDistribution = nextProbs(startingProbs,nextTransitionMatrix)
# print("Proportions after 3 years:")
# for party, proportion in zip(states, newDistribution):
#     print(f"{party}\t{proportion}")
# print("Make sure they sum to 1.0:")
# print(sum(newDistribution))

#Exercise 5: create a python def that takes a mean election cycle, mu, and finds lambda, 1/mu, and using monte carlo simulations, returns inverse results of exponential distribution
#Plot these samples on a histogram using plt
#Solution:

#Exercise 6: Using these time samples, simulate the markov chain now at each time, using the numerical integration tool above for the times
#HINT: at each sample, you'll get a distribution at that time. You then use monte carlo on that dsitribution to get the party that that voter has chosen
#Solution: