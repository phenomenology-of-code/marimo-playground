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
def nextProbs(probs,transitionMatrix):
    return probs @ transitionMatrix
#
results_2029 = nextProbs(startingProbs, transitionMatrix)
print("2029 modelled results:")
for party, proportion in zip(states, results_2029):
    print(f"{party}\t{proportion}")
#Extension: what's the long-term behaviour of this model? Can you work this out in two different ways, for example through lots of transitions or via finding stationary solution?
#HINT: for lots of transitions, think about a for loop.
#HINT: for linear algebra solution, np.linalg.eig(matrix) returns eigenvalues and eigenvectors of the matrix. Note that xP=x can be transposed to get P^T x^T = x^T
#we also note that the complex eigenvalues aren't possible, and additionally, if abs(eigenvalue)<1, this just goes to 0 as n approaches infinity, so we're searching for the one where eigenvalue =1
#
#Solution, modelled:
long_term = startingProbs
for i in range(1000):
    long_term = nextProbs(long_term, transitionMatrix)

print("Long term modelled results:")
for party, proportion in zip(states, long_term):
    print(f"{party}\t{proportion}")

#Solution, linear algebra
eigenvalues, eigenvectors = np.linalg.eig(transitionMatrix)#long term from solving stationary equation
lookup_eigvalue = np.complex128(1.0,0.0) # 1.0 in complex format i.e. 1 + 0j i.e. 1.0 + 0.0j
long_term_LA = eigenvectors[0]
a_tol=1e-6
# Find the index of the eigenvalue closest to 1
for eigval_index in range(len(eigenvalues)):
    if np.isclose(lookup_eigvalue,eigenvalues[eigval_index]):#Why do we use np.close()? What are some of the limitations with storing numbers or calculating numbers in computers? Happy to chat :)
        long_term_LA = eigenvectors[eigval_index]
        break

# normalise and make real
long_term_LA = np.real(long_term_LA)
long_term_LA /= long_term_LA.sum()
long_term_LA = long_term_LA.T #Transpose to get 1x9 matrix
print("Long term mathematical results:")
for party, proportion in zip(states, long_term_LA):
    print(f"{party}\t{proportion}")#What's the limitation of this model? What are we seeing? How could we improve it? When do new voters (people who've turned 16 since 2024, naturalised British citizens etc.) come in, if ever, in this model?

#Exercise 2: Make a python def that takes in the startingProbs (or cumulative starting probs, either works for this example) and gives the index in the states, or currentPub as a name, based on monte carlo method
#HINT: google monte carlo, or chat about it, if you're unsure that is - briefly, it's getting a random number between 0 and 1, which represents a probability that we can use to sample
# Make it so that this def works for the starting probabilities or the probabilities in 2029

#Solution:
cumulativeStartingProbs = startingProbs.copy()
for i in range(1,len(cumulativeStartingProbs)):
    cumulativeStartingProbs[i] += cumulativeStartingProbs[i-1]
print("Cumulative starting proportions:")
for party, proportion in zip(states, cumulativeStartingProbs):
    print(f"{party}\t{proportion}")#NOTE: using np.cumsum() is fine here instead of a loop, and actually so much funnier hehe cumsum

def sample_monte_carlo(cumulativeProbabilities, states, sims=1000, seed = None):
    generator = np.random.default_rng(seed)
    # cumulativeStartingProbs = np.cumsum(startingProbs)
    random_samples = generator.random(sims)
    
    indices = np.searchsorted(cumulativeProbabilities, random_samples)
    results = [states[i] for i in indices]
    
    return results

results = sample_monte_carlo(cumulativeStartingProbs,states, 1000, 123)

#EXTENSION: plot these on a histogram, or a graph showing proportion by party

unique_states, counts = np.unique(results, return_counts=True)

colours = ['blue', 'red', 'orange','skyblue','green','grey','purple','darkred','black']
colours = dict(zip(states, colours))
unique_colours = [colours[s] for s in unique_states]
plt.figure(figsize=(10, 6))
bars = plt.bar(unique_states, counts, color=unique_colours, edgecolor='black')
# states =                   ["Conservative","Labour","LibDem", "Reform", "Green", "Other", "NVA", "YP", "Dead"]
# Formatting the chart
plt.title(f'Monte Carlo Simulation Results For Starting Probabilities (n={1000})', fontsize=14)
plt.xlabel('Outcomes', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

cumulative_results_2029 = np.cumsum(results_2029)
results_monte_2029 = sample_monte_carlo(cumulative_results_2029,states,1000,123)

unique_states_2029, counts_2029 = np.unique(results_monte_2029, return_counts=True)

unique_colours_2029 = [colours[s] for s in unique_states_2029]
plt.figure(figsize=(10, 6))
bars = plt.bar(unique_states_2029, counts_2029, color=unique_colours_2029, edgecolor='black')
# states =                   ["Conservative","Labour","LibDem", "Reform", "Green", "Other", "NVA", "YP", "Dead"]
# Formatting the chart
plt.title(f'Monte Carlo Simulation Results For 2029 (n={1000})', fontsize=14)
plt.xlabel('Outcomes', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#Exercise 3 - Find the entries, for tories and labour, of the generator matrix that models the 25% decrease of their voter base to the other party every 5 years
#Solution:
# 25%=1-exp(-k*5), hence 0.75=exp(-5k), so k = -1/5 * ln(0.75)
k = -1/5*np.log(0.75)
check_k = 1-np.exp(-5*k)
print("This must be 25% for k to have been calculated correctly:")
print(check_k)
print("Value of k is:")
print(k)
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
def normaliseTransitionMatrix(matrix):
    newMatrix = matrix
    lastRow = newMatrix.shape[0]
    lastRow-=1
    for i in range(lastRow+1):
        newMatrix[i,]/=sum(newMatrix[i,])
    return newMatrix
        

def getGextTransitionMatrix(transitionMatrix,generatorMatrix,dt,t):
    newTransitionMatrix = transitionMatrix
    steps = int(t/dt)#Is this acceptable? Should we instead use round? What are the considerations for computational complexity? Is it that big of a deal if we use 3000 for 3000.95 instead of 3000? What about 1 for 1.95 instead of 2?
    for step in range(steps):
        dP = (generatorMatrix@newTransitionMatrix)*dt
        newTransitionMatrix +=dP
        newTransitionMatrix = normaliseTransitionMatrix(newTransitionMatrix)
    return newTransitionMatrix
nextTransitionMatrix = getGextTransitionMatrix(transitionMatrix,generatorMatrix,dt,3)
totalProbsNext = list(map(sum, nextTransitionMatrix))
print("Make sure these are all 1.0:")
print(totalProbsNext)
newDistribution = nextProbs(startingProbs,nextTransitionMatrix)
print("Proportions after 3 years:")
for party, proportion in zip(states, newDistribution):
    print(f"{party}\t{proportion}")
print("Make sure they sum to 1.0:")
print(sum(newDistribution))

#Exercise 5: create a python def that takes a mean election cycle, mu, and finds lambda, 1/mu, and using monte carlo simulations, returns inverse results of exponential distribution
#Plot these samples on a histogram using plt
#Solution:
def exponentialSamples(mu=5,sims=1000,seed=None):
    #lmda=1/mu
    generator = np.random.default_rng(seed)
    random_samples = generator.random(sims)
    results = []
    for sample in random_samples:
        results.append(-mu*np.log(1-min(sample,0.99999)))
    return results
exponentialSamples = exponentialSamples(5,1000,123)
plt.hist(exponentialSamples, bins=15, density=True, color='skyblue', alpha=0.7, label='Exponential Samples')
plt.show()

#Exercise 6: Using these time samples, simulate the markov chain now at each time, using the numerical integration tool above for the times
#HINT: at each sample, you'll get a markov distribution at that time. You then use monte carlo on that markov distribution to get the party that that voter has chosen
#Solution:
resultsPoissonProcess = []
for exponentialSample in exponentialSamples:
    newTransitionMatrix = getGextTransitionMatrix(transitionMatrix,generatorMatrix,dt,exponentialSample)
    resultsPoissonProcess.append(nextProbs(startingProbs,newTransitionMatrix))
print(resultsPoissonProcess)
unique_states_Poisson, counts_Poisson = np.unique(resultsPoissonProcess, return_counts=True)

unique_colours_Poisson = [colours[s] for s in unique_states_Poisson]
plt.figure(figsize=(10, 6))
bars = plt.bar(unique_states_Poisson, counts_Poisson, color=unique_colours_Poisson, edgecolor='black')
# states =                   ["Conservative","Labour","LibDem", "Reform", "Green", "Other", "NVA", "YP", "Dead"]
# Formatting the chart
plt.title(f'Monte Carlo Simulation Results For 2029 Poisson Process (n={1000})', fontsize=14)
plt.xlabel('Outcomes', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()