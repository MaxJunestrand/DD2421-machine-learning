import random
import statistics

import matplotlib.pyplot as plt
import numpy as np

import dtree
import monkdata as m

m1 = m.monk1
m2 = m.monk2
m3 = m.monk3
monks = [m1, m2, m3]
# Assignment 0
# Monk 1 is hardest to learn since the concept gives us the least information about the nature of the data set. Attribute 5 might be easy to separate, but linking
# attribute 1 and 2 might be tricky.
# Monk 2 will probably be easy, as the rule is quite simple.
# We believe that Monk 3 will be the easiest, as there are many attributes to choose from. 

# Assignment 1
print("Dataset: monk-1: Entropy = " + str(dtree.entropy(m1)))
print("Dataset: monk-2: Entropy = " + str(dtree.entropy(m2)))
print("Dataset: monk-3: Entropy = " + str(dtree.entropy(m3)))

# Assignment 2
# Entropy can be viewed a bit like standard deviation. So high entropy means that many (if not all) outcomes have the same or a similar value.
# A uniform distribution is a distribution where all outcomes are as likely, ex. a good number generator or a dice, and therefore has a lot of entropy.
# For entropy, we can look at the example of a dice roll, where all outcomes are as likely and thus lead to a high entropy (2.58).
# A fake dice where not all outcomes are as likely (and therefore a non-uniform distribution), will have a lower entropy. An example is a die where
# p_6 = 0.5. Here we will have an entropy of 2.16.

#assignement 3
m1_gains = []
m2_gains = []
m3_gains = []

for attribute in m.attributes:
    m1_gains.append(dtree.averageGain(m1, attribute))
for attribute in m.attributes:
    m2_gains.append(dtree.averageGain(m2, attribute))
for attribute in m.attributes:
    m3_gains.append(dtree.averageGain(m3, attribute))

print("Information Gain")
print("Dataset: monk-1: " + str(m1_gains))
print("Dataset: monk-2: " + str(m2_gains))
print("Dataset: monk-3: " + str(m3_gains))

# assignement 4
# Choose attribute 5, as it leads to the most gain of info.
# The subset S_k has the least entropy as the attribute 5 is the best at identifying the solution.
# Therefore the information gain is the best heuristic when deciding where to split the tree, as we want to maximize the expected reduction of entropy in the decision tree.



# assignment 5
print("Assignment 5")

t1 = dtree.buildTree(m.monk1, m.attributes)
t2 = dtree.buildTree(m.monk2, m.attributes)
t3 = dtree.buildTree(m.monk3, m.attributes)
print("Monk1: E_train", 1-dtree.check(t1, m.monk1test), "  E_test: ", 1-dtree.check(t1, m.monk1))
print("Monk2: E_train", 1-dtree.check(t2, m.monk2test), "  E_test: ", 1-dtree.check(t2, m.monk2))
print("Monk3: E_train", 1-dtree.check(t3, m.monk3test), "  E_test: ", 1-dtree.check(t3, m.monk3))

# The test scores were as we thought, as the decision trees were built on these, so therefore the decision trees should be entirely correct on this.
# We thought that Monk1 would be the most hard to learn as we thought that it would have the least data. We were however proved wrong as Monk2 had a higher
# error rate, and therefore was harder.
# Monk3 has many attributes leading to information, which meant that it had a very low error rate, which even managed to break through the slightly higher amount of noise
# in the training dataset.


# Assignment 6
# Here we simplify the model by pruning away nodes (i.e. reducing model complexity) which reduces the variance (the error from sensitivity to small fluctuations in the training set, or overfitting).
# However, making the model less complex means that the bias error will be greater (error from erroneous assumptions in the learning algorithm, or underfitting).


# Assignment 7
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return (ldata[:breakPoint], ldata[breakPoint:])

#ret = partition(m.monk1, 0.6)
#monk1train = ret[0]
#monk1val = ret[1]
#t1 = dtree.buildTree(monk1train, m.attributes)
#errorBeforePruning = 1 - dtree.check(t1, monk1val)

def errorAfterPruning(t, monkval):
    while True:
        tPrunedList = dtree.allPruned(t) #return all possible pruned trees
        bestTree = None
        lowestError = 9999999999999
        for prunedTree in tPrunedList:
            if (1 - dtree.check(prunedTree, monkval)) < lowestError:
                bestTree = prunedTree
        if bestTree == None:
            break
        t = bestTree

    errorAfterPruning = 1 - dtree.check(t, monkval)
    return errorAfterPruning


  #largest value - lowest value
def plotErrorOnFraction(monk_data, label_string):
    meanList = []
    errorList = []
    fractionList = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for fraction in fractionList:
        tempMeanList = []
        for i in range(100):
            ret = partition(monk_data, fraction)
            monk_train = ret[0]
            monk_valuation = ret[1]
            tree = dtree.buildTree(monk_train, m.attributes)
            tempMeanList.append(errorAfterPruning(tree, monk_valuation))

        meanList.append(statistics.mean(tempMeanList))
        errorList.append(max(tempMeanList)-min(tempMeanList))


    plt.errorbar(fractionList, meanList, yerr = errorList, fmt = '.k', label=label_string)
    
    
    plt.legend()
    plt.xlabel('Fraction of partition to training')
    plt.ylabel('Error level')
    plt.show()

plotErrorOnFraction(m.monk1, "Monk 1 datapoints")
plotErrorOnFraction(m.monk3, "Monk 3 datapoints")

# Monk1: 0.3 and 0.5 seems to be sweetpoints, however the erorrs get larger and large

