import math
import random

import matplotlib.pyplot as plt
import numpy
from scipy.optimize import minimize

# Generate test data
spread = 0.5
classA = numpy.concatenate((
    numpy.random.randn(20, 2) * spread + [0, 0],
    numpy.random.randn(20, 2) * spread + [0, 0]
))
classB = numpy.random.randn(10, 2) * spread + [0, 0]

inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate((
    numpy.ones(classA.shape[0]),
    -numpy.ones(classB.shape[0])
))
N = inputs.shape[0]  # Number of rows (samples)
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]


def RBFkernal(x, y):
    # The parameter σ is used to control the smoothness of the boundary.
    sigma = 5
    dist = numpy.linalg.norm(x-y)  # Euclidian distance
    return numpy.power(math.e, (-dist/(2*numpy.power(sigma, 2))))

def PolyKernel(x, y):
    p = 5
    return numpy.power(numpy.dot(x, y) + 1, p)


def LinKernel(x, y):
    return numpy.dot(x, y)


# Set the kernal globally.
kernal = RBFkernal

# Calculate P globally with Numpy matrix (numpy array)
P = numpy.zeros((N, N))
for i in range(N):
    for j in range(N):
        P[i][j] = targets[i]*targets[j]*kernal(inputs[i], inputs[j])


def zerofun(alpha):
    # Implements equation (10) from the assignment. This will have to be 0.
    return numpy.dot(alpha, targets)


def objective(vector_alfa):
    # Implements equation (4) from the assignment.
    return (1/2)*numpy.dot(vector_alfa, numpy.dot(vector_alfa, P)) - numpy.sum(vector_alfa)


# Set input for call to minimize from assignment page five.
# Set start
start = numpy.zeros(N)
# Set bounds
#C = None
C = 1
B = [(0, C) for b in range(N)]
# Set contraint
XC = {'type': 'eq', 'fun': zerofun}

ret = minimize(objective, start, bounds=B, constraints=XC)
if (not ret['success']):
    print('couldnt find solution')
alpha = ret['x']


# Take out values bigger than threshold
threshold = 10e-5
# Non_zero list is (Alpha, point, target)
non_zero = []
for i in range(N):
    if abs(alpha[i]) > threshold:
        non_zero.append((alpha[i], inputs[i], targets[i]))


# Calculate the treshhold b (equation (7))
def calculate_threshhold():
    b = -non_zero[0][2]
    s = non_zero[0][1]  # Any support vector, lets just take the first
    for value in non_zero:
        b += value[0]*value[2]*kernal(s, value[1])
    return b


# Indicator function (equation (6))
def indicator(s_vector, b):
    result = -b
    for value in non_zero:
        result += value[0]*value[2]*kernal(s_vector, value[1])
    return result


def plotData():
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    # adding yellow dot for sv
    #plt.plot([p[1][0] for p in non_zero], [p[1][1] for p in non_zero], 'yo')

    xgrid = numpy.linspace(-3, 3)
    ygrid = numpy.linspace(-3, 3)
    grid = numpy.array([[indicator([x, y], calculate_threshhold())
                         for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
                colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    plt.axis('equal')  # Force same scale on both axes
    plt.savefig('svmplot.pdf')  # Save a copy in a file
    plt.show()  # Show the plot on the screen


plotData()
