import pickle
from fileinput import filename

import autograd.numpy as np
from autograd import grad

import matplotlib
import matplotlib.pyplot as plt
import random
import os
from openpyxl import Workbook

matplotlib.rcParams.update({'font.size': 10})

PATH = 'C:/Users/raghu/Desktop/Namita NN/XYZ_/'

#to unpack any nested tuple
def unpack(parent):
    for child in parent:
        if type(child) == tuple:
            yield from unpack(child)
        else:
            yield child

#repacks any flat tuple according to the structured tuple provided as first argument
def repack(structured, flat):
    output = []
    global flatlist
    flatlist = list(flat)
    for child in structured:
        if type(child) == tuple:
            output.append(repack(child, flatlist))
        else:
            output.append(flatlist.pop(0))
    return tuple(output)


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def initial_nn_weight(dimensions):
    n = len(dimensions)
    weightMatrix = [0] * (n - 1)
    for i in range(0, n - 1):
        weightMatrix[i] = tuple(
            map(tuple, np.sqrt(1 / dimensions[i]) * np.random.randn(dimensions[i], dimensions[i + 1])))
        # print(weightMatrix[i])
        # print("-----")

    biasMatrix = [0] * (n - 1)
    for i in range(0, n - 1):
        temparray = np.zeros(shape=(1, dimensions[i + 1]))
        biasMatrix[i] = totuple(temparray)
        # print(biasMatrix[i])
        # print("----------------")

    result = (tuple(weightMatrix), tuple(biasMatrix))
    # print(result)
    return result


def generateTrueDynamics(StartXY, typeOfParticle, iterations, flatTrueParams, whichModel):
    AllTrueDynamics = []
    for i in range(len(StartXY)):
        AllTrueDynamics.append(
            relaxSystem(StartXY[i], typeOfParticle[i], DeltaT, flatTrueParams, iterations, whichModel))
    return np.array(AllTrueDynamics)


def loss(parameters, tSample, observationIndex, iterations, whichModel):
    tempStartXY = TrueDynamics[observationIndex][tSample]  # getting XY_coordinates to send as initial positions
    tempParticleType = typeOfParticle[observationIndex]  # getting type of particles from the chosen observations

    #This loss is sum of two losses, where loss1 is difference between true dynamics and neural network prediction, and loss2
    #is the difference between neural network prediction and physical model prediction
    #whichModel = 0 - Physical model
    #whichModel = 1 - Neural Network
    #whichModel = 2 - combined loss
    if whichModel == 2:
        loss1 = np.mean(np.abs(TrueDynamics[observationIndex, tSample:iterations + tSample + 1, :] - relaxSystem(
            tempStartXY, tempParticleType, DeltaT, parameters[noOfPhysicalModelParams:], iterations, 1)))

        loss2 = np.mean(np.abs(relaxSystem(
            tempStartXY, tempParticleType, DeltaT, parameters[noOfPhysicalModelParams:], iterations, 1) - relaxSystem(
            tempStartXY, tempParticleType, DeltaT, parameters[:noOfPhysicalModelParams], iterations, 0)))

        return loss1 + loss2

    #this loss is the difference between true dynamics and prediction of whichever model is in action(physical or neural network)
    if whichModel == 0 or whichModel == 1:
        return np.mean(np.abs(TrueDynamics[observationIndex, tSample:iterations + tSample + 1, :] - relaxSystem(
            tempStartXY, tempParticleType, DeltaT, parameters, iterations, whichModel)))


def solveByAD(grad, whichModel, parameters, batchSize, path=PATH, callback=None, num_iters=100, step_size=0.001, b1=0.9,
              b2=0.999,
              eps=10 ** -8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(parameters))
    v = np.zeros(len(parameters))

    # to keep track of bestLoss, paramsAtBestLoss and earlyStopping
    bestLoss = 1000
    bestParams = []
    earlystopping = 0

    count = 0
    allSamples = []
    allParamsPhysicalModel = []
    lossList = []

    lossInitial = loss(parameters, 0, 0, MaxIterations, whichModel)

    lossList.append(lossInitial)

    if (whichModel == 0):
        allParamsPhysicalModel = [parameters]

    #splitting the array to only take the physical modelparameters and not neural network parameters
    elif (whichModel == 2):
        allParamsPhysicalModel = [parameters[:noOfPhysicalModelParams]]

    for i in range(0, int(num_iters / batchSize)):

        # accumulate the gradient over batchsize random sub samples
        g = np.zeros(len(parameters))
        for j in range(0, batchSize):  # update the parameters based on the gradient from this time-slice sample
            timeSteps = SampleTimeWindow

            observationIndex = random.randint(0, N - 1)# generate a random observation index
            tSample = random.randint(0, MaxIterations - timeSteps)
            allSamples.append(tSample)
            g = g + grad(parameters, tSample, observationIndex, timeSteps, whichModel)

        g = np.divide(g, batchSize)  # average gradient
        count = count + 1  # total number of updates
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g ** 2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1 ** (count + 1))  # Bias correction.
        vhat = v / (1 - b2 ** (count + 1))
        # update the parameters.
        gradient = mhat / (np.sqrt(vhat) + eps)
        parameters = parameters - step_size * gradient

        # clip when using physical force
        if (whichModel == 0):
            parameters = np.clip(parameters, 0.05, 100)  # parameters cannot be zero

        # clip only the physical parameters if combined parameters received
        if (whichModel == 2):
            parameters = np.array(parameters)
            parameters[:noOfPhysicalModelParams] = np.clip(parameters[:noOfPhysicalModelParams], 0.05, 100)
            parameters = totuple(parameters)

        # appending the updated parameters to parameter list
        if (whichModel == 0):
            allParamsPhysicalModel.append(parameters)
        if (whichModel == 2):
            allParamsPhysicalModel.append(parameters[:noOfPhysicalModelParams])

        if (count % DisplaySample == 0) and callback:
            currLoss = loss(parameters, 0, 0, MaxIterations, whichModel)
            lossList.append(currLoss)

            if (currLoss < bestLoss):
                bestLoss = currLoss
                bestParams = parameters
                earlystopping = 0
            else:
                earlystopping = earlystopping + 1

            print("iteration = %d loss = %.4f" % (count, currLoss))
            # callback to save images after every displaySample runs
            callback(parameters, lossList=lossList, count=count, allParamsPhysicalModel=allParamsPhysicalModel,
                     allSamples=allSamples, whichModel=whichModel, path=path)

        #if no decrease in loss for 1000 iterations, then stop learning and return
        if (earlystopping * DisplaySample == 1000):
            earlystopping = i #i is the iteration count
            return earlystopping, bestParams, bestLoss
    callback(parameters, lossList=lossList, count=count, allParamsPhysicalModel=allParamsPhysicalModel,
             allSamples=allSamples, whichModel=whichModel, path=path)

    #if early stopping doesnt occur, just return i, i.e. iteration count along with the loss and parameters
    return i, bestParams, bestLoss


#####################   BIOMECHANICAL MODEL OF PARTICLE MOVEMENT #################################

def relaxSystem(XY, particleType, deltaT, parameters, howManyIterations, whichModel):
    # applies a fixed number of iterations to the particles
    allCoords = []
    allCoords.append(XY)
    for i in range(0, howManyIterations):
        (XY, movement) = relaxForcesOne(XY, particleType, deltaT, parameters, whichModel)
        allCoords.append(XY)
    return np.array(allCoords)


def relaxForcesOne(XY, particleType, deltaT, parameters, whichModel):
    # calculate the forces and moves the particles for one iteration
    # (radius, boundMultiplier, springConstant, stiffnessMultiplier) = parameters
    # calculate the force between each pair of particles
    # accumulate the force on each particle due to each other particle
    # print("inside relaxforceone- paramstypeNN", paramsTypeNN)
    forceMatrix = []
    for i in range(0, len(XY)):
        forceOni = 0
        for j in range(0, len(XY)):
            if i == j:
                force = 0
                continue

            #if whichModel is 0, we calculate physical force
            #depending upon the types of both the cells, appropriate parameters are sent to the forceBetween function
            if (whichModel == 0):
                if (particleType[i] == 0 and particleType[j] == 0):
                    parametersNested = repack(getTrueParameters(), parameters)
                    force = forceBetween(XY[i], XY[j], parametersNested[0][0], parametersNested[0][0],
                                         parametersNested[0][1])
                elif (particleType[i] == 0 and particleType[j] == 1):
                    parametersNested = repack(getTrueParameters(), parameters)
                    force = forceBetween(XY[i], XY[j], parametersNested[0][0], parametersNested[1][0],
                                         parametersNested[0][2])
                elif (particleType[i] == 1 and particleType[j] == 0):
                    parametersNested = repack(getTrueParameters(), parameters)
                    force = forceBetween(XY[i], XY[j], parametersNested[1][0], parametersNested[0][0],
                                         parametersNested[0][2])
                else:
                    parametersNested = repack(getTrueParameters(), parameters)
                    force = forceBetween(XY[i], XY[j], parametersNested[1][0], parametersNested[1][0],
                                         parametersNested[1][1])

                # print("------------------Physicalforce-------------------------------", force)

            #when whichModel is 1, we calculate force using the neural network
            if (whichModel == 1):
                force = forceNN(vectorMag(XY[i] - XY[j]), parameters, particleType[i], particleType[j])
                # print(force)
                # print("------------------NNforce-------------------------------", force)

            unitVector = np.divide(XY[i] - XY[j], vectorMag(XY[i] - XY[j]))
            forceOni = forceOni + np.multiply(unitVector, force)
        forceMatrix.append(forceOni)
    forceMatrix = np.array(forceMatrix)
    # calculate movements on each particle
    movementList = []
    for i in range(0, len(XY)):
        movement = np.divide(np.multiply(forceMatrix[i], deltaT), ViscousConstant)
        movementList.append(movement)
    # add noise
    move = np.array(movementList) + np.random.normal(0, NoiseMag, size=(NumberOfParticles, 2))
    # move the particles
    return (XY + move, move)


def forceNN(distance, nnParameters, type0, type1):
    # changing type 0 to -1 to avoid giving 0 in the input vector
    if type0 == 0:
        type0 = -1
    if type1 == 0:
        type1 = -1

    # repacking the flat NN parameters in the shape of initial nested parameters
    nnParameters = repack(nestedInitialNNParameters, nnParameters)

    # forward propagation through our network
    #for 2 hidden layers
    inputToNN = (distance, type0, type1)
    nn_distance_samples.append(distance)
    z = np.dot(inputToNN, nnParameters[0][0]) + nnParameters[1][0]
    z2 = tanh(z[0])  # activation function
    z3 = np.dot(z2, nnParameters[0][1]) + nnParameters[1][1]
    z4 = tanh(z3[0])
    z5 = np.dot(z4, nnParameters[0][2]) + nnParameters[1][2]
    y = z5[0]
    return y[0]


def tanh(s):
    lengthOfInput = len(s)
    activation = []
    for k in range(0, lengthOfInput):
        value = (np.exp(s[k]) - np.exp(-s[k])) / (np.exp(s[k]) + np.exp(-s[k]))
        activation.append(value)
    return activation


def forceBetween(xy0, xy1, radius1, radius2, parameters):
    r1 = radius1
    r2 = radius2

    # returns the magnitude of the force between two particles based on distance
    return forceBetweenDistance(vectorMag(xy0 - xy1), r1, r2, parameters)


def forceBetweenDistance(distance, r1, r2, parameters):
    # model used in biofilm wrinkling
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0191089
    # use a smooth threashold rather than an if statement for learning
    (boundMultiplier, springConstant, stiffnessMultiplier) = parameters
    x = r1 + r2 - distance
    # if distance > boundMultiplier*radius:
    #     return 0.0
    # else:
    #     return springConstant * x * np.tanh(stiffness * np.abs(x))
    # negative is attractive force, possitive is repelent force
    stiffness = ViscousConstant * stiffnessMultiplier
    return threashold((r1 + r2) * boundMultiplier - distance, BondSmoothness) * springConstant * x * np.tanh(
        stiffness * np.abs(x))


def threashold(diff, bondSmoothness):
    # moves smoothly between 0 when diff is neg to 1 when diff is +
    return 1.0 / (1 + np.exp(-diff / bondSmoothness))


def vectorMag(oneXY):
    return np.sqrt(oneXY[0] * oneXY[0] + oneXY[1] * oneXY[1])


def forceBetweenDistanceIF(distance, r1, r2, parameters):
    # model used in biofilm wrinkling
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0191089
    # use a smooth threashold rather than an if statement for learning
    (boundMultiplier, springConstant, stiffnessMultiplier) = parameters
    stiffness = ViscousConstant * stiffnessMultiplier
    x = (r1 + r2) - distance
    if distance > boundMultiplier * (r1 + r2):
        return 0.0
    else:
        return springConstant * x * np.tanh(stiffness * np.abs(x))
    # negative is attractive force, possitive is repelent force


##########################################################################

# SUPPORT CODE
def saveImageOne(fig, name, iteration, PATH_2):
    fileName = name + "_" + str(iteration).rjust(4, '0')
    onePath = PATH_2
    if not os.path.exists(onePath):
        os.makedirs(onePath)
    # print(onePath + fileName + '.png')
    fig.savefig(onePath + fileName + '.png', dpi=100)


def printParameters(params):
    names = "       "
    values = "Found= "
    trueValues = "True=  "
    for i in range(0, len(params)):
        names = names + ParameterNames[i].ljust(13)
        values = values + ("%.4f" % params[i]).ljust(13)
        trueValues = trueValues + ("%.4f" % flatTrueParams[i]).ljust(13)

    print(names)
    print(values)
    print(trueValues)


# DISPLAY
def plotForce(r1, r2, parameters, type1, type2):
    fig = plt.figure(figsize=(16, 16), facecolor='white')
    plot = fig.add_subplot(1, 1, 1, frameon=True)
    plot.cla()
    plot.set_title('Force as a function of separation distance')
    plot.set_xlabel('distance')
    plot.set_ylabel('Force between type' + str(type1) + ' and type' + str(type2))
    # ax_loss.set_aspect(1.0)
    distances = np.arange(0, 12, 0.1)
    forces = [forceBetweenDistance(distances[i], r1, r2, parameters) for i in range(0, len(distances))]
    forcesIF = [forceBetweenDistanceIF(distances[i], r1, r2, parameters) for i in range(0, len(distances))]
    plot.plot(distances, forces, linestyle='solid', color='m', linewidth=8, label='Original Model', alpha=0.6)
    plot.plot(distances, forcesIF, linestyle='solid', color='b', linewidth=6, label='Smooth Model', alpha=0.6)
    plot.legend(loc='upper right')
    if (min(forces + forcesIF) == max(forces + forcesIF)):
        plot.set_ylim(-0.5, 0.5)
    else:
        plot.set_ylim(min(forces + forcesIF) - 1, max(forces + forcesIF) + 1)
    plt.show()
    # plt.savefig('forcePlot'+str(type1)+str(type2))


def generateAnimation(dynamics, downSample=1, lastOnly=False, name="run", path=PATH):
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax_spaceTime1 = fig.add_subplot(1, 1, 1, frameon=True)
    ax_spaceTime1.cla()
    ax_spaceTime1.set_title("Particle movement 2D")
    ax_spaceTime1.set_ylim(0 - 0.5, XYrange + 0.5)
    ax_spaceTime1.set_xlim(0 - 0.5, XYrange + 0.5)
    # ax_spaceTime1.set_xlabel('X position')
    # ax_spaceTime1.set_ylabel('Y position')
    ax_spaceTime1.set_aspect(1.0)
    for iteration in range(1, len(dynamics), downSample):
        if lastOnly and iteration != len(dynamics) - 1:
            continue
        byParticle = dynamics[0:iteration].T
        for i in range(0, byParticle.shape[1]):
            if (typeOfParticle[0][i] == 1):
                color = 'b'
            else:
                color = 'r'
            ax_spaceTime1.plot(byParticle[0, i], byParticle[1, i], color, linewidth=3)
            ax_spaceTime1.scatter([byParticle[0, i, -1]], [byParticle[1, i, -1]], marker="o", color=color, s=4500.0,
                                  alpha=0.2)
        # plt.show()
        # plt.pause(0.001)
        saveImageOne(fig, "Observe_" + name, iteration, path)
        if iteration != len(dynamics) - 1:
            fig.clf()
        ax_spaceTime1 = fig.add_subplot(1, 1, 1, frameon=True)
        ax_spaceTime1.cla()
        ax_spaceTime1.set_title("Particle movement 2D")
        ax_spaceTime1.set_ylim(0 - 0.5, XYrange + 0.5)
        ax_spaceTime1.set_xlim(0 - 0.5, XYrange + 0.5)
        # ax_spaceTime1.set_xlabel('X position')
        # ax_spaceTime1.set_ylabel('Y position')
        ax_spaceTime1.set_aspect(1.0)


def displayDynamics(parameters, name="run", lossList=[], count=0, allParamsPhysicalModel=[], allSamples=[],
                    whichModel=0, path=PATH):
    # loss0 = loss(parameters, 0, 0, MaxIterations, whichModel)

    ################## Particle Movement
    if (whichModel == 0):
        learnedDynamics = relaxSystem(StartXY[0], typeOfParticle[0], DeltaT, parameters, MaxIterations, 0)
    if (whichModel == 1):
        learnedDynamics = relaxSystem(StartXY[0], typeOfParticle[0], DeltaT, parameters, MaxIterations, 1)
    if (whichModel == 2):
        learnedDynamics = relaxSystem(StartXY[0], typeOfParticle[0], DeltaT, parameters[:noOfPhysicalModelParams],
                                      MaxIterations, 0)

    ax_spaceTime1.cla()

    ax_spaceTime1.set_title("Particle movement 2D")
    ax_spaceTime1.set_ylim(0, XYrange)
    ax_spaceTime1.set_xlim(0, XYrange)
    ax_spaceTime1.set_xlabel('X position')
    ax_spaceTime1.set_ylabel('Y position')
    ax_spaceTime1.set_aspect(1.0)

    byParticle = TrueDynamics[0].T
    firstType0 = 0
    firstType1 = 0
    flag0 = 0
    flag1 = 0
    for i in range(0, byParticle.shape[1]):
        if (typeOfParticle[0][i] == 1):
            color = 'b'
            if (flag1 == 0):
                firstType1 = i
                flag1 = 1
        else:
            color = 'r'
            if (flag0 == 0):
                firstType0 = i
                flag0 = 1
        ax_spaceTime1.plot(byParticle[0][i], byParticle[1][i], color, linewidth=3, alpha=0.5)
        ax_spaceTime1.scatter([byParticle[0][i][-1]], [byParticle[1][i][-1]], marker="o", color=color, s=1500.0,
                              alpha=0.1)
    ax_spaceTime1.plot(byParticle[0][firstType0], byParticle[1][firstType0], 'r', linewidth=2, label='Type 0 True')
    ax_spaceTime1.plot(byParticle[0][firstType1], byParticle[1][firstType1], 'b', linewidth=2, label='Type 1 True')

    byParticle = learnedDynamics.T
    for i in range(0, byParticle.shape[1]):
        if (typeOfParticle[0][i] == 1):
            color = 'g'
        else:
            color = 'm'
        ax_spaceTime1.plot(byParticle[0][i], byParticle[1][i], color, linewidth=3, alpha=0.5)
        ax_spaceTime1.scatter([byParticle[0][i][-1]], [byParticle[1][i][-1]], marker="o", color=color, s=1500.0,
                              alpha=0.1)
    ax_spaceTime1.plot(byParticle[0][firstType0], byParticle[1][firstType0], 'm', linewidth=2, label='Type 0 Predicted')
    ax_spaceTime1.plot(byParticle[0][firstType1], byParticle[1][firstType1], 'g', linewidth=2, label='Type 1 Predicted')
    ax_spaceTime1.legend(loc='upper left')

    ################## LOSS graph
    ax_loss.cla()
    ax_loss.set_title('Train Loss')
    ax_loss.set_xlabel('learning iterations')
    ax_loss.set_ylabel('loss')
    ax_loss.set_yscale('log')
    # ax_loss.set_aspect(1.0)
    if len(lossList) > 1:
        # Capture loss each display iteration
        iterations = np.arange(0, len(lossList), 1) * DisplaySample
        ax_loss.step(iterations, lossList, '-', where='mid', label='Current', linestyle='solid', color='b',
                     linewidth=1)
        ax_loss.set_xlim(0, iterations.max())
        ax_loss.set_aspect(abs(iterations.max() - iterations.min()) / abs(max(lossList) - min(lossList)))

    ################ force function learned

    # ax_force.set_title('Force between particles')

    distances = np.arange(0, 15, 0.1)

    # getting true forces between all types of particles
    trueForces00 = [forceBetweenDistance(distances[i], TrueParameters[0][0], TrueParameters[0][0], TrueParameters[0][1])
                    for i
                    in range(0, len(distances))]
    trueForces01 = [forceBetweenDistance(distances[i], TrueParameters[0][0], TrueParameters[1][0], TrueParameters[0][2])
                    for i in range(0, len(distances))]
    trueForces11 = [forceBetweenDistance(distances[i], TrueParameters[1][0], TrueParameters[1][0], TrueParameters[1][1])
                    for i in range(0, len(distances))]

    # getting forces from physical model
    if (whichModel == 0):
        nestedLearnedParams = repack(getTrueParameters(), parameters)

        predictedForces00 = [forceBetweenDistance(distances[i], nestedLearnedParams[0][0], nestedLearnedParams[0][0],
                                                  nestedLearnedParams[0][1]) for i
                             in range(0, len(distances))]

        predictedForces01 = [forceBetweenDistance(distances[i], nestedLearnedParams[0][0], nestedLearnedParams[1][0],
                                                  nestedLearnedParams[0][2]) for i
                             in range(0, len(distances))]

        predictedForces11 = [forceBetweenDistance(distances[i], nestedLearnedParams[1][0], nestedLearnedParams[1][0],
                                                  nestedLearnedParams[1][1]) for i
                             in range(0, len(distances))]
    # getting forces by NN
    if (whichModel == 1):
        predictedForces00 = [forceNN(distances[i], parameters, 0, 0) for i in range(0, len(distances))]

        predictedForces01 = [forceNN(distances[i], parameters, 0, 1) for i in range(0, len(distances))]

        predictedForces11 = [forceNN(distances[i], parameters, 1, 1) for i in range(0, len(distances))]

    # getting forces from physical model
    if (whichModel == 2):
        nestedLearnedParams = repack(getTrueParameters(), parameters[:noOfPhysicalModelParams])

        predictedForces00 = [forceBetweenDistance(distances[i], nestedLearnedParams[0][0], nestedLearnedParams[0][0],
                                                  nestedLearnedParams[0][1]) for i
                             in range(0, len(distances))]

        predictedForces01 = [forceBetweenDistance(distances[i], nestedLearnedParams[0][0], nestedLearnedParams[1][0],
                                                  nestedLearnedParams[0][2]) for i
                             in range(0, len(distances))]

        predictedForces11 = [forceBetweenDistance(distances[i], nestedLearnedParams[1][0], nestedLearnedParams[1][0],
                                                  nestedLearnedParams[1][1]) for i
                             in range(0, len(distances))]

    ax_f00.cla()
    ax_f00.set_title('Force between type 0-0')
    ax_f00.set_xlabel('distance between particles')
    ax_f00.set_ylabel('Force')
    ax_f00.plot(distances, trueForces00, '-', label='True', linestyle='solid', color='r', linewidth=2)
    ax_f00.plot(distances, predictedForces00, '-', label='Predicted', linestyle='solid', color='g', linewidth=2)
    ax_f00.set_ylim(min(trueForces00) - 0.1, max(trueForces00) + 0.1)
    ax_f00.legend(loc='upper right')
    ax_f00.set_aspect(abs(max(distances) - min(distances)) / abs(max(trueForces00) - min(trueForces00) + 0.01))

    ax_f01.cla()
    ax_f01.set_title('Force between type 0-1')
    ax_f01.set_xlabel('distance between particles')
    ax_f01.set_ylabel('Force')
    ax_f01.plot(distances, trueForces01, '-', label='True', linestyle='solid', color='r', linewidth=2)
    ax_f01.plot(distances, predictedForces01, '-', label='Predicted', linestyle='solid', color='g', linewidth=2)
    # ax_f01.set_ylim(-0.5 , 0.5)
    ax_f01.set_ylim(min(trueForces01) - 0.1, max(trueForces01) + 0.1)
    ax_f01.set_xlim(min(distances) - 0.5, max(distances) + 0.5)
    ax_f01.legend(loc='upper right')
    ax_f01.set_aspect(abs(max(distances) - min(distances)) / abs(max(trueForces01) - min(trueForces01) + 0.01))

    ax_f11.cla()
    ax_f11.set_title('Force between type 1-1')
    ax_f11.set_xlabel('distance between particles')
    ax_f11.set_ylabel('Force')
    ax_f11.plot(distances, trueForces11, '-', label='True', linestyle='solid', color='r', linewidth=2)
    ax_f11.plot(distances, predictedForces11, '-', label='Predicted', linestyle='solid', color='g', linewidth=2)
    ax_f11.set_ylim(min(trueForces11) - 0.1, max(trueForces11) + 0.1)
    ax_f11.legend(loc='upper right')
    ax_f11.set_aspect(abs(max(distances) - min(distances)) / abs(max(trueForces11) - min(trueForces11) + 0.01))

    # plt.show()
    # plt.draw()
    # plt.pause(0.001)
    saveImageOne(figForce, "Force_" + name, len(lossList), path + "force/")
    plt.close(figForce)

    ############PARAMETERS
    if whichModel == 0 or whichModel == 2:
        if len(allParamsPhysicalModel) > 1:

            byParameter = np.array(allParamsPhysicalModel).T
            ax_parameters.cla()
            ax_parameters.set_title('Model parameter Error')
            ax_parameters.set_xlabel('Learning iteration')
            ax_parameters.set_ylabel('Parameter value')
            # ax_parameters.set_yscale('log')
            ax_loss.set_aspect(1.0)

            iterations = np.arange(0, len(allParamsPhysicalModel), 1)

            colors = ['darkgoldenrod', 'g', 'r', 'c', 'm', 'darkorange', 'k', 'yellow', 'blue', 'maroon', 'lime']
            for i in range(0, len(flatTrueParams)):
                thisTrue = np.array([flatTrueParams[i] for j in iterations])
                error = (byParameter[i] - thisTrue)
                # if error < 0.005:
                #     linewidth = 5 +
                linewidth = 2
                ax_parameters.plot(iterations, error, '-', label=ParameterNames[i], linestyle='solid', color=colors[i],
                                   linewidth=linewidth)
                ax_parameters.fill_between(iterations, error, step="pre", color=colors[i], alpha=0.1)
                # ax_parameters.plot(iterations, [trueParameters[i] for j in iterations], '-', linestyle = 'dotted', color = colors[i], linewidth=2)
            ax_parameters.set_ylim(-8.0, 8.0)  # max(trueParameters)+2)
            ax_parameters.set_aspect(abs(len(iterations)) / abs(8))
            ax_parameters.legend(loc='lower right')

    # plt.show()
    # plt.draw()
    # plt.pause(0.001)
    if (whichModel == 0):
        printParameters(parameters)
    if (whichModel == 2):
        printParameters(parameters[:noOfPhysicalModelParams])

    saveImageOne(fig, "Learn_" + name, len(lossList), path + "movement/")
    plt.close(fig)
    # return this loss
    # return loss0


############## Display
fig = plt.figure(figsize=(24, 8), facecolor='white')
ax_spaceTime1 = fig.add_subplot(1, 4, 1, frameon=True)
ax_loss = fig.add_subplot(1, 4, 2, frameon=True)
# ax_force = fig.add_subplot(1, 4, 3, frameon=True)
ax_parameters = fig.add_subplot(1, 4, 3, frameon=True)

figForce = plt.figure(figsize=(24, 8), facecolor='white')
ax_f00 = figForce.add_subplot(1, 3, 1, frameon=True)
ax_f01 = figForce.add_subplot(1, 3, 2, frameon=True)
ax_f11 = figForce.add_subplot(1, 3, 3, frameon=True)

############## model parameters
MaxIterations = int(60)
DeltaT = 0.01 * 2
NoiseMag = 0.03  # 0.08 #0.01 #can go to 0.05
NumberOfParticles = 20
ViscousConstant = 0.5
XYrange = 12
BondSmoothness = 0.5
ParameterNames = ["radiusD0", "boundX00", "springC00", "stiffnessX00", "boundX01", "springC01", "stiffnessX01",
                  "radiusD1", "boundX11", "springC11", "stiffnessX11"]
N = 1 # N number of observations
TypesOfParticles = 2

############## Initializing the particle location and type
# generate N different simulation observations
StartXY = np.ndarray(shape=(N, NumberOfParticles, 2))
for j in range(0, N):
    for i in range(0, NumberOfParticles):
        StartXY[j][i] = [np.random.uniform(low=1.5, high=XYrange - 1.5), np.random.uniform(low=0.2, high=XYrange - 0.2)]
StartXY[0][0][0] = 1.0 / 2
StartXY[0][0][1] = 1.0 / 2
StartXY[0][1][0] = XYrange - 1.0 / 2
StartXY[0][1][1] = XYrange - 1.0 / 2
StartXY[0][3][0] = 1.0 / 2
StartXY[0][3][1] = XYrange - 1.0 / 2
StartXY[0][4][0] = XYrange - 1.0 / 2
StartXY[0][4][1] = 1.0 / 2

# To save particle positions as pickle file
# with open(PATH+'/startxy.pkl', 'wb') as f:
#     pickle.dump(StartXY, f)

# To load particle positions from pickle file
# with open('C:/Users/Namita/Desktop/research/NewExperimentsPM/2.0/regenerateNewParam/startxy.pkl', 'rb') as f:
#     StartXY = pickle.load(f)

# generating the types of particles(0 or 1)
typeOfParticle = np.ndarray(shape=(N, NumberOfParticles))
for i in range(0, N):
    for j in range(0, NumberOfParticles):
        typeOfParticle[i][j] = random.randint(0, TypesOfParticles - 1)


############## True Parameters
def getTrueParameters():
    TrueParams = ((2, (2, 1.5, 0.4), (1, 0.5, 0.4)), (2, (2, 1.5, 0.4)))  # cellSorting1
    # TrueParams = ((2, (4, 1.5, 0.4), (1, 0.5, 0.4)), (2, (4, 1.5, 0.4))) #cellSorting2
    # TrueParams = ((3, (1, 1.5, 0.4), (4, 0.5, 0.4)), (3, (1, 1.5, 0.4))) #alternatePattern
    return TrueParams


TrueParameters = getTrueParameters()
flatTrueParams = tuple(unpack(getTrueParameters()))


############## Parameters to be learned


def getInitialParameters():
    radius0 = np.random.uniform(low=0.0, high=8.0)
    boundMultiplier00 = np.random.uniform(low=0.0, high=8.0)
    springConstant00 = np.random.uniform(low=0.0, high=8.0)
    stiffnessMultiplier00 = np.random.uniform(low=0.0, high=8.0)

    boundMultiplier01 = np.random.uniform(low=0.0, high=8.0)
    springConstant01 = np.random.uniform(low=0.0, high=8.0)
    stiffnessMultiplier01 = np.random.uniform(low=0.0, high=8.0)

    radius1 = np.random.uniform(low=0.0, high=8.0)
    boundMultiplier11 = np.random.uniform(low=0.0, high=8.0)
    springConstant11 = np.random.uniform(low=0.0, high=8.0)
    stiffnessMultiplier11 = np.random.uniform(low=0.0, high=8.0)

    #packing the 11 model parameters
    initialParams = ((radius0, (boundMultiplier00, springConstant00, stiffnessMultiplier00),
                      (boundMultiplier01, springConstant01, stiffnessMultiplier01)),
                     (radius1, (boundMultiplier11, springConstant11, stiffnessMultiplier11)))
    return initialParams


############# Learning parameters
nn_distance_samples = []  # for distance histogram
DisplaySample = 1  # capture the loss every i learning iterations
batchSize = 1  # higher is better with noise
SampleTimeWindow = 1  # how many time steps to simulated during learning, smaller is better with noise
gradientFunction = grad(loss)
learningRate = 0.007  # 0.002 # # 2 #4
learningIterations = 3
NNDimensions = (3, 2, 1, 1)

# Physical model initial parameters(nested and flat)
nestedInitialParameters = getInitialParameters()
flatInitialParameters = tuple(unpack(nestedInitialParameters))

# No of physical model parameters
noOfPhysicalModelParams = len(flatInitialParameters)

# initial NN parameters(nested and flat)
nestedInitialNNParameters = initial_nn_weight(NNDimensions)
flatInitialNNParameters = tuple(unpack(nestedInitialNNParameters))

# initial combined parameters
combinedNestedParams = (nestedInitialParameters, nestedInitialNNParameters)
flatCombinedParams = tuple(unpack(combinedNestedParams))

# This variable decides which model to run; 0 - only run physical Model ; 1- only run NN ; 2-run both combined
# when running learner with combined parameters the animation will be saved for physical model only
whichModel = 0

if (whichModel == 0):
    paramsToLearn = flatInitialParameters
elif (whichModel == 1):
    paramsToLearn = flatInitialNNParameters
else:
    paramsToLearn = flatCombinedParams

#this is used to save the excel files
wb = Workbook()

# uncomment this line to generate the simulation for 1st observation out of N
# if you write TrueDynamics[1] in place of TrueDynamics[0] you can see the simulation for 2nd Observation and so on
trueSimulationPath = PATH + "trueSimulation/"
TrueDynamics = generateTrueDynamics(StartXY, typeOfParticle, MaxIterations, flatTrueParams, 0)
generateAnimation(TrueDynamics[0], 1, name="05 xbound ", path=trueSimulationPath)  # MaxIterations-1

# # #uncomment this block to see the trueForces between all combination of particles
# for i in range(0,len(TrueParameters)):
#     flag = 1
#     for j in range(i, len(TrueParameters)):
#         r1 = TrueParameters[i][0]
#         r2 = TrueParameters[j][0]
#         params = TrueParameters[i][flag]
#         flag += 1
#         plotForce(r1, r2, params, i, j)


for randomRun in range(0, 3):
    PATH_2 = PATH + "_" + str(randomRun) + "/"
    paramsCurr = tuple(unpack(getInitialParameters()))
    dynamicsWithRandomParam = generateTrueDynamics(StartXY, typeOfParticle, MaxIterations, paramsCurr, 0)
    randomSimulationPath = PATH_2 + "SimulationWithInitialParam/"
    generateAnimation(dynamicsWithRandomParam[0], 1, name="Random_" + str(randomRun) + "_", path=randomSimulationPath)

    '''
    uncomment this block to start the learner 
    '''
    earlyStoppingAt, paramsAtBestLoss, bestLossAchieved = solveByAD(gradientFunction, whichModel, paramsCurr, batchSize,
                                                                    PATH_2, callback=displayDynamics,
                                                                    num_iters=batchSize * learningIterations,
                                                                    step_size=learningRate)

    ws = wb.active
    ws.append([randomRun, str(paramsCurr), bestLossAchieved, str(paramsAtBestLoss), earlyStoppingAt])
    wb.save("testing.xlsx")

content = "True Parameters - " + str(TrueParameters) + "\nDeltaT-" + str(DeltaT) + "\nXYRange-" + str(
XYrange) + "\nBatch Size-" + str(batchSize) + "\nSampleTimeWindow-" + str(
SampleTimeWindow) + "\nlearning rate-" + str(learningRate) \
      + "\nLearningIterations - " + str(learningIterations) + "\nNoOfObservations - " + str(
N) + "\nNoOfParticles - " + str(NumberOfParticles) + "\nNOise- " + str(NoiseMag)

filePath = PATH + "/configuration.txt"
f = open(filePath, "w")
f.write(content)