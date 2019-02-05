import numpy as np
import matplotlib.pyplot as plt

from big_data.data_utils import *

from q3_sgd import load_saved_params, sgd
from q4_softmaxreg import softmaxRegression, getSentenceFeature, accuracy, softmax_wrapper

# Try different regularizations and pick the best!
# NOTE: fill in one more "your code here" below before running!
#REGULARIZATION = None   # Assign a list of floats in the block below
### YOUR CODE HERE
# #REGULARIZATION = [0.001, 0.0005, 0.0000002, 0.1, 0.00006, 0.0000031, 0.000000008]
#REGULARIZATION = [3e-1, 2e-2, 1e-3, 4e-4, 2e-5, 1e-6, 4e-7]
#REGULARIZATION = [3e-1, 2e-4, 1e-5, 4e-5, 2e-5, 1e-3, 6e-3, 7e-7]
REGULARIZATION = [1e-2, 1e-3, 7e-4, 4e-4, 1e-4, 7e-5, 4e-5, 1e-5, 7e-6, 4e-6, 1e-6, 1e-7]
#raise NotImplementedError
### END YOUR CODE

# Load the dataset
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# Load the word vectors we trained earlier
_, wordVectors0, _ = load_saved_params()
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])
dimVectors = wordVectors.shape[1]

# Load the train set
trainset = dataset.getTrainSentences()
nTrain = len(trainset)
trainFeatures = np.zeros((nTrain, dimVectors))
trainLabels = np.zeros((nTrain,), dtype=np.int32)
for i in xrange(nTrain):
    words, trainLabels[i] = trainset[i]
    trainFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

# Prepare dev set features
devset = dataset.getDevSentences()
nDev = len(devset)
devFeatures = np.zeros((nDev, dimVectors))
devLabels = np.zeros((nDev,), dtype=np.int32)
for i in xrange(nDev):
    words, devLabels[i] = devset[i]
    devFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

# Try our regularization parameters
results = []
bestReg = None
bestWt = None
bestAcc = 0.0
for regularization in REGULARIZATION:
    random.seed(3141)
    np.random.seed(59265)
    weights = np.random.randn(dimVectors, 5)
    print "Training for reg=%f" % regularization

    # We will do batch optimization
    weights = sgd(lambda weights: softmax_wrapper(trainFeatures, trainLabels,
        weights, regularization), weights, 3.0, 10000, PRINT_EVERY=100)

    # Test on train set
    _, _, pred = softmaxRegression(trainFeatures, trainLabels, weights)
    trainAccuracy = accuracy(trainLabels, pred)
    print "Train accuracy (%%): %f" % trainAccuracy

    # Test on dev set
    _, _, pred = softmaxRegression(devFeatures, devLabels, weights)
    devAccuracy = accuracy(devLabels, pred)
    print "Dev accuracy (%%): %f" % devAccuracy

    # Save the results and weights
    if bestAcc < devAccuracy:
        bestAcc = devAccuracy
        bestReg = regularization
        bestWt = weights
    results.append({
        "reg" : regularization,
        "weights" : weights,
        "train" : trainAccuracy,
        "dev" : devAccuracy})

# Print the accuracies
print ""
print "=== Recap ==="
print "Reg\t\tTrain\t\tDev"
for result in results:
    print "%E\t%f\t%f" % (
        result["reg"],
        result["train"],
        result["dev"])
print ""


# Pick the best regularization parameters

### YOUR CODE HERE
bestREGULARIZATION = bestReg
bestWEIGHTS = bestWt
#raise NotImplementedError
### END YOUR CODE

# Test your findings on the test set
testset = dataset.getTestSentences()
nTest = len(testset)
testFeatures = np.zeros((nTest, dimVectors))
testLabels = np.zeros((nTest,), dtype=np.int32)
for i in xrange(nTest):
    words, testLabels[i] = testset[i]
    testFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

_, _, pred = softmaxRegression(testFeatures, testLabels, bestWEIGHTS)
print "Best regularization value: %E" % bestREGULARIZATION
print "Test accuracy (%%): %f" % accuracy(testLabels, pred)


# Make a plot of regularization vs accuracy
plt.plot(REGULARIZATION, [x["train"] for x in results])
plt.plot(REGULARIZATION, [x["dev"] for x in results])
plt.xscale('log')
plt.xlabel("regularization")
plt.ylabel("accuracy")
plt.legend(['train', 'dev'], loc='upper left')
plt.savefig("q4_reg_v_acc.png")
#plt.show()