"""network2.py"""
import random

import numpy as np
import pickle as pick

# Filename is network2, rather than network1, because it is the second
# iteration of the network, a complete re-creation of network1 do to numerous issues
class Network:
    """Neural Network Class"""
    def tanh(self, x, layer):
        """activation function, such as sigmoid or tanh"""
        return np.tanh(x / self.k)

    def tanh_p(self, x, layer):
        """Derivative of the activation function"""
        return 1 / self.k * (1 - np.tanh(x / self.k) ** 2)
    
    def quadr_cost(self, goal):
        """Quadratic Cost Function"""
        return (goal - self.activs[-1]) ** 2
    def quadr_cost_p(self, goal):
        """Derivative of the Cost Function"""
        return -2 * (goal - self.activs[-1])
    def quadr_delta(self, goal):
        return self.cost_p(goal) * self.actpFuns[-1](self.z[-1], -1)
    def __init__(self, sizes, k = 1, reg_param = 0,
                actFuns = None, actpFuns = None,
                costFun = None, costpFun = None,
                deltaFun = None):
        """
        Initializes the networks using the sizes array. 
        The biases are initialize with a normal distribution, and the 
        weights are initialized with a constrained gaussian distribution
        actFuns/actpFuns allow for differing activation functions between layers
        All actFuncs and their derivatives take two arguments; value and layer
        """
        self.cost = costFun
        self.cost_p = costpFun
        self.sizes = sizes
        self.lay_ct = len(sizes)
        self.weights = [np.random.randn(x, y) / np.sqrt(x)
                         for x, y in zip(sizes[1:], sizes[:-1])]
        self.biases = [np.random.randn(y)
                        for y in sizes[1:]]
        self.k = k
        self.reg_param = reg_param

        if type(actFuns) == type(None):
            self.actFuns = [self.tanh for x in self.sizes[1:]]
        if len(self.actFuns) > len(self.sizes) - 1:
            self.actFuns = self.actFuns[:len(self.sizes) - 1]
        if len(self.actFuns) < len(self.sizes) - 1:
            while len(self.actFuns) < len(self.sizes) - 1:
                self.actFuns.append(self.act)

        if type(actpFuns) == type(None):
            self.actpFuns = [self.tanh_p for x in self.sizes[1:]]
        if len(self.actpFuns) > len(self.sizes) - 1:
            self.actpFuns = self.actFuns[:len(self.sizes) - 1]
        if len(self.actpFuns) < len(self.sizes) - 1:
            while len(self.actpFuns) < len(self.sizes) - 1:
                self.actpFuns.append(self.act)
        self.cost = costFun
        self.cost_p = costpFun
        self.deltaFun = deltaFun
        if type(costFun) == type(None):
            self.cost = self.quadr_cost
        if type(costpFun) == type(None):
            self.cost_p = self.quadr_cost_p
        if type(deltaFun) == type(None):
            self.deltaFun = self.quadr_delta
    def feedUp(self, inp, dropout = 2):
        """Feed up, or forwards, through the network
        Dropout is mainly for backprop() to use, and must be in the range 0-1
        It will set the the probability of a given neuron being
        'disabled', or simply set to 0. This is common practice,
        and helps improve the network's ability to generalize from a training set"""
        self.activs = [inp]
        self.z = []
        for i in range(1, self.lay_ct):
            self.z.append(self.weights[i - 1].dot(self.activs[-1])
                          +self.biases[i - 1])
            self.activs.append(self.actFuns[i - 1](self.z[-1], -1))
            # ==Implement dropout, only if the dropout variable is defined. Otherwise
            # random.random() never produces a value greater than 2, and thus no
            # dropout occurs
            for j in range(0, self.sizes[i]):
                rnum = random.random()
                if rnum > dropout:
                    self.activs[-1][j] = 0

    def backprop(self, inp, goal, dropProb):
        """
        Backpropogate through the network, return a tuple
        containing the resulting weight and bias partials. Whilst I won't
        go into exact detail of the math (as there is simply not enough space
        here to explain it), the gist is that you start by computing a special delta
        term by 'propogating' backwards through the network, and computing the
        corresponding weight and bias changes utilizing said delta term. Hence, the
        name "backpropogation"
        """
        self.feedUp(inp, dropProb)  # Get the activs/zs for the network.
        w_ch = [np.zeros(w.shape) for w in self.weights]  # start of with zeros
        b_ch = [np.zeros(b.shape) for b in self.biases]
        delta = self.deltaFun(goal)
        b_ch[-1] = delta
        # w_ch[-1] = delta.dot(self.activs[-2].transpose())
        for i in range(0, self.sizes[-1]):
            w_ch[-1][i] = delta[i] * self.activs[-2]
        for i in range(2, self.lay_ct):
            delta = delta.dot(self.weights[-i + 1]) * \
                self.actpFuns[-i](self.z[-i], -i)
            b_ch[-i] = delta
            # w_ch[-i] = delta.dot(self.activs[-i - 1].transpose())
            for j in range(0, self.sizes[-i]):
                w_ch[-i][j] = delta[j] * self.activs[-i - 1]
        return (b_ch, w_ch)

    def batchProp(self, inps, goals, dropProb):
        """Return the net weight/bias partials for the items within a batch"""
        w_ch = [np.zeros(w.shape)for w in self.weights]
        b_ch = [np.zeros(b.shape)for b in self.biases]
        for inp, goal in zip(inps, goals):
            # Inp is divided by 255 to reduce the values to between 0 and 1
            b, w = self.backprop(inp / 255, goal, dropProb)
            w_ch = [x + y for x, y in zip(w_ch, w)]
            b_ch = [x + y for x, y in zip(b_ch, b)]
        return (b_ch, w_ch)

    def run_batch(self, inps, goals, lRate, batch_size, dropProb):
        """Runs the input inps/goals as a batch,
         and assigns the changes to the network"""
        b_ch, w_ch = self.batchProp(inps, goals, dropProb)
        self.biases = [a - b * lRate / batch_size for
                       a, b in zip(self.biases, b_ch)]
        self.weights = [(1 - self.reg_param / batch_size) * a
                        -b * lRate / batch_size for
                        a, b in zip(self.weights, w_ch)]
    def train(self, inps, goals, lRate, batch_size, dropProb):
        """Given a training set/goals, trains the network across the set"""
        if len(inps) != len(goals):
            raise IndexError("Length of input and goal arrays must be equal")
        for i in range(0, len(inps), batch_size):
            self.run_batch(inps[i:i + batch_size],
                            goals[i:i + batch_size], lRate, batch_size, dropProb)

    def repTrain(self, inps, lbls, _lRate, batch_size, epochCt, dropProb, valInps, valLbls, output_file=None):
        """Train the network for the inps/lbls epoch_ct times."""
        if len(inps) != len(lbls):
            raise IndexError("Length of input and goal arrays must be equal")
        goals = []
        for i in range(0, len(inps)):
            """Turn the labels into arrays, with the correct index set to 1, 
            all others set to 0"""
            goals.append(np.zeros((self.sizes[-1])))
            goals[-1][lbls[i]] = 1
        if len(valInps) == len(valLbls):
                ctRt, ctWr, cost = self.eval(valInps, valLbls)
                print("Accuracy prior to training is", ctRt * 100
                      / (ctRt + ctWr), "% with a cost of", cost)
        for i in range(0, epochCt):
            """Mix up the inps/labels, as not doing so causes problems"""
            print("epoch ", i, "...")
            order = np.arange(1, len(inps))
            np.random.shuffle(order)
            _inps = []
            _goals = []
            for x in order:
                _inps.append(inps[x])
                _goals.append(goals[x])

            """Gradually decrease the learning rate"""
            lRate = _lRate / (1 + i) ** .03125  # Gradually decrease the learning rate

            """Train the network"""
            self.train(_inps, _goals, lRate, batch_size, dropProb)

            ctRt, ctWr, cost = self.eval(valInps, valLbls)
            print("Accuracy on Epoch", i, "is", ctRt * 100
                  / (ctRt + ctWr), "% with a cost of", cost) 
            if output_file:
                op = output_file + "_e" + str(i) + "_"  + str(ctRt * 100 \
                          // (ctRt + ctWr)) + "-" + str(i) + ".net"
                print("Storing to ", op)
                self.storeNet(op)  
        if output_file:
            self.storeNet(output_file)
    def choice(self, inp):
        """Return the network's decision for a given input"""
        self.feedUp(inp)
        return np.argmax(self.activs[-1])

    def evalCase(self, inp, lbl):
        """Given an input and label, return if the input matches the label"""
        m = self.choice(inp)
        if m == lbl:
            return True
        else:
            return False

    def eval(self, inps, lbls, printGsAs = False):
        """Return how many correct and incorrect classifications the network
        has for a given input array and corresponding label array"""
        ctRt = 0
        ctWr = 0
        cost = 0
        for inp, lbl in zip(inps, lbls):
            rt = self.evalCase(inp, lbl)
            goal = np.zeros(self.sizes[-1])
            goal[lbl] = 1
            cost += sum(self.cost(goal))
            if rt:
                ctRt += 1
            else:
                ctWr += 1
        if printGsAs:
            for i, L in zip(inps, lbls):
                self.feedUp(i)
                rt = self.evalCase(i, L)

                print("Label:", L)
                print("Input:", i)
                print("Activs:", self.activs[-1])
                if rt:
                    print("Evaluated Correctly")
                else:
                    print("Evaluated Incorrectly")
        return (ctRt, ctWr, cost)

    def storeNet(self, filePath):
        """Store this network to filePath on the disk drive"""
        file = open(filePath, "wb")
        pick.dump(self, file)
        file.close()

    @staticmethod
    def loadNet(filePath):
        """"Return the object (Hopefully a network :) stored at filePath"""
        file = open(filePath, "rb")
        obj = pick.load(file)
        file.close()
        return obj

"""End of network2.py"""

