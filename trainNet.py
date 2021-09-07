#! /usr/bin/env python3
"""trainNet.py"""

from netWrapper import mnist
from network2 import Network
import sys
import os

"""Load the train/test images from the mnist data set using the class in netWrapper.py"""
data = mnist()
print("Expanding Data...")
data.expandData()
print("Data expanded")
tr_ims = data.train_imgs
tr_lbs = data.train_lbls
print("There are ", len(tr_ims), " training images")

t_usedCt = 10000
te_ims = data.test_imgs[:t_usedCt]
te_lbs = data.test_lbls[:t_usedCt]

"""Define the network structure"""
sizes = [784, 100, 100, 10]

"""Define the network training parameters"""
k = 5
lRate = .15
b_size = 200
epoch_ct = 100
dropProb = .75
reg_param = 0

fout = "Net"
if len(sys.argv) > 1:
	fout = sys.argv[1]
print("Writing network to: ", fout)

"""Initialize the network"""
if os.path.exists(fout):
	print("Loading network from ", fout)
	n = Network.loadNet(fout)
elif os.path.exists(fout + ".net"):
	print("Loading network from ", fout + ".net")
	n = Network.loadNet(fout + ".net")
else:
	n = Network(sizes, k, reg_param)
"""Train the network based on the parameters above"""
n.repTrain(tr_ims, tr_lbs, lRate, b_size, epoch_ct, dropProb, te_ims, te_lbs, fout)

"""Print its accuracy on the test data at the end of training"""
ctRt, ctWr, cost = n.eval(te_ims, te_lbs, False)
print("ctRt:", ctRt, "ctWr:", ctWr, "cost:", cost)

"""Store the Network"""
n.storeNet(fout)

"""End of trainNet"""
