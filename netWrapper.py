"""netWrapper.py"""

from PIL import Image, ImageFilter
#import mnist as mn
from network2 import Network
import numpy as np
import mnist as mn


class mnist:
    """Class which is used to easily handle usage of MNIST data"""
    def __init__(self):
        self._train_imgs = mn.train_images()
        self.train_imgs = [x.ravel()for x in self._train_imgs]
        self.train_lbls = mn.train_labels()
        self._test_imgs = mn.test_images()
        self.test_imgs = [x.ravel()for x in self._test_imgs]
        self.test_lbls = mn.test_labels()
    def expandData(self):
        new_tr_ims = self.train_imgs  # Start with the original training data
        new_tr_lbs = self.train_lbls.tolist()
        for img, lbl in zip(self._train_imgs, self.train_lbls):
            img = Image.fromarray(img, "L")
            p_imgs = []
            #p_imgs.append(img.filter(ImageFilter.EDGE_ENHANCE))
            #p_imgs.append(img.rotate(0, translate=(1, 1)))
            #p_imgs.append(img.rotate(0, translate=(-1, -1)))
            p_imgs.append(img.rotate(15))
            p_imgs.append(img.rotate(-15))
            for pim in p_imgs:
                new_tr_ims.append(np.array(pim))
                new_tr_lbs.append(lbl)
        self._train_imgs = new_tr_ims
        self.train_lbls = np.array(new_tr_lbs)
        self.train_imgs = [x.ravel() for x in self._train_imgs]



class wrapper:
    """Wrapper class for working between image drawing and the network.
    Handles normalizing input images, and utilizing from the network(s)"""
    def __init__(self, paths):
        """Set up the wrapper, using the filepath's of the network objects"""
        self.allnets = []
        for pth in paths:
            self.allnets.append(Network.loadNet(pth))
            if self.allnets[-1] == None:
                self.allnets.pop()
    def evalNets(self, inp ,):
        """
        Take an input, and run it through the given networks.
        Return the best voted image.
        """
        choices = np.zeros(10)
        inp = inp.ravel() / 255  # reformat the input to feed into the network
        for net in self.allnets:
            choices[net.choice(inp)] += 1
        return (np.argmax(choices), (choices[np.argmax(choices)] * 100) // np.sum(choices))
    def normalizeInp(self, inp):
        """
        Centers the image by pixel mass
        Note that it is mass-centered, rather than the bounding box
        it fills being centered on a canvas of size equal to the original
        input.
        """
        inH, inW = inp.shape
        inp = inp[~np.all(inp == 0, axis = 1)]  # Cut off all rows/columns that consist only of zeros
        inp = inp[:, ~np.all(inp == 0, axis = 0)]
        curH, curW = inp.shape
        """Going into this if statement, note that the input is now in its cropped form; having no columns or rows consisting of only zeros"""
        if inp.shape != (0, 0):  # If the shape is (0,0), its only blank
            _vect = np.arange(0, curW * curH)
            _vect.resize(curH, curW)  # Will be used to set up the mass-based centering
            vect = _vect % curW  # Each row in the matrix now contains the values 0 through curW; [0,curW)
            wes = inp / 255  # Get the weight values based on inp; Ensure all weights are in the interval [0,1]
            cenx = int(np.average(np.average(vect, axis = 1, weights = wes),
                                  weights = np.sum(inp, axis = 1)))  # This is the fun bit of the code.
    # Utilizing vect, which is a matrix consisting of a bunch of values in the interval [0,curW), take their weighted averages of vect with respect to inp.
    # However, only do so by rows, producing a matrix that consists of each row's weighted average. Now, take that matrix and compute its weighted average
    # with respect to the corresponding row's sum. In other words, the rows with a greater amount of total 'pixel mass' will be weighted heavier in this
    # stage of the average. This returns a final value, which corresponds to the whole input's center-by-mass x coordinate.

            # Transpose inp and _vect, and then perform virtually identical operations to compute the input's center-by-mass y coordinate
            _vect.transpose()
            inp.transpose()
            vect = _vect % curH
            wes = inp / 255
            ceny = int(np.average(np.average(vect, axis = 1, weights = wes),
                                   weights = np.sum(inp, axis = 1)))
            inp.transpose()  # Return the cropped inp matrix to being right-side-up
        else:
            return(np.zeros((inH, inW)).astype(np.int))  # There's nothing to bother moving around if its all black..., return a black image
        # Use PIL's Image tools to return the image to its original size, using the now-computed center-by-mass to center the image by.
        img = Image.fromarray(inp,  "L")
        img = img.crop((-inW // 2 + cenx, -inH // 2 + ceny,
                        inW // 2 + cenx, inH // 2 + ceny))  # Even though the function is called crop, we're using it to actually expand the image.

        out = np.array(img).astype(int)  # Return the image to a numpy array, and ensure its of type int
        return out
"""End of netWrapper.py"""
