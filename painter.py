#! /bin/python3
"""paintor.py"""

from netWrapper import wrapper, mnist
import numpy as np
import tkinter as tk
import sys
import os

cur_choice_int = 10

data = mnist()
# ==this file's, paintor.py's, code is (very) loosely sourced from
# https://stackoverflow.com/questions/40604233/draw-on-python-tkinter-canvas-using-mouse-and-obtain-points-to-a-list
class Drawing(tk.Tk):
    """The class that will handle the drawing operations."""
    def __init__(self, canvShape = (400, 400), arrShape = (100, 100),
                 arr = None, outline = False, clrType = "L",
                 title = "Digit Recognizer", auto_normalize = True):
        """Initialize the drawing object. Set variables. ya' know.. regular stuff"""
	
        self.cur_choice = 11
        """Initialize Variables"""
        self.arr = arr
        self._arr = arr  # Store the original array...
        self.arrH, self.arrW = arrShape
        self.width, self.height = canvShape
        if type(self.arr) == type(None):
            self.arr = np.zeros(canvShape).astype(int)
            self._arr = self.arr
        self.x = self.y = 0
        self.prevarrX = self.prevarrY = 0
        self.initNets()
        self.isShift = False
        self.clrType = clrType
        self.auto_normalize = auto_normalize
        """Start with the Tkinter stuff"""
        tk.Tk.__init__(self)

        self.title(title)  # Set the title

        """Draw the canvas, whose job is to be drawn on"""
        self.canvas = tk.Canvas(self, width = self.width + 1,
                                height = self.height + 1,
                                bg = "black", cursor = "cross")
        self.canvas.clrType = "L"
        self.canvas.outline = True
        self.canvas.hSpl = self.width // self.arrW
        self.canvas.wSpl = self.height // self.arrH


        """Set up the buttons, clear and eval. """

        """Define button objects and their text"""
        self.button_eval_txt = tk.StringVar()
        self.button_norm_toggle_text = tk.StringVar()
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear)
        self.button_norm_toggle_text.set("Press to disable auto-normalization" if
                                         self.auto_normalize else
                                         "Press to enable auto-normalization")
        self.button_eval_txt.set("Press to evauluate")
        self.button_eval = tk.Button(self, textvariable =
                                     self.button_eval_txt, command = self.eval)
        self.button_norm_toggle = tk.Button(self, textvariable =
                                            self.button_norm_toggle_text,
                                            command = self.norm_toggle)
        self.button_norm_manual = tk.Button(self, text = "Press to normalize image",
                                            command = self.manual_normalize)
        """Place Buttons  and the drawing canvas on the screen"""
        self.canvas.grid(column = 0, row = 0, columnspan = 1, rowspan = 5)  # Place buttons on grid
        self.button_clear.grid(column = 0, row = 5)
        self.button_eval.grid(column = 0, row = 6)
        self.button_norm_toggle.grid(column = 0, row = 7)
        if ~self.auto_normalize:
            self.button_norm_manual.grid(column = 0, row = 8)



        """Set binding for mouse-events"""
        self.canvas.bind("<Motion>", self.curr_loc)
        self.canvas.bind("<B1-Motion>", self.dragged_ev)
        self.canvas.bind("<Button-1>", self.clicked_ev)
        self.canvas.bind("<B3-Motion>", self.r_dragged_ev)

        """Draw the initial image-array on the drawing canvas. """
        self.refresh(self.canvas, self.arr)
    def manual_normalize(self):
        """Normalizes the drawing on the canvas, updating the array"""
        self.arr = self.netwrapper.normalizeInp(self.arr)
        self.refresh(self.canvas, self.arr)
    def norm_toggle(self):
        """
        Toggles whether images are automatically normalized on the drawing canvas
        when the user evaluates an image
        """
        if self.auto_normalize:
            self.auto_normalize = False
            self.button_norm_toggle_text.set("Press to enable auto-normalization")
            self.button_norm_manual.grid(column = 0, row = 8)
        else:
            self.auto_normalize = True
            self.button_norm_toggle_text.set("Press to disable auto-normalization")
            self.button_norm_manual.grid_forget()
    def refresh(self, canvas, arr):
        """Redraw self.arr on the drawing canvas."""
        arrH, arrW = arr.shape
        clrFunc = self.GSVtoStr if canvas.clrType == "L" else self.RGPosNeg
        for y in range(0, arrH):
            for x in range(0, arrW):
                canvas.create_rectangle(
                    x * canvas.wSpl, y * canvas.hSpl, (x + 1) *
                    canvas.wSpl, (y + 1) * canvas.hSpl,
                    fill = clrFunc(arr[y][x]),
                    outline = ("black" if canvas.outline else
                                            clrFunc(arr[y][x])))
    def GSVtoStr(self, GSV):
        """
        Convert a grayscale int from 0-255 to a
        String of its hex-value RGB form , with a '#' at the front
        Used in any Tkinter color-calls
        """
        par = hex(GSV)[2:]  # Get the number part
        if par[0] == "x" :
            par = par[1:]
        if len(par) == 1:
            par = "0" + par  # Add a 0 if the number is only 1-digit in hex
        return "#" + par * 3  # Take "#", then add par 3 times, since its grayscale

    def clear(self):
        """Clear the canvas"""
        self.arr = self._arr
        self.canvas.delete("all")
        self.button_eval_txt.set("Image Cleared. Press to re-evaluate")
        self.arr = np.zeros((self.arrH, self.arrW)).astype(int)
        data = mnist()
        #self.arr = data._train_imgs[self.cur_choice]
        self.cur_choice += 1
        self.refresh(self.canvas, self.arr)

    def curr_loc(self, event):
        """Store the current x,y coords of the cursor."""
        self.x = event.x
        self.y = event.y

    def drawToArr(self, x, y, z = 12):
        """
        Given an x,y coordinate, draw the corresponding
        brush-stroke. I have it set to a 2x2 box, with the top-left the brightest
        and the bottom-left the darkest.
        """
        PtsDraw = []  # Points to be drawn to as tuples in the form (X,Y,Value)
        if x >= 0 and y >= 0 and x <= self.arrW - 1 and y <= self.arrH - 1:
            # Only run if you're acutally in bounds to draw

            # The mins/maxes are so you dont go out of bounds with the excess
            startX = max(0, x)
            endX = min(x + 2, self.arrW)
            startY = max(0, y)
            endY = min(y + 2, self.arrH)
            for x, xV in zip(range(startX, endX), [2 * z, z]):
                for y, yV in zip(range(startY, endY), [2 * z, z]):
                    """Adjust self.arr for the x,y coords, maxing out at 255, and bottoming out at 0"""
                    self.arr[y][x] = max(min(self.arr[y][x] + xV + yV, 255), 0)
                    PtsDraw.append((x, y, self.arr[y][x]))

        for x, y, val in PtsDraw:
            """
            Redraw all pixels that were drawn over. CONSIDERABLY more
            efficient than refreshing the whole image every time. BY FAR more
            efficient
            """
            if self.canvas.clrType == "L":
                fi = self.GSVtoStr(val)
            else:
                fi = self.RGPosNeg(val)
            self.canvas.create_rectangle(
                x * self.canvas.wSpl, y * self.canvas.hSpl,
                (x + 1) * self.canvas.wSpl, (y + 1) * self.canvas.hSpl,
                fill = fi,
                outline = ("black" if self.canvas.outline else
                                         fi))

    def clicked_ev(self, event):
        """When the user triggers the mouse-click event, draw where they clicked"""
        self.draw_at_pos(event, 255)

    def dragged_ev(self, event):
        """When the user is dragging the mouse, draw as they drag"""
        self.draw_at_pos(event, 128)

    def r_clicked_ev(self, event):
        """When the user right-clicks, erase where they clicked"""
        self.draw_at_pos(event, -128)

    def r_dragged_ev(self, event):
        """When the user right-click drags the mouse, erase as they drag"""
        self.draw_at_pos(event, -255 )

    def draw_at_pos(self, event, z):
        """Draw to the canvas where the user clicks"""

        """Get the x,y coords of the mouse"""
        self.x = event.x
        self.y = event.y

        """Get the corresponding coord on the numpy pixel array for where the
        window-based x,y coordinate was drawn"""
        arrX = self.x // self.canvas.wSpl
        arrY = self.y // self.canvas.hSpl

        """Draw the new pixel"""
        self.drawToArr(arrX, arrY, z)

    def initNets(self):
        """Initialize the wrapper class, and the net files we'll use"""
        """"I trained several networks, to diminish fluctuations in results
        This loads all of them from their filepath strings """
        paths = None
        if len(sys.argv) > 1:
        	paths = sys.argv[1:]
        else:
        	paths = [os.path.join('nets', p) for p in os.listdir('nets')]
        self.netwrapper = wrapper(paths)

    def eval(self):
        """
        Return the networks' overall evaluation of our image, after normalizing the image
        If self.auto_normalize is True, update the canvas, otherwise only normalize internally
        """
        if np.all(self.arr == np.zeros(self.arr.shape)):  # If the image is blank, inform at the user
            self.button_eval_txt.set("Image is blank. Draw something!")
            return None  # Exit
        arr_normed = self.netwrapper.normalizeInp(self.arr)
        outp, conf = self.netwrapper.evalNets(self.arr)
        self.button_eval_txt.set("The network guesses a " + str(outp))
        if self.auto_normalize:
            """If we're auto-normalizing, refresh the screen with the normalized image"""
            self.arr = arr_normed
            self.refresh(self.canvas, self.arr)


if __name__ == "__main__":
    """Run the program"""

    """Window Size Setting... Set so that the window is large 
    enough so that things are overly squished, but small enough
    so that the pixelation of a 28x28 image isn't too bad"""
    canvShape = (28 * 10, 28 * 10)
    cur_choice_int = 10
    data = mnist()
    arr = np.zeros((28, 28)).astype(int)
    """Initialize Drawing"""
    dr = Drawing(canvShape, arr.shape,
                 arr, outline = False,
                 auto_normalize = False)

    """Begin the Tkinter loop, so that we can acutally do stuff..."""
    dr.mainloop()

"""End of paintor.py"""
