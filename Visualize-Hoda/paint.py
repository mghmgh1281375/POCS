from tkinter import *
from tkinter.colorchooser import askcolor
from ModelCNN import ModelCNN


class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.color_button = Button(self.root, text='color', command=self.choose_color)
        self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='eraser', command=self.delete_all) # use_eraser
        self.eraser_button.grid(row=0, column=3)

        self.choose_size_button = Scale(self.root, from_=30, to=40, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.predict = Button(self.root, text='predict', command=self.predict)
        self.predict.grid(row=0, column=5)

        self.width, self.height = 600, 600

        self.c = Canvas(self.root, bg='white', width=self.width, height=self.height)
        self.c.grid(row=1, columnspan=5)

        # Loading Model

        self.model = ModelCNN()
        self.model.load_weights('/home/mohammad/Projects/POCS/Visualize-Hoda/models/model_conv_20181024081423.h5')


        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def delete_all(self):
        self.c.delete("all")

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None
    
    def predict(self):
        self.c.postscript(file="tmp.eps")
        from PIL import Image
        import numpy as np
        import cv2
        img = Image.open("tmp.eps")
        img_arr = np.array(img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        img_arr = 255 - img_arr
        img_arr = cv2.resize(img_arr, (20, 20))
        img_arr = np.reshape(img_arr, (20, 20, 1))

        classes_, probas_ = self.model.predict_single(img_arr)
        print(classes_, probas_)


if __name__ == '__main__':
    Paint()
