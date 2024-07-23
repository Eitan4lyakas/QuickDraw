#this file is the "main" file of the project. this file is the file that needed to be activate to run the game
#this file contains the graphic functions and also calling the predicting functions

import json
import tkinter as tk

from tkinter import messagebox
from PIL import Image, ImageDraw
from modelTraining import ModelPredict

class QuickDrawGraphics:
    #this class contains all the grphics function- creating the window, drawing methods, buttons...
    def __init__(self):
        with open("QuickDrawData\\ObjectsList.json", "r") as file:
            self.objectList = json.load(file)

        self.WIDTH = 500
        self.HEIGHT = 500
        self.WHITE = (255, 255, 255)
        self.brushWidth = 8

        self.root = tk.Tk()
        self.root.title("Quick Draw")
        self.root.resizable(False, False) #makes the window unresizeble

        self.canvas = tk.Canvas(self.root, width = self.WIDTH - 10, height = self.HEIGHT - 10, bg = 'white')
        self.canvas.pack(expand = tk.YES, fill = tk.BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.currentDrawing = Image.new("RGB", (self.WIDTH, self.HEIGHT), "white")  # create a white image
        self.draw = ImageDraw.Draw(self.currentDrawing)  # create a draw object for the image
        
        self.color = "black"

        self.createButtons()
    
    def activatePaint(self):
        #this function activating the paint function (after pressing the pencil button)
        self.brushWidth = 8
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        #this function contains the drawing method- getting the indexes of the mouse, than patining the place with black pixel when clicking at the screen
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)

        self.canvas.create_rectangle(x1, y1, x2, y2, fill = 'black', width = self.brushWidth)
        self.draw.rectangle([x1, y1, x2 + self.brushWidth, y2 + self.brushWidth], fill = 'black', outline = 'black')

    def activateEraser(self):
        #this function activating the erase function (after pressing the eraser button)
        self.brushWidth = 30
        self.canvas.bind("<B1-Motion>", self.erase)

    def erase(self, event):
        #this function contains the erasing method- getting the indexes of the mouse, than patining the place with white pixel when clicking at the screen
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)

        self.canvas.create_rectangle([x1, y1, x2 + self.brushWidth, y2 + self.brushWidth], fill = 'white', outline = 'white')

    def createButtons(self):
        #this function creates the buttons on the main screen
        buttonFrame = tk.Frame(self.root)
        buttonFrame.pack(side = 'left')

        #pencil button- drawing
        pencilImage = tk.PhotoImage(file = "Art/pencil.png")
        pencilImage = pencilImage.subsample(5, 5)
        pencilButton = tk.Button(buttonFrame, image=pencilImage, command=self.activatePaint)
        pencilButton.image = pencilImage  
        pencilButton.pack(side = 'left') 

        #eraser button- erasing
        eraserImage = tk.PhotoImage(file = "Art/eraser.png")
        eraserImage = eraserImage.subsample(5, 5)
        eraserButton = tk.Button(self.root, image=eraserImage, command=self.activateEraser)
        eraserButton.image = eraserImage
        eraserButton.pack(side = 'left') 

        #predict button
        predictImage = tk.PhotoImage(file = "Art/brain.png")
        predictImage = predictImage.subsample(5, 5)
        predictButton = tk.Button(buttonFrame, image=predictImage, command=self.predict)
        predictButton.image = predictImage
        predictButton.pack(side = 'left')

        #clear button- clearing all the drawing on the screen
        clearImage = tk.PhotoImage(file = "Art/restart.png")
        clearImage = clearImage.subsample(5, 5)
        clearButton = tk.Button(buttonFrame, image=clearImage, command=self.clearCnvas)
        clearButton.image = clearImage
        clearButton.pack(side = 'left')

        #info button- navigating to other window with information about the game
        infoImage = tk.PhotoImage(file = "Art/questionMark.png")
        infoImage = infoImage.subsample(5, 5)
        infoButton = tk.Button(buttonFrame, image=infoImage, command=self.info)
        infoButton.image = infoImage
        infoButton.pack(side = 'left')

        #calling the closing function when pressing the X button at the top of the screen
        self.root.protocol("WM_DELETE_WINDOW", self.onClose)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    def clearCnvas(self):
        #this function clears all the drawing in the canvas when activated- deleting everything. than to make sure drawing a big white sqaure on top of the canvas
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.WIDTH, self.HEIGHT], fill = self.WHITE)
    
    def info(self):
        #this function creates the information window- that contanins the rules of the game, and more buttons
        infoWindow = tk.Toplevel(self.root)
        infoWindow.title("Info") 

        #the text that will be shown on the window
        helloText = "Hey There! wellcome to Quick Draw game!"
        interductionText = '''I am the smart AI that will try to guess what you draw. \n just draw something and click the brain button to activate the magic!'''

        buttonExplainText = '''click the gallery button to see the list of the objects i can recognize \n click the button button to get information about every button \n Now, click back and get to draw!'''

        endText = "Have fun!"

        #definening the font the the size of the texts
        helloLabel = tk.Label(infoWindow, text = helloText, font = ("fixedsys", 22, "bold"))
        helloLabel.pack()

        interductionLabel = tk.Label(infoWindow, text = interductionText, font = ("fixedsys", 18))
        interductionLabel.pack()

        buttonExplainLabel = tk.Label(infoWindow, text = buttonExplainText, font = ("Helvetica", 20))
        buttonExplainLabel.pack()

        endLabel = tk.Label(infoWindow, text = endText, font = ("Helvetica", 25, "bold"))
        endLabel.pack()

        #arrow button- closing the window and navigating back to the drawing window
        backImage = tk.PhotoImage(file = "Art/backArrow.png")
        backImage = backImage.subsample(5, 5)
        backButton = tk.Button(infoWindow, image = backImage, command = infoWindow.destroy)
        backButton.image = backImage
        backButton.pack(side = "left")

        #gallery button- navigating to another window, with a list of the object the model can recognize
        galleryImage = tk.PhotoImage(file = "Art/gallery.png")
        galleryImage = galleryImage.subsample(5, 5)
        galleryButton = tk.Button(infoWindow, image = galleryImage, command = lambda: self.showGallery(infoWindow))
        galleryButton.image = galleryImage
        galleryButton.pack(side = "left")

        #button button- navigating to another window, with an explenation of every button in the game
        buttonsImage = tk.PhotoImage(file = "Art/button.png")
        buttonsImage = buttonsImage.subsample(5, 5)
        buttonsButton = tk.Button(infoWindow, image = buttonsImage, command = lambda: self.ButtonsExplain(infoWindow))
        buttonsButton.image = buttonsImage
        buttonsButton.pack(side = "left")
    
    def showGallery(self, infoWindow):
        #this function creates the gallery window- a list with all the objects the model can recognize
        infoWindow.destroy() #closing the info window
        galleryWindow = tk.Toplevel(self.root)
        galleryWindow.title("Gallery")

        galleryText = '''Here is the list of the objects i can recognize:'''

        galleryLabel = tk.Label(galleryWindow, text = galleryText, font = ("fixedsys", 22, "bold"))
        galleryLabel.pack()

        for object in self.objectList: #getting the object from the json file that saves all the objects
            objLabel = tk.Label(galleryWindow, text = object, font = ("fixedsys", 18))
            objLabel.pack()

        #arrow button- closing the window and navigating back to the drawing window
        backImage = tk.PhotoImage(file = "Art/backArrow.png")
        backImage = backImage.subsample(5, 5)
        backButton = tk.Button(galleryWindow, image=backImage, command = galleryWindow.destroy)
        backButton.image = backImage
        backButton.pack()

    def ButtonsExplain(self, infoWindow):
        #this function creates the button explenation window- a list with all the button in the game and their role
        infoWindow.destroy() #closing the info window
        buttonsWindow = tk.Toplevel(self.root)
        buttonsWindow.title("Buttons Explainaion")
        
        titleText = "Buttons Explanation:"
        pencilText = '''-The pencil button is used to draw on the canvas. \n click on it to draw with black color'''
        eraserText = '''-The eraser button is used to erase the drawing on the canvas. \n click on it to erase with white color'''
        predictText = '''-The brain button is used to predict the object you drew. \n click on it to see the magic!'''
        restartText = '''-if you want to clear the canvas, click the restart button'''
        infoText = '''-if you want to get information about the buttons, click the question mark button'''

        title = tk.Label(buttonsWindow, text = titleText, font = ("Helvetica", 22, "bold"))
        title.pack()

        pencilLabel = tk.Label(buttonsWindow, text = pencilText, font = ("Helvetica", 18))
        pencilLabel.pack()

        eraserLabel = tk.Label(buttonsWindow, text = eraserText, font = ("Helvetica", 18))
        eraserLabel.pack()

        predictLabel = tk.Label(buttonsWindow, text = predictText, font = ("Helvetica", 18))
        predictLabel.pack()

        restartLabel = tk.Label(buttonsWindow, text = restartText, font = ("Helvetica", 18))
        restartLabel.pack()

        infoLabel = tk.Label(buttonsWindow, text = infoText, font = ("Helvetica", 18))
        infoLabel.pack()

        backImage = tk.PhotoImage(file = "Art/backArrow.png")
        backImage = backImage.subsample(5, 5)
        backButton = tk.Button(buttonsWindow, image=backImage, command = buttonsWindow.destroy)
        backButton.image = backImage
        backButton.pack()

    def predict(self):
        #this function activates the predicting method- saving the drawing as a png file. than calling the predict function from the "modelTraining.py" file with the path of the current drawing
        self.currentDrawing.save(f"QuickDrawData/ToPredict/drawing.png")

        modelPrediction = ModelPredict()
        predictions = modelPrediction.predictDrawing("QuickDrawData/ToPredict/drawing.png")

        if predictions is None:
            messagebox.showinfo("Error, Error", f"Pls draw something. i cant predict a white canvas :/") #wrtining an error message if there is nothing on the canvas
        else:
            textPrediction = ""
            for prediction, probability in predictions:
                textPrediction += f"{prediction} : {probability}% \n"

            messagebox.showinfo("Prediction", f"Top three predictions for the drawing: \n {textPrediction}")
        

    def onClose(self):
        self.root.destroy()
        exit()

def main():
    graphics = QuickDrawGraphics()
    graphics.root.mainloop()

if __name__ == "__main__":
    main()