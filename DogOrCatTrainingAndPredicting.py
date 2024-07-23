#In this file, i tested the algorithm for training with array of pixels 
#I tried this with only two objects- dog drawings and cat drawings

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, models
from keras.models import load_model

from saveDrawingsFromQD import DrawingSaving

class convertToPixels():
    #this class contains all the methods that converts the images to array of pixels
    def __init__(self):
        self.dogsPath = 'QuickDrawData\\dog\\' 
        self.catsPath = 'QuickDrawData\\cat\\'
        
        self.drawingArray = [] #array that saves the pixels arrays of the drawings
        self.numOfObject = [] #array that saves the serial number of every drawing 1 for cat drawings, and 0 for dog drawings

        with open("QuickDrawData\\ObjectsList.json", "r") as file:
            self.objects = json.load(file)


    def convertImageToPixels(self, imagePath):
        #in: path for drawing
        #out: array with the pixels of the drawing: 1- white, 0- black

        img = Image.open(imagePath)
        imgConv = img.convert('1').resize((49, 49), Image.LANCZOS)
        pixels = np.array(imgConv, dtype = int).tolist()
        return pixels

    #1 is cat, 0 is dog
    def convertAllImagesToPixels(self, numOfImages):
        #this function does the same as the previous finction, but for a specific amount of drawings
        #than, it saves it in the arrays
         
        for i in range(numOfImages):
            for object in self.objects:
                objectPixels = self.convertImageToPixels(f'QuickDrawData\\{object}\\{i}.png')
                self.drawingArray.append(objectPixels)
                self.dogsOrCats.append(self.objects.index(object))


            dogPixels = self.convertImageToPixels(self.dogsPath + str(i) + '.png')
            catPixels = self.convertImageToPixels(self.catsPath + str(i) + '.png')
            self.drawingArray.append(dogPixels)
            self.drawingArray.append(catPixels)
            self.dogsOrCats.append(0)
            self.dogsOrCats.append(1)
  
        print(len(self.drawingArray))
        print(len(self.dogsOrCats))
        print('done')

    def saveInNpyFiles(self):
        #this function is saving the arrays as npy files for the training methods
        self.drawingArray = np.array(self.drawingArray)
        np.save('firstTestPrediction\\drawings.npy', self.drawingArray)

        self.dogsOrCats = np.array(self.dogsOrCats)
        np.save('firstTestPrediction\\dogOrCat.npy', self.dogsOrCats)

class modelTraining():
    #this class contanins all the methods for the model training
    def __init__(self):
        #x is the drawing converted to pixels
        #y says if it dog or cat, 1- cat, 0- dog

        self.x = np.load('firstTestPrediction\\drawings.npy')
        self.y = np.load('firstTestPrediction\\dogOrCat.npy')

        self.model = self.createModel()

    def createModel(self):
        #this function create the model with all its layers and hyperparametes
        model = models.Sequential([
            layers.Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(49, 49, 1), padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(6, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def trainModel(self):
        #this function is activating the training function:
        #first, it spilts the data and take 25% of the data for test
        #than, starting to train the model and printing the result, saving the model, and show a graph of the results

        self.y = keras.utils.to_categorical(self.y)

        xTrain, xTest, yTrain, yTest = train_test_split(self.x, self.y, test_size=0.25, random_state=0)

        # Check the shape of yTrain and yTest
        print("Shape of yTrain:", yTrain.shape)
        print("Shape of yTest:", yTest.shape)

        history = self.model.fit(
            x=xTrain,
            y=yTrain,
            epochs=100,
            shuffle=True
        )

        score = self.model.evaluate(xTest, yTest, verbose=0)
        scoreTrain = self.model.evaluate(xTrain, yTrain, verbose=0)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        self.model.save("firstTestPrediction\\dogOrCatModel.keras")

        acc=history.history['accuracy']
        print("accuracy - train",scoreTrain[1])

        loss=history.history['loss']


        epochs=range(1,len(acc)+1)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.plot(epochs, acc, 'bo', label='Training acc')

        plt.title('Training  accuracy')
        plt.legend()

class ModelPredict():
    #this class contains all the methods for predicting
    def __init__(self):
        self.trainedModel = load_model("firstTestPrediction\\dogOrCatModel.keras")
        print("model loaded:")
        self.convertor = convertToPixels()

    def predictDrawing(self, drawingPath):
        #this function gets a path to a drawing, converting it to array of pixels
        #then, outputs the serial number of the drawing
        
        print('Predicting...')
        drawingPixels = self.convertor.convertImageToPixels(drawingPath)
        drawingPixels = np.array(drawingPixels).reshape(1, 49, 49, 1)
        prediction = self.trainedModel.predict(drawingPixels)
        
        return np.argmax(prediction)

def main():
    saver = DrawingSaving()
    saver.saveDrawings()

    convertor = convertToPixels()
    convertor.convertAllImagesToPixels(10000)
    convertor.saveInNpyFiles()

    trainer = modelTraining()
    trainer.trainModel()
    
    #predictor = ModelPredict()
    #prediction = predictor.predictDrawing('QuickDrawData\\cat\\929.png')
    #print(f"The drawing is a {prediction}")

if __name__ == '__main__':  
    main()

#1500 images of dogs and cats, total- 3000 images
#0.99 acuracy, 0.8 test