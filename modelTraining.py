#In this file i activated all the training and predicting algorithms for my project

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, models
from keras.models import load_model
from keras.callbacks import EarlyStopping, LearningRateScheduler


from saveDrawingsFromQD import DrawingSaving

class convertToPixels():
     #this class contains all the methods that converts the images to array of pixels
    def __init__(self):        
        self.drawingArray = []#array of all the drawings converted to pixels
        self.numOfObject = []#array of the serial number of the object

        #open the json file that contains the list of objects
        with open("QuickDrawData\\ObjectsList.json", "r") as file:
            self.objects = json.load(file)

    def convertImageToPixels(self, imagePath):
        #in: path to the image
        #out: array of the pixels pixels of the image

        img = Image.open(imagePath)
        imgConv = img.convert('1').resize((49, 49), Image.LANCZOS) #- with resize
        #imgConv = img.convert('1') #- without resize
        pixels = np.array(imgConv, dtype = int).tolist()
        return pixels
    
    def convertAllImagesToPixels(self, numOfImages):
        #in: number of images
        #converts all the images to pixels and saves them in the drawingArray

        for i in range(numOfImages):
            if i % 1000 == 0:
                print(i)

            for object in self.objects:
                objectPixels = self.convertImageToPixels(f'QuickDrawData\\{object}\\{i}.png')
                self.drawingArray.append(objectPixels)
                self.numOfObject.append(self.objects[object])

        print(len(self.drawingArray))
        print(len(self.numOfObject))
        print('done')

    def saveInNpyFiles(self):
        #saves the drawingArray and the numOfObject in npy files

        self.drawingArray = np.array(self.drawingArray)
        np.save('C:\\Users\\Eitan\\Desktop\\QuickDrawBigFiles\\drawings.npy', self.drawingArray) #- with resize
        #np.save('afterTraining\\drawingsWithoutResize.npy', self.drawingArray) #- after resize

        self.numOfObject = np.array(self.numOfObject)
        np.save('C:\\Users\\Eitan\\Desktop\\QuickDrawBigFiles\\numOfObject.npy', self.numOfObject)



class modelTraining():
    #this class contanins all the methods for the model training
    def __init__(self):
        #i have tried the training functions twice-
        #first when i resize the image to speed up the training process
        #than, when i am not resizeing the images, to see if the results are better or not


        self.pixeledDrawings = np.load('C:\\Users\\Eitan\\Desktop\\QuickDrawBigFiles\\drawings.npy') #-with resize
        #self.pixeledDrawings = np.load('afterTraining\\drawingsWithoutResize.npy') #- without resize

        self.serialNumbers = np.load('C:\\Users\\Eitan\\Desktop\\QuickDrawBigFiles\\numOfObject.npy')

        with open("QuickDrawData\\ObjectsList.json", "r") as file:
            self.objects = json.load(file)

        self.model = self.createModel()

    def createModel(self):
        #this function create the model with all its layers and hyperparametes
        
        '''model = models.Sequential([
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(49, 49, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Dropout(0.3),  
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),  
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),  
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5), 
            layers.Dense(len(self.objects), activation='softmax')
        ])'''

        model = models.Sequential([
            layers.Conv2D(32, kernel_size=(5, 5), activation= 'relu', input_shape=(49, 49, 1), padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Dropout(0.4),

            layers.Conv2D(64, kernel_size=(5, 5), activation= 'relu', input_shape=(49, 49, 1), padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Dropout(0.4),

            layers.Conv2D(128, kernel_size=(5, 5), activation= 'relu', input_shape=(49, 49, 1), padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Dropout(0.4),

            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5), 
            layers.Dense(len(self.objects), activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model
    
    def lrScheduler(self, epoch):
        #this function creates the Learning Rate Schedule
        return 0.001 * np.exp(-epoch / 20)

    def trainModel(self):
        #this function is activating the training function:
        #first, it spilts the data and take 25% of the data for test
        #than, starting to train the model and printing the result, saving the model, and show a graph of the results

        self.serialNumbers = keras.utils.to_categorical(self.serialNumbers)

        xTrain, xTest, yTrain, yTest = train_test_split(self.pixeledDrawings, self.serialNumbers, test_size=0.25, random_state=0)

        scheduler = LearningRateScheduler(self.lrScheduler)
        earlyStopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = self.model.fit(
            x=xTrain,
            y=yTrain,
            epochs=100,
            batch_size=64,
            shuffle=True,
            validation_data=(xTest, yTest),
            callbacks=[scheduler, earlyStopping]
        )

        score = self.model.evaluate(xTest, yTest, verbose=0)
        scoreTrain = self.model.evaluate(xTrain, yTrain, verbose=0)

        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        self.model.save('afterTraining\\TrainedModelNumTwo.keras') #-with resize
        #self.model.save('afterTraining\\TrainedModelWithoutResize.keras') #-without resize

        acc = history.history['accuracy']
        valAcc = history.history['val_accuracy']
        loss = history.history['loss']
        valLoss = history.history['val_loss']
                
        print("accuracy - train", scoreTrain[1])

        epochs = range(1, len(acc) + 1)

        plt.figure(figsize = (10, 5))

        #acuurcy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b', label='Training accuracy')
        plt.plot(epochs, valAcc, 'r', label='Validation accuracy')
        plt.title('Model accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        #loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, valLoss, 'r', label='Validation loss')
        plt.title('Model loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

        
class ModelPredict():
    def __init__(self):
        self.trainedModel = load_model('afterTraining\\TrainedModel.keras')
        self.convertor = convertToPixels()

        with open("QuickDrawData\\ObjectsList.json", "r") as file:
            self.objects = json.load(file)

    def predictDrawing(self, drawingPath):
        print('Predicting...')
        drawingPixels = self.convertor.convertImageToPixels(drawingPath)

        if np.unique(drawingPixels).size == 1:
            return None

        drawingPixels = np.array(drawingPixels).reshape(1, 49, 49, 1)
        predictions = self.trainedModel.predict(drawingPixels)[0]

        top_three_indices = np.argsort(predictions)[-3:][::-1]  # Get indices of top three predictions
        
        top_three_predictions = []
        for index in top_three_indices:
            object_name = list(self.objects.keys())[index]
            probability = predictions[index]
            top_three_predictions.append((object_name, probability))

        return top_three_predictions
    
    
def main():
    #saver = DrawingSaving()
    #saver.saveDrawings()

    #convertor = convertToPixels()
    #convertor.convertAllImagesToPixels(50000)
    #convertor.saveInNpyFiles()

    trainer = modelTraining()
    trainer.trainModel()
    
    #predictor = ModelPredict()
    #prediction = predictor.predictDrawing('QuickDrawData\\cat\\929.png')
    #print(f"The drawing is a {prediction}")

if __name__ == '__main__':
    main()


#I saved the result after each training here, as a comment

#1500 images of dogs and cats, total- 3000 images
#0.99 acuracy, 0.8 test

#10000 images of dogs and cats, total- 20000 images
#1.0 acuracy, 0.83 test

#10000 images of 6 objects, total- 60000 images
#0.99 acuracy, 0.82 test

#10000 images of 6 objects, total- 60000 images
#0.99 acuracy, 0.85 test

#10000 images of 6 objects, total- 60000 images
#0.91 acuracy, 0.88 test
    
#10000 images of 6 objects, total- 60000 images. removed the resize (much bigger pixels arrays)
#didnt work, takes too much time

#50000 images of 6 objects, total- 300000 images
#0.91 acuracy, 0.91 test 