#this file contains the funcion that saves all the drawing from the QuickDrawData library
#it is taking all the objects that the user chose, and then saves 10000 images from every object in the data
from quickdraw import QuickDrawDataGroup


class DrawingSaving:
    def __init__(self):
        self.objectsToBeSaved = [
            "car", "camera", "book", "dog", "cat"
        ]


    def saveDrawings(self, numOfDrawings):
        #in: the number of drawings the user want from each object
        #saving all the images based on the list of objects


        for object in self.objectsToBeSaved:
            objectCounter = 0
            objectGroup = QuickDrawDataGroup(object, max_drawings = numOfDrawings)


            for animalDrawing in objectGroup.drawings: #for every drawing
                animalDrawing.image.save(f"QuickDrawData\{object}\{objectCounter}.png")
                objectCounter += 1


def main():
    savingProgram = DrawingSaving()
    savingProgram.saveDrawings(50000)

if __name__ == "__main__":
    main()
