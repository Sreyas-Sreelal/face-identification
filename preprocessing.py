import pathlib
import numpy
import cv2
import os

# configurations
DATASET_DIR = "dataset"
PROCESSED_DIR = "processed"

def process_dataset():
    path = pathlib.Path(DATASET_DIR)
    classes = []
    for directory in path.glob("*"):
        if not directory.is_dir():
            continue
        current_class = str(directory).split("\\")[1]
        idx = 0
        try:
            os.mkdir(PROCESSED_DIR + '\\' + current_class)
        except:
            print("[Warning] couldn't create directory, make sure directory is clean")
        classes.append(current_class)
        for image_files in directory.glob("*.jpg"):
            image = cv2.imread(str(image_files))
            gray_scaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(
                gray_scaled_image,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(30, 30)
            )
            if len(faces) > 0:
                for x,y,w,h in faces:
                    if h < 200 or w < 200:
                        continue
                    image = image[y:y+h,x:x+w]
                    try:
                        cv2.imwrite(PROCESSED_DIR + '\\' + current_class + '\\' + str(idx) + '.jpg',image)
                        idx+=1
                    except Exception as e:
                        print(str(e))
                        continue
        
if __name__ == "__main__":
    process_dataset()