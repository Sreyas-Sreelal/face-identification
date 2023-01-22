from preprocessing import DATASET_DIR,PROCESSED_DIR
from skimage.io import imread
from skimage.transform import resize
from sklearn import svm
import pandas
import pathlib
import numpy
import cv2
import os
import pickle

def train():
    path = pathlib.Path(PROCESSED_DIR)
    classes = {}
    X = []
    Y = []
    count =0
    for directory in path.glob("*"):
        if not directory.is_dir():
            continue
        count +=1
        classes[count]=str(directory).split("\\")[1]

        for image_file in directory.glob("*.jpg"):
            image = imread(str(image_file))
            resized = resize(image, (150,150,3))
            X.append(resized.flatten())
            Y.append(count)
    df=pandas.DataFrame(numpy.array(X))
    df['target'] = numpy.array(Y)
    X=df.iloc[:,:-1]
    Y=df.iloc[:,-1]
    svc=svm.SVC(probability=True)
    svc.fit(X,Y)
    model = {"model":svc,"classes":classes}
    pickle.dump(model, open('model', 'wb'))
    return model

try:
    model = pickle.load(open('model', 'rb'))
except:
    print("No models found. Retraining from preprocessed data...")
    model = train()

""" image = cv2.imread('test2.jpg')
gray_scaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
    gray_scaled_image,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)
x,y,w,h = faces[0]
image = image[y:y+h,x:x+w]
test =[]
resized = resize(image, (150,150,3))
test.append(resized.flatten())
print(model["classes"])
print(model["model"].predict(test)) """

vid = cv2.VideoCapture(0)
while True:
    ret,image = vid.read()
    cv2.imshow('detection',image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    if  len(faces) ==0 :
        continue
    for x,y,w,h in faces:
    
        resized = image
        resized = resize(resized, (150,150,3))
        predict = model["model"].predict_proba([resized.flatten()])
        text = ' '.join([ "{} - {:.2F}".format(model["classes"][idx+1][0],accuracy) for idx,accuracy in enumerate(predict[0])])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imshow('detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    

vid.release()
cv2.destroyAllWindows()