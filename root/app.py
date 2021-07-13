import cv2
import numpy as np
from tensorflow import  keras
from flask import Flask,jsonify,request,json
from flask_cors import CORS
import base64
import re
import json
a_dict={'id1' : [0,1] }

fresh_emotions={}
faceCascade = cv2.CascadeClassifier('root/models/haarcascade_frontalface_alt.xml')
emotions=["Angry","confused","Fear","Happy","Sad","Surprise","Neutral","no_face"]
app = Flask(__name__)
CORS(app)





def add_key(id,value):
    global a_dict
    global fresh_emotions
    exist = False
    for key in a_dict:
        if(key == id):
            exist = True
    if(exist == True):
        a = a_dict.get(id)
        a.append(value)
    else:
        a_dict[id] = [value]

    fresh_emotions[id]=emotions[value]

@app.route('/getClear')
def get_clear():
    global a_dict
    a_dict={'id0':[6,6]}
    global fresh_emotions
    fresh_emotions={}

    return "json_object"

@app.route('/getFresh')
def get_fresh():
    json_object = json.dumps(fresh_emotions, indent=4)

    return json_object



@app.route('/getBar')
def get_satisfied():
    global a_dict
    unsatisfied = 0
    satisfied = 0
    for key, value in a_dict.items():
        for v in value:
            if(v in  (0 , 1 , 2 , 4)):
                unsatisfied = unsatisfied+1
            elif(v in ( 3 , 5 , 6)):
                satisfied = satisfied+1


    return jsonify({"satisfied": satisfied, "unsatisfied": unsatisfied})


@app.route('/getPie')
def pieStat():
    global a_dict
    paying_attention = 0
    not_paying_attention = 0
    for key, value in a_dict.items():
        for v in value:
            if (v == 7):
                not_paying_attention = not_paying_attention + 1
            else:
                paying_attention = paying_attention + 1


    return jsonify({"payingAttention": paying_attention, "notPayingAttention": not_paying_attention})


@app.route('/getDoughnut')
def all_emotions():
    global a_dict
    angry = 0
    confused = 0
    fear = 0
    happy = 0
    sad = 0
    surprise = 0
    neutral = 0

    for key, value in a_dict.items():
        for v in value:
            if(v == 0):
                angry = angry + 1
            if(v == 1):
                confused = confused + 1
            if(v == 2):
                fear = fear + 1
            if(v == 3):
                happy = happy + 1
            if(v == 4):
                sad = sad + 1
            if(v == 5):
                surprise = surprise + 1
            if(v == 6):
                neutral = neutral + 1
    print(angry)
    print(confused)
    print(fear)
    print(happy)
    print(sad)
    print(surprise)
    print(neutral)

    return jsonify({"angry": angry, "confused": confused , "fear": fear, "happy": happy , "sad" : sad, "surprise": surprise , "neutral": neutral})







@app.route('/')
def Index():
    return "Hello"

@app.route('/getEmotion', methods=['POST'])
def emo():
    global fresh_emotions
    req = request.get_data()
    js = json.loads(req)
    try:
        image_data = re.sub('data:image/.+;base64,', '', js['url'])
        stID=(js['id'])
        imgg = base64.b64decode(image_data)
        nparr = np.fromstring(imgg, np.uint8)
        image = cv2.imdecode(nparr, -1)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        res_num = start(img)
        res_emotion = emotions[res_num]
        add_key(stID,int(res_num))

        json_object = json.dumps(fresh_emotions, indent=4)

        return json_object
    except:
        return jsonify({"Error": "input not correct"})





ESC = 27


def find_faces(image):

    coordinates = locate_faces(image)
    cropped_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in coordinates]

    normalized_faces = [normalize_face(face) for face in cropped_faces]

    return zip(normalized_faces, coordinates)

def normalize_face(face):
    test_image = cv2.resize(face, (48, 48))
    test_image = np.array(test_image)
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    gray = gray / 255
    # reshaping image (-1 is used to automatically fit an integer at it's place to match dimension of original image)
    gray = gray.reshape(-1, 48, 48, 1)


    return gray;

def locate_faces(image):
    faces = faceCascade.detectMultiScale(np.asarray(image), 1.3, 5)

    return faces


def analyze_picture(model_emotion, path,image):
    result_num=7


    for normalized_face, (x, y, w, h) in find_faces(image):
        res = model_emotion.predict(normalized_face)
        # argmax returns index of max value
        result_num = np.argmax(res)

    return result_num

def start(image):
    # emotions = ["Angry","confused","Fear","Happy","Sad","Surprise","Neutral"]

    pretrained_model = keras.models.load_model("root/models/my_model3.h5")
    path=""

    res_num = analyze_picture(pretrained_model, path, image)
    return res_num

if __name__ == '__main__':
    app.run(debug=True)