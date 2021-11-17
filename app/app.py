import os, cv2, yaml
# import sys
# sys.path.insert(0, os.getcwd())
from logging import debug
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from datetime import datetime, date


with open(r"C:/Users/ASUS/ProjectDA/ECG_Classification/app/config.yaml", encoding="utf8") as file_:
    configs_ = yaml.load(file_, Loader=yaml.FullLoader)
config = configs_["app"]

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

def load_model():
    global model
    MODEL_PATH = config["model"]
    model = tf.keras.models.load_model(MODEL_PATH)
    print("* Model loaded!")

def model_predict(image_path, model):
    image_ = cv2.imread(image_path)
    image_ = cv2.resize(image_, (128,128))
    image_ = tf.keras.preprocessing.image.array_to_img(image_)
    image_ = np.expand_dims(image_, axis=0)
    image_ = np.vstack([image_])

    pre_test = model.predict(image_)
    
    return pre_test

print("* Loading Keras model!")
load_model()

@app.route("/", methods = ["GET"])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        star_time = datetime.now().time()
        pre_test = model_predict(file_path, model)
        end_time = datetime.now().time()
        time_a =  datetime.combine(date.min, end_time) - datetime.combine(date.min, star_time)
        time_predict = str(time_a)
        class_cu = ["F","N","Q","S","V"]
        pre_test_cau = class_cu[np.argmax(pre_test[0])]
        score = 100 * np.max(pre_test[0])
        
        
        #defin file json
        mang = {}
        mang["filename"] = os.path.basename(file_path)
        mang["time_predict"] = time_predict[6:None] + "s"
        # print(time_predict)
        mang["score"] = "%.2f" % score + "%"
        mang["class"] = pre_test_cau
        status = ""
        sick = ""
        if pre_test_cau == "F":
            status = config["F"]
            sick = config["F_benh"]
        if pre_test_cau == "N":
            status = config["N"]
            sick = config["N_benh"]
        if pre_test_cau == "Q":
            status = config["Q"]
            sick = config["Q_benh"]
        if pre_test_cau == "S":
            status = config["S"]
            sick = config["S_benh"]
        if pre_test_cau == "V":
            status = config["V"]
            sick = config["V_benh"]
        mang["status"] = status
        mang["sick"] = sick
        each_class = []
        for i in range(len(pre_test[0])):
            item = {}
            for j in range(len(class_cu)):
                if i == j:
                    score2 = pre_test[0][i] * 100 
                    item[class_cu[j]] = "%.2f" % score2 + "%"
                    each_class.append(item)
        mang["result"] = each_class
        # print(mang)
        result = jsonify(mang)
        return result

    return None


if __name__ == "__main__":
    # serve(app, host="0.0.0.0", port=8080)
    app.run(debug=True, port=8080, host="localhost")