#importing libraries
import flask
import numpy as np
from flask import Flask, render_template, request
from flask import request
from ultralytics import YOLO
from PIL import Image


app=Flask(__name__)
model = YOLO("yolov8n-oiv7.pt")


@app.route('/')
def index():
    return flask.render_template("index.html")
    #return "Hello World"

#prediction function
@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        result_arr = []
        img = request.files["img"]
        img = Image.open(img)
        img.save("Image.jpg")
        img = Image.open("Image.jpg")
        results = model.predict(source="Image.jpg")
        for result in results:
            for box in result.boxes:
                result_arr.append(model.names[box.cls[0].item()])
        # if result_arr is None:
        #      return "No object detected"
        return result_arr
    return "this shouldn't have happened"


if __name__ == "__main__":
	app.run(debug=True)