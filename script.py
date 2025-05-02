#importing libraries
import flask
import numpy as np
from flask import Flask, render_template, request
from flask import request, jsonify
import requests
from ultralytics import YOLO
from PIL import Image
import json


app=Flask(__name__)
model = YOLO("yolov8n-oiv7.pt")

OPENROUTER_API_KEY = "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"

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
        ans = []
        for item in result_arr:
            response = requests.post(
					url="https://openrouter.ai/api/v1/chat/completions",
					headers = {
						"Authorization": f"Bearer {OPENROUTER_API_KEY}",
						"Content-Type": "application/json"
					},
					data=json.dumps({
						"model": "deepseek/deepseek-r1:free",
						"messages": [
						{
							"role": "system",
							"content": "You are an expert in recycling guidelines."
						},
						{
							"role": "user",
							"content": f"Provide material type, recyclable yes/no, instructions, environment impact for plastic in JSON format."
						}
						],
					})
					)
            ans.append(response.json())
        return jsonify({"result": ans})
		
        
        
if __name__ == "__main__":
	app.run(host='0.0.0.0')