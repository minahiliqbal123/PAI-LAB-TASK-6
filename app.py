from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load YOLO Model
try:
    net = cv2.dnn.readNet("C:\\Users\\MY PC\\Desktop\\Animal detection\\project\\yolov4.weights", 
                       "C:\\Users\\MY PC\\Desktop\\Animal detection\\project\\yolov4.cfg")

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  
except cv2.error as e:
    print("Error loading YOLO model:", e)
    exit(1)

# Load class labels
classes = []
coco_path = r"C:\Users\MY PC\Desktop\Animal detection\project\coco.names"
if os.path.exists(coco_path):
    with open(coco_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
else:
    print("Error: coco.names file not found.")
    exit(1)

layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten().tolist()]

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return "Invalid file type", 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    detected_file = detect_animals(filepath, filename)
    return render_template('result.html', filename=detected_file)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/get_locations', methods=['GET'])
def get_locations():

    locations = [
        {"lat": 37.7749, "lng": -122.4194, "animal": "Libra"},
        {"lat": 34.0522, "lng": -118.2437, "animal": "lion"}
    ]
    return jsonify(locations)

def detect_animals(image_path, filename):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image {image_path}")
        return None

    height, width, channels = image.shape

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)

    class_ids, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    cv2.imwrite(result_path, image)
    return filename

if __name__ == '__main__':
    app.run(debug=True, port=5001)

