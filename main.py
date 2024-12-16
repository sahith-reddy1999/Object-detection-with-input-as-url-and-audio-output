import cv2
import numpy as np
import pyttsx3
import requests
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import base64
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Initialize the speech engine for audio alerts
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speech rate
engine.setProperty('volume', 1)  # Set volume

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Path to YOLO weights and cfg
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO labels (for YOLO)
with open("coco.names", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Object detection function
def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Set confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indexes

# Function to encode frame as base64
def encode_frame(frame):
    _, encoded_img = cv2.imencode('.jpg', frame)
    return base64.b64encode(encoded_img).decode('utf-8')

# Function to speak out the detected objects
def speak_objects(object_details):
    for obj in object_details:
        engine.say(obj)
        engine.runAndWait()  # Wait until speech is finished before moving to next

# Route to process image URL
@app.post("/process_url")
async def process_url(url: str = Form(...)):
    # Fetch the image from URL
    try:
        response = requests.get(url)
        
        # Check if the response was successful (status code 200)
        if response.status_code != 200:
            return {"error": "Failed to retrieve image. Please check the URL."}

        # Convert the image to a numpy array
        img = np.array(bytearray(response.content), dtype=np.uint8)
        
        # Decode the image
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # Check if the image was decoded properly
        if frame is None:
            return {"error": "Failed to decode the image. Please check the image format."}

        # Perform object detection
        boxes, confidences, class_ids, indexes = detect_objects(frame)

        # Object details
        object_details = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(labels[class_ids[i]])
                confidence = confidences[i] * 100  # Convert confidence to percentage
                object_details.append(f"{label}: {confidence:.2f}%")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame to base64 to send back
        frame_base64 = encode_frame(frame)

        # Speak out detected objects
        speak_objects(object_details)

        # Return image as base64 and object details
        return {
            "image": frame_base64,
            "objects": object_details
        }
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Serving the HTML page
@app.get("/", response_class=HTMLResponse)
async def serve_html():
    with open("index.html", "r") as file:
        return HTMLResponse(content=file.read())
