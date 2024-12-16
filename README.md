# Offline Object Detection with Audio Alerts

This project uses YOLOv3 for real-time object detection on images fetched from URLs. The detected objects and their confidence scores are displayed on the image, and the system speaks the detected objects aloud using Python's `pyttsx3` library.

## Features:
- paste a URL of an image to detect objects.
- Object detection on the image using the YOLOv3 model.
- Audio alerts for detected objects.
- The detected objects with their confidence scores are shown below the image.

## Installation

To get started with this project, you'll need Python 3.8+.

### Step 1: Clone the repository

```bash
git clone https://github.com/sahith-reddy1999/Object-detection-with-input-as-url-and-audio-output.git
cd Object-detection-with-input-as-url-and-audio-output

### Step 2: Set up a virtual environment (optional but recommended)
You can create a virtual environment to keep dependencies isolated from your system Python.

python -m venv venv
source venv/bin/activate  # For Linux/MacOS
venv\Scripts\activate     # For Windows

### Step 3: Install the required dependencies
Use pip to install all required packages listed in requirements.txt.

pip install -r requirements.txt

### Step 4: Download YOLOv3 weights and config files
Download the following files from the YOLOv3 repository:

yolov3.weights (YOLOv3 pre-trained weights)
yolov3.cfg (YOLOv3 configuration file)
coco.names (File with class labels used by YOLO)

### Step 5: Run the server
After installing all the required packages and placing the YOLOv3 files, run the FastAPI server:

uvicorn app:app --reload
This will start the server at http://localhost:8000.

### Step 6: Access the web interface
Open your browser and navigate to http://localhost:8000. You will see a simple interface where you can paste an image URL. Once you enter the URL and click "Process Image," the image will be displayed with bounding boxes around detected objects, and the objects will be announced via audio.

Contributing
Feel free to fork the repository and submit pull requests. Contributions are welcome!
