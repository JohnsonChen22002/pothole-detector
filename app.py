import gradio as gr
from ultralytics import YOLO
from PIL import Image

# Load trained model
model = YOLO("runs/detect/pothole_detector2/weights/best.pt")  


def detect_potholes(image):
    # Run YOLOv8 model inference on the uploaded image
    results = model.predict(source=image, conf=0.25, save=False)

    # Draw bounding boxes and labels on the image
    return results[0].plot()  

gr.Interface(   
    fn=detect_potholes,                                                         # The function to call when the user uploads an image   
    inputs=gr.Image(type="pil"),                                                # Input: an image, passed as a PIL object to the function
    outputs=gr.Image(type="pil"),                                               # Output: a PIL image (with bounding boxes drawn)
    title="Pothole Detector",                                                   # Title shown at the top of the web app
    description="Upload an image and detect potholes with severity levels."     # Subtitle shown below the title
).launch()                                                                      # Starts the Gradio web server and opens the app
