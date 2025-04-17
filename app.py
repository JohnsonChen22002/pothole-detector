import gradio as gr
from ultralytics import YOLO
from PIL import Image

# Load your trained model
model = YOLO("runs/detect/pothole_detector2/weights/best.pt")  # Adjust path if needed

# Define prediction function
def detect_potholes(image):
    results = model.predict(source=image, conf=0.25, save=False)
    return results[0].plot()  # Returns a PIL image with bounding boxes + labels

# Launch web app
gr.Interface(
    fn=detect_potholes,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Pothole Detector",
    description="Upload an image and detect potholes with severity levels."
).launch()
