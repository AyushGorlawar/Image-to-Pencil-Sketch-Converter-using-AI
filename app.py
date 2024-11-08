from flask import Flask, render_template, request, redirect, url_for
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load pretrained model for style transfer with updated weights parameter
model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).eval()

def convert_to_sketch(image_path):
    image = cv2.imread(image_path)
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_image = cv2.bitwise_not(gray_image)
    blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(gray_image, 255 - blurred_image, scale=256)

    # Ensure the sketch is 3-channel for display or saving
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    return sketch_rgb


def enhance_with_ai(sketch):
    # Convert grayscale sketch to RGB for compatibility with VGG model
    pil_image = Image.fromarray(sketch).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension

    # Use only feature extraction layers from VGG19
    feature_extractor = torch.nn.Sequential(*list(model.features.children())[:21])

    with torch.no_grad():
        output = feature_extractor(input_tensor)

    # Reduce the output tensor to a single 2D image by averaging across channels
    output_image = output.squeeze(0).mean(dim=0).numpy()

    # Normalize and convert to uint8 format for saving as an image
    output_image = cv2.normalize(output_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert single-channel image to 3-channel RGB for saving
    output_image_rgb = cv2.merge([output_image] * 3)
    
    return output_image_rgb


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Convert to sketch
        sketch = convert_to_sketch(filepath)

        # Save enhanced sketch (just the sketch in this case)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'enhanced_' + file.filename)
        cv2.imwrite(output_path, sketch)

        return render_template('index.html', 
                               uploaded_image=url_for('static', filename='uploads/' + file.filename),
                               sketch_image=url_for('static', filename='uploads/enhanced_' + file.filename))

if __name__ == "__main__":
    app.run(debug=True)
