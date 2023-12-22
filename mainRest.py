import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from io import BytesIO
import torch.nn.functional as F  # Import the functional module for additional functions

import level1, ela
from model import IMDModel

app = Flask(__name__)

def infer(img_path, model, device):
    print("Performing Level 1 analysis...")
    level1.findMetadata(img_path=img_path)

    print("Performing Level 2 analysis...")
    ela.ELA(img_path=img_path)

    img = Image.open("temp/ela_img.jpg")
    img = img.resize((128,128))
    img = np.array(img, dtype=np.float32).transpose(2,0,1)/255.0
    img = np.expand_dims(img, axis=0)

    out = model(torch.from_numpy(img).to(device=device))
    y_pred = torch.max(out, dim=1)[1]
    probabilities = F.softmax(out, dim=1)
    predicted_probability = probabilities[0, y_pred].item()


    print("Prediction:",end=' ')
    print("Authentic" if y_pred else "Tampared") # auth -> 1 and tp -> 0
    print("Probability:",predicted_probability)
    result = {
        'predicted_class': "Authentic" if y_pred else "Tampered"
    }
    return result

@app.route('/detect_manipulation', methods=['POST'])
def detect_manipulation():
    try:
        image = request.files['image'].read()
        img = Image.open(BytesIO(image))
        img_path = "temp/uploaded_image.jpg"
        img.save(img_path,quality=100)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        model_path = "model/model_c1.pth"
        model = torch.load(model_path)
        result = infer(img_path,model,device)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Working on", device)

    model_path = "model/model_c1.pth"
    model = torch.load(model_path)
    app.run(debug=True)
