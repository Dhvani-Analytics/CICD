from PIL import Image
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import numpy as np
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
from skimage import io
import boto3
import json
from flask import Flask
from flask_cors import CORS, cross_origin
from torch.autograd import Variable
import torchvision.transforms.functional as TF
import json
from pycocotools import mask
from skimage import measure
from torchvision import transforms

def find_bucket_key(s3_path, id):

    s3_components = s3_path.split('/')
    bucket = s3_components[2].split('.')
    bucket_name = bucket[0]
    key = s3_components[len(s3_components)-1]
    conn = boto3.client('s3','ap-southeast-1')
    response = conn.get_object(Bucket=bucket_name, Key=key)
    body = response['Body']
    image1 = Image.open(body)
    image =np.array(image1)
    io.imsave("preprocessedimage_"+str(id)+".jpg", image)  



app = Flask(__name__)
CORS(app, resources={r"/home": {"origins": "*"}})


@app.route('/home', methods=['POST'])
#@cross_origin()
def home():

    if request.method == 'POST':

        data = request.get_json()      

        id= data.get('id')
        s3_bucketpath=data.get('path')
        find_bucket_key (s3_bucketpath, id)
        inference_model = data.get('model') 
        filepath ="preprocessedimage_"+str(id)+".jpg"

        #filepath = s3_bucketpath

        input_image = Image.open(filepath)
        image_size = input_image.size
        width,height=image_size[0],image_size[1]

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        model.eval()
        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes

        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        # Read the categories
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        # Show top categories per image
        top_prob, top_catid = torch.topk(probabilities, 1)

        output = {"class":categories[top_catid[0]],
                    "confidence" : top_prob[0].item()}                
        
        return json.dumps(dict(output))

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0',port=5000)
    