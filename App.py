from flask import Flask, request, jsonify
import torch
from torchvision.transforms import transforms
from torchvision.models import resnet18
import torch.nn as nn
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
from torch.nn.functional import softmax
from flask_cors import CORS

from VGG import Vison_model
from Multimodal import Multimodal, Vision_model, Language_model
from ConvNext import ConvNextModel
from EfficientNetB0 import EfficientNetB0Model
from MobileNetV2 import MobileNetV2Model
from ViT import VisionTransformerModel
from ResNet18 import ResNet18Model
from Proposed_Multimodal_Model import Vision_model, Language_model, MultiHeadAttention, MultiCrossAttention, PositionalEncoding, ProposedMultimodalModel
from RNN import RNNModel
from LSTM import LSTMModel
from MLP import MLPModel


app = Flask(__name__)
CORS(app)

tokenizer = SentenceTransformer('sentence-transformers/use-cmlm-multilingual')
model = Multimodal(Vision_model,Language_model,num_classes=16)
model.load_state_dict(torch.load('C:/Users/Admin/Desktop/PBL6/model_checkpoint/checkpoint/version_1/checkpoint.pth'))
model.eval()

VGG = Vison_model()
checkpoint_vgg = 'C:/Users/Admin/Desktop/PBL6/model_checkpoint/checkpoint/version_1/VGG.pth'
VGG.load_state_dict(torch.load(checkpoint_vgg, map_location=torch.device('cpu')))
VGG.eval()

ConvNext = ConvNextModel()
checkpoint_convnet = 'C:/Users/Admin/Desktop/PBL6/model_checkpoint/checkpoint/version_1/ConvNext.pth'
ConvNext.load_state_dict(torch.load(checkpoint_convnet, map_location=torch.device('cpu')))
ConvNext.eval()

EfficientNetB0Model = EfficientNetB0Model()
checkpoint_efficientnetb0 = 'C:/Users/Admin/Desktop/PBL6/model_checkpoint/checkpoint/version_1/EfficientNetB0.pth'
EfficientNetB0Model.load_state_dict(torch.load(checkpoint_efficientnetb0, map_location=torch.device('cpu')))
EfficientNetB0Model.eval()

MobileNetV2Model = MobileNetV2Model()
checkpoint_mobilenetv2 = 'C:/Users/Admin/Desktop/PBL6/model_checkpoint/checkpoint/version_1/MobileNetV2.pth'
MobileNetV2Model.load_state_dict(torch.load(checkpoint_mobilenetv2, map_location=torch.device('cpu')))
MobileNetV2Model.eval()

VisionTransformerModel = VisionTransformerModel()
checkpoint_vit = 'C:/Users/Admin/Desktop/PBL6/model_checkpoint/checkpoint/version_1/ViT.pth'
VisionTransformerModel.load_state_dict(torch.load(checkpoint_vit, map_location=torch.device('cpu')))
VisionTransformerModel.eval()

ResNet18Model = ResNet18Model()
checkpoint_resnet18 = 'C:/Users/Admin/Desktop/PBL6/model_checkpoint/checkpoint/version_1/ResNet18.pth'
ResNet18Model.load_state_dict(torch.load(checkpoint_resnet18, map_location=torch.device('cpu')))
ResNet18Model.eval()

ProposedMultimodalModel = ProposedMultimodalModel(Vision_model,Language_model,num_classes=16)
checkpoint_proposed_multimodal = 'C:/Users/Admin/Desktop/PBL6/model_checkpoint/checkpoint/version_1/Proposed_Multimodal_Model.pth'
ProposedMultimodalModel.load_state_dict(torch.load(checkpoint_proposed_multimodal, map_location=torch.device('cpu')))
ProposedMultimodalModel.eval()

RNN = RNNModel(num_classes=16)
checkpoint_rnn = 'C:/Users/Admin/Desktop/PBL6/model_checkpoint/checkpoint/version_1/RNN.pth'
RNN.load_state_dict(torch.load(checkpoint_rnn, map_location=torch.device('cpu')))
RNN.eval()

LSTM = LSTMModel(num_classes=16)
checkpoint_lstm = 'C:/Users/Admin/Desktop/PBL6/model_checkpoint/checkpoint/version_1/LSTM.pth'
LSTM.load_state_dict(torch.load(checkpoint_lstm, map_location=torch.device('cpu')))
LSTM.eval()

MLP = MLPModel(num_classes=16)
checkpoint_mlp = 'C:/Users/Admin/Desktop/PBL6/model_checkpoint/checkpoint/version_1/MLP.pth'
MLP.load_state_dict(torch.load(checkpoint_mlp, map_location=torch.device('cpu')))
MLP.eval()

class_labels = {0: 'Cell Phone', 1: 'Chair', 2: 'Digital Camera', 3: 'Fridge', 4: 'Headphone', 
                5: 'Iron', 6: 'Keyboard', 7: 'Lamp', 8: 'Laptop', 9: 'Mouse', 
                10: 'Printer', 11: 'Speaker', 12: 'Table', 13: 'Tablet', 
                14: 'Television', 15: 'Vaccuum'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        results_multimodal = [] 
        results_vision = []
        results_convnext = []
        results_efficientnetb0 = []
        results_mobilenetv2 = []
        results_vit = []
        results_resnet18 = []
        results_proposed_multimodal = []
        results_rnn = []
        results_lstm = []
        results_mlp = []

        for i in range(1, 6): 
            text_input_name = f'text_input{i}'
            image_input_name = f'image_input{i}'

            text_input = request.form.get(text_input_name)
            image_file = request.form.get(image_input_name)

            if not text_input or not image_file:
                break

            text_encoded = tokenizer.encode(text_input, convert_to_tensor=True).unsqueeze(0)
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            image_tensor = image_transform(image).unsqueeze(0)  

            with torch.no_grad():
                # Predict using Multimodal Model
                prediction_logits_multimodal = model(image_tensor, text_encoded)
                prediction_probabilities_multimodal = softmax(prediction_logits_multimodal, dim=1)
                predicted_class_multimodal = prediction_probabilities_multimodal.argmax(dim=1).item()
                confidence_multimodal = prediction_probabilities_multimodal[0][predicted_class_multimodal].item()
                predicted_label_multimodal = class_labels.get(predicted_class_multimodal, 'Unknown Class')
                confidence_percentage_multimodal = f'{confidence_multimodal * 100:.2f}'
                results_multimodal.append((predicted_label_multimodal, confidence_percentage_multimodal, image_file))
                print(f'Multimodal Result {i}: {predicted_label_multimodal} with confidence {confidence_percentage_multimodal}%')

                # Predict using VGG Model
                prediction_logits_vision = VGG(image_tensor)
                prediction_probabilities_vision = softmax(prediction_logits_vision, dim=1)
                predicted_class_vision = prediction_probabilities_vision.argmax(dim=1).item()
                confidence_vision = prediction_probabilities_vision[0][predicted_class_vision].item()
                predicted_label_vision = class_labels.get(predicted_class_vision, 'Unknown Class')
                confidence_percentage_vision = f'{confidence_vision * 100:.2f}'
                results_vision.append((predicted_label_vision, confidence_percentage_vision, image_file))
                print(f'Vision Result {i}: {predicted_label_vision} with confidence {confidence_percentage_vision}%')
                
                # Predict using ConvNet Model
                prediction_logits_convnet = ConvNext(image_tensor)
                prediction_probabilities_convnet = softmax(prediction_logits_convnet, dim=1)
                predicted_class_convnet = prediction_probabilities_convnet.argmax(dim=1).item()
                confidence_convnet = prediction_probabilities_convnet[0][predicted_class_convnet].item()
                predicted_label_convnet = class_labels.get(predicted_class_convnet, 'Unknown Class')
                confidence_percentage_convnet = f'{confidence_convnet * 100:.2f}'
                results_convnext.append((predicted_label_convnet, confidence_percentage_convnet, image_file))
                print(f'ConvNet Result {i}: {predicted_label_convnet} with confidence {confidence_percentage_convnet}%')
                
                # Predict using EfficientNetB0 Model
                prediction_logits_efficientnetb0 = EfficientNetB0Model(image_tensor)
                prediction_probabilities_efficientnetb0 = softmax(prediction_logits_efficientnetb0, dim=1)
                predicted_class_efficientnetb0 = prediction_probabilities_efficientnetb0.argmax(dim=1).item()
                confidence_efficientnetb0 = prediction_probabilities_efficientnetb0[0][predicted_class_efficientnetb0].item()
                predicted_label_efficientnetb0 = class_labels.get(predicted_class_efficientnetb0, 'Unknown Class')
                confidence_percentage_efficientnetb0 = f'{confidence_efficientnetb0 * 100:.2f}'
                results_efficientnetb0.append((predicted_label_efficientnetb0, confidence_percentage_efficientnetb0, image_file))
                print(f'EfficientNetB0 Result {i}: {predicted_label_efficientnetb0} with confidence {confidence_percentage_efficientnetb0}%')
                
                # Predict using MobileNetV2 Model
                prediction_logits_mobilenetv2 = MobileNetV2Model(image_tensor)
                prediction_probabilities_mobilenetv2 = softmax(prediction_logits_mobilenetv2, dim=1)
                predicted_class_mobilenetv2 = prediction_probabilities_mobilenetv2.argmax(dim=1).item()
                confidence_mobilenetv2 = prediction_probabilities_mobilenetv2[0][predicted_class_mobilenetv2].item()
                predicted_label_mobilenetv2 = class_labels.get(predicted_class_mobilenetv2, 'Unknown Class')
                confidence_percentage_mobilenetv2 = f'{confidence_mobilenetv2 * 100:.2f}'
                results_mobilenetv2.append((predicted_label_mobilenetv2, confidence_percentage_mobilenetv2, image_file))
                print(f'MobileNetV2 Result {i}: {predicted_label_mobilenetv2} with confidence {confidence_percentage_mobilenetv2}%')
                
                # Predict using ViT Model
                prediction_logits_vit = VisionTransformerModel(image_tensor)
                prediction_probabilities_vit = softmax(prediction_logits_vit, dim=1)
                predicted_class_vit = prediction_probabilities_vit.argmax(dim=1).item()
                confidence_vit = prediction_probabilities_vit[0][predicted_class_vit].item()
                predicted_label_vit = class_labels.get(predicted_class_vit, 'Unknown Class')
                confidence_percentage_vit = f'{confidence_vit * 100:.2f}'
                results_vit.append((predicted_label_vit, confidence_percentage_vit, image_file))
                print(f'ViT Result {i}: {predicted_label_vit} with confidence {confidence_percentage_vit}%')
                
                # Predict using ResNet18 Model
                prediction_logits_resnet18 = ResNet18Model(image_tensor)
                prediction_probabilities_resnet18 = softmax(prediction_logits_resnet18, dim=1)
                predicted_class_resnet18 = prediction_probabilities_resnet18.argmax(dim=1).item()
                confidence_resnet18 = prediction_probabilities_resnet18[0][predicted_class_resnet18].item()
                predicted_label_resnet18 = class_labels.get(predicted_class_resnet18, 'Unknown Class')
                confidence_percentage_resnet18 = f'{confidence_resnet18 * 100:.2f}'
                results_resnet18.append((predicted_label_resnet18, confidence_percentage_resnet18, image_file))
                print(f'ResNet18 Result {i}: {predicted_label_resnet18} with confidence {confidence_percentage_resnet18}%')
                
                # Predict using RNN Model
                prediction_logits_rnn = RNN(text_encoded)
                prediction_probabilities_rnn = softmax(prediction_logits_rnn, dim=1)
                predicted_class_rnn = prediction_probabilities_rnn.argmax(dim=1).item()
                confidence_rnn = prediction_probabilities_rnn[0][predicted_class_rnn].item()
                predicted_label_rnn = class_labels.get(predicted_class_rnn, 'Unknown Class')
                confidence_percentage_rnn = f'{confidence_rnn * 100:.2f}'
                results_rnn.append((predicted_label_rnn, confidence_percentage_rnn, text_input))
                print(f'RNN Result {i}: {predicted_label_rnn} with confidence {confidence_percentage_rnn}%')
                
                # Predict using LSTM Model
                prediction_logits_lstm = LSTM(text_encoded)
                prediction_probabilities_lstm = softmax(prediction_logits_lstm, dim=1)
                predicted_class_lstm = prediction_probabilities_lstm.argmax(dim=1).item()
                confidence_lstm = prediction_probabilities_lstm[0][predicted_class_lstm].item()
                predicted_label_lstm = class_labels.get(predicted_class_lstm, 'Unknown Class')
                confidence_percentage_lstm = f'{confidence_lstm * 100:.2f}'
                results_lstm.append((predicted_label_lstm, confidence_percentage_lstm, text_input))
                print(f'LSTM Result {i}: {predicted_label_lstm} with confidence {confidence_percentage_lstm}%')
                
                # Predict using MLP Model
                prediction_logits_mlp = MLP(text_encoded)
                prediction_probabilities_mlp = softmax(prediction_logits_mlp, dim=1)
                predicted_class_mlp = prediction_probabilities_mlp.argmax(dim=1).item()
                confidence_mlp = prediction_probabilities_mlp[0][predicted_class_mlp].item()
                predicted_label_mlp = class_labels.get(predicted_class_mlp, 'Unknown Class')
                confidence_percentage_mlp = f'{confidence_mlp * 100:.2f}'
                results_mlp.append((predicted_label_mlp, confidence_percentage_mlp, text_input))
                print(f'MLP Result {i}: {predicted_label_mlp} with confidence {confidence_percentage_mlp}%')
                
                # Predict using Proposed Multimodal Model
                prediction_logits_proposed_multimodal = ProposedMultimodalModel(image_tensor, text_encoded)
                prediction_probabilities_proposed_multimodal = softmax(prediction_logits_proposed_multimodal, dim=1)
                predicted_class_proposed_multimodal = prediction_probabilities_proposed_multimodal.argmax(dim=1).item()
                confidence_proposed_multimodal = prediction_probabilities_proposed_multimodal[0][predicted_class_proposed_multimodal].item()
                predicted_label_proposed_multimodal = class_labels.get(predicted_class_proposed_multimodal, 'Unknown Class')
                confidence_percentage_proposed_multimodal = f'{confidence_proposed_multimodal * 100:.2f}'
                results_proposed_multimodal.append((predicted_label_proposed_multimodal, confidence_percentage_proposed_multimodal, image_file))
                print(f'Proposed Multimodal Result {i}: {predicted_label_proposed_multimodal} with confidence {confidence_percentage_proposed_multimodal}%')
                
            i += 1

        return jsonify(results_multimodal=results_multimodal, results_vision=results_vision, results_convnext=results_convnext, results_efficientnetb0=results_efficientnetb0, results_mobilenetv2=results_mobilenetv2, results_vit=results_vit, results_resnet18=results_resnet18, results_rnn=results_rnn, results_lstm=results_lstm, results_mlp=results_mlp, results_proposed_multimodal=results_proposed_multimodal)
    
if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8080)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         results = [] 
#         for i in range(1, 6): 
#             text_input_name = f'text_input{i}'
#             image_input_name = f'image_input{i}'

#             text_input = request.form.get(text_input_name)
#             image_file = request.form.get(image_input_name)

#             if not text_input or not image_file:
#                 break

#             text_encoded = tokenizer.encode(text_input, convert_to_tensor=True).unsqueeze(0)
#             response = requests.get(image_file)
#             image = Image.open(BytesIO(response.content)).convert('RGB')
#             image_transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#             ])
#             image_tensor = image_transform(image).unsqueeze(0)  

#             with torch.no_grad():
#                 prediction_logits = model(image_tensor, text_encoded)
#                 prediction_probabilities = softmax(prediction_logits, dim=1)
#                 predicted_class = prediction_probabilities.argmax(dim=1).item()
#                 confidence = prediction_probabilities[0][predicted_class].item()

#             predicted_label = class_labels.get(predicted_class, 'Unknown Class')
#             confidence_percentage = f'{confidence * 100:.2f}'

#             results.append((predicted_label, confidence_percentage,image_file)) 

#             i += 1

#         return results

# if __name__ == '__main__':
#     app.run(debug=True, host='localhost', port=8080)