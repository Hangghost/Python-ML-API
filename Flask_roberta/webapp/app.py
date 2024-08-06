from flask import Flask, request, jsonify
import torch
import numpy as np
from transformers import RobertaTokenizer
import onnxruntime

app = Flask(__name__)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# ort_session = onnxruntime.InferenceSession("roberta-sequence-classification-9.onnx")
ort_session = onnxruntime.InferenceSession("roberta_Opset18.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


@app.route('/')
def home():
    return '<h2>RoBERTa Sentiment Analysis API<h2>'


@app.route('/predict', methods=['POST'])
def predict():
    input_ids = torch.tensor(
        tokenizer.encode(request.json[0], add_special_tokens=True)
        ).unsqueeze(0)
    inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids)}
    out = ort_session.run(None, inputs)

    result = np.argmax(out)

    return jsonify({'positive': bool(result)})


if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')
