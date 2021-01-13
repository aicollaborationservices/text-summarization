import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import yaml

app = Flask(__name__)
cors = CORS(app)

model = AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6')
tokenizer = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
device = torch.device('cpu')


API_V1 = '/api/1.0'

@app.route(API_V1 + '/ping', methods=['GET'])
def ping():
    return "pong"

@app.route(API_V1 + '/definition', methods=['GET'])
def definition():
    with open("./openapi.yml", 'r') as stream:
        try:
            return jsonify(yaml.safe_load(stream))
        except yaml.YAMLError as exception:
            return jsonify(exception)

@app.route(API_V1 + '/info', methods=['GET'])
def info():
    return jsonify({
        'version': API_V1,
        'project': '5 elements of AI',
        'service': 'text-summarization',
        'language': 'python',
        'type': 'api',
        'date': str(datetime.datetime.now()),
    })


@app.route(API_V1 + '/predict', methods=['POST', 'OPTIONS'])
@cross_origin(origin='localhost')
def predict():
    data = request.json
    
    preprocess_text = data['context'].strip().replace("\n","")
    config = data ['config']
    config_max_length = config ['max_length']

    print ("original text preprocessed: \n", preprocess_text)

    input = tokenizer.batch_encode_plus([preprocess_text], return_tensors='pt', max_length=1024)['input_ids'].to(device)

    summary_ids = model.generate(input, num_beams=4, length_penalty=2.0, max_length=config_max_length, no_repeat_ngram_size=3, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify({ 'summary': summary })
   

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
