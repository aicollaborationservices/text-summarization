import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch
import yaml

app = Flask(__name__)
cors = CORS(app)

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
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
    t5_prepared_Text = "summarize: "+ preprocess_text
    print ("original text preprocessed: \n", preprocess_text)

    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    
    summary_ids = model.generate(
        tokenized_text,
        num_beams=4,
        no_repeat_ngram_size=2,
        min_length=30,
        max_length=100,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify({ 'summary': summary })
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)