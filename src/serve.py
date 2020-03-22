from flask import Flask, request
from generate import generate_response
from flask import jsonify

app = Flask(__name__)


@app.route('/')
def index():
    return f'SageMaker Inferencing'


@app.route('/ping')
def ping():
    return jsonify('pong')


@app.route('/invocations', methods=['POST'])
def invocations():
    model_input = request.json['input']
    model_output = generate_response(model_input)
    print(f'input: {model_input}\n output: {model_output}')

    return jsonify({
        model_output: model_output
    })
