from flask import Flask, request
from deploy.generate import generate_response
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
    prompt = request.json['prompt']
    response = generate_response(prompt)
    print(f'prompt: {prompt}\n response: {response}')

    return jsonify({
        prompt: prompt,
        response: response
    })
