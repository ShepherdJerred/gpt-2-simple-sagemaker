import json
from os import path

SAGEMAKER_HYPERPARAMETERS_PATH = '/opt/ml/input/config/hyperparameters.json'
SAGEMAKER_MODEL_PATH = '/opt/ml/input/data/gpt-2/'
SAGEMAKER_INPUT_PATH = '/opt/ml/input/data/text/input.txt'


def get_hyperparameters():
    with open(SAGEMAKER_HYPERPARAMETERS_PATH, 'r') as file:
        file_json_content = json.load(file)
        print(json.dumps(file_json_content))
        return file_json_content


def get_gpt2_parameter_version():
    valid = ['117M', '124M', '345M', '355M', '762M', '774M', '1558M']
    candidate = get_hyperparameters()['parameter_version']
    if candidate in valid:
        return candidate
    else:
        raise Exception(f'Invalid parameter version ${candidate}')


def get_gpt2_model_path():
    gpt2_parameter_version = get_gpt2_parameter_version()
    model_path = SAGEMAKER_MODEL_PATH + gpt2_parameter_version
    if path.exists(model_path):
        return SAGEMAKER_MODEL_PATH
    else:
        raise Exception(f'Directory does not exist: ${model_path}')


def get_steps():
    return get_hyperparameters()['steps']


def get_input_file_path():
    candidate = SAGEMAKER_INPUT_PATH
    if path.exists(candidate):
        return candidate
    else:
        raise Exception(f'Path does not exist: ${candidate}')
