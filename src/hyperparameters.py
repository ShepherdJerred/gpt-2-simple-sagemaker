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


def get_is_multi_gpu():
    return get_hyperparameters()['is_multi_gpu']


def get_batch_size():
    return int(get_hyperparameters()['batch_size'])


def get_learning_rate():
    return float(get_hyperparameters()['learning_rate'])


def get_accumulate_gradients():
    return float(get_hyperparameters()['accumulate_gradients'])


def get_sample_interval():
    return int(get_hyperparameters()['sample_interval'])


def get_sample_length():
    return int(get_hyperparameters()['sample_length'])


def get_sample_number():
    return int(get_hyperparameters()['sample_number'])


def get_save_interval():
    return int(get_hyperparameters()['save_interval'])


def get_print_interval():
    return int(get_hyperparameters()['print_interval'])
