import json
from os import path

from sagemaker import SAGEMAKER_HYPERPARAMETERS_PATH, SAGEMAKER_MODEL_PATH, \
    SAGEMAKER_INPUT_PATH, SAGEMAKER_CHECKPOINT_DIR, SAGEMAKER_SAMPLE_DIR


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
    return int(get_hyperparameters()['accumulate_gradients'])


def get_sample_interval():
    return int(get_hyperparameters()['sample_interval'])


def get_sample_length():
    return int(get_hyperparameters()['sample_length'])


def get_sample_number():
    return int(get_hyperparameters()['sample_number'])


def get_save_interval():
    return int(get_hyperparameters()['save_interval'])


def get_status_print_interval():
    """
    Controls how often the status (step number, loss) are printed to the console.
    :return: The print interval in steps.
    """
    return int(get_hyperparameters()['print_interval'])


def get_checkpoint_directory():
    """
    Sets where checkpoints should be saved.
    :return: The directory that checkpoints should be saved to.
    """
    return SAGEMAKER_CHECKPOINT_DIR


def get_sample_directory():
    """
    Sets where text samples generated during training should be saved.
    :return: The directory where samples should be saved.
    """
    return SAGEMAKER_SAMPLE_DIR
