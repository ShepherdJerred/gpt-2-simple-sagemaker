import gpt_2_simple as gpt2
import hyperparameters as hp

SAGEMAKER_CHECKPOINT_DIR = '/opt/ml/checkpoints'
SAGEMAKER_SAMPLE_DIR = '/opt/ml/model/samples'

input_file = hp.get_input_file_path()
model_dir = hp.get_gpt2_model_path()
model_parameter_version = hp.get_gpt2_parameter_version()
steps = hp.get_steps()

session = gpt2.start_tf_sess()
gpt2.finetune(sess=session,
              dataset=input_file,
              steps=steps,
              model_name=model_parameter_version,
              model_dir=model_dir)

gpt2.generate(session)
