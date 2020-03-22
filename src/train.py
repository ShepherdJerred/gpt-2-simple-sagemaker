import gpt_2_simple as gpt2
import hyperparameters as hp

input_file = hp.get_input_file_path()
model_dir = hp.get_gpt2_model_path()
model_parameter_version = hp.get_gpt2_parameter_version()
steps = hp.get_steps()
is_multi_gpu = hp.get_is_multi_gpu()
batch_size = hp.get_batch_size()
learning_rate = hp.get_learning_rate()
accumulate_gradients = hp.get_accumulate_gradients()
sample_interval = hp.get_sample_interval()
sample_length = hp.get_sample_length()
sample_number = hp.get_sample_number()
save_interval = hp.get_save_interval()
print_interval = hp.get_status_print_interval()
checkpoint_directory = hp.get_checkpoint_directory()

session = gpt2.start_tf_sess()
gpt2.finetune(sess=session,
              dataset=input_file,
              steps=steps,
              model_name=model_parameter_version,
              model_dir=model_dir,
              combine=50000,
              batch_size=batch_size,
              learning_rate=learning_rate,
              accumulate_gradients=accumulate_gradients,
              restore_from='latest',
              run_name='run1',
              checkpoint_dir=checkpoint_directory,
              sample_every=sample_interval,
              sample_length=sample_length,
              sample_num=sample_number,
              multi_gpu=is_multi_gpu,
              save_every=save_interval,
              print_every=print_interval,
              max_checkpoints=1,
              use_memory_saving_gradients=False,
              only_train_transformer_layers=False,
              optimizer='adam',
              overwrite=False)

gpt2.generate(session)
