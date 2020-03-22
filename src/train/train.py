import gpt_2_simple as gpt2
from train import hyperparameters as hp

input_file = hp.get_input_file_path()
model_dir = hp.get_gpt2_model_path()
steps = hp.get_steps()
is_multi_gpu = hp.get_is_multi_gpu()
batch_size = hp.get_batch_size()
learning_rate = hp.get_learning_rate()
accumulate_gradients = hp.get_accumulate_gradients()
sample_interval = hp.get_sample_interval()
sample_length = hp.get_sample_length()
sample_number = hp.get_sample_count()
save_interval = hp.get_save_interval()
print_interval = hp.get_status_print_interval()
checkpoint_directory = hp.get_checkpoint_directory()
combine = hp.get_combine_input_size()
restore_from = hp.get_restore_from()
run_name = hp.get_run_name()
max_checkpoints = hp.get_max_checkpoints()
should_use_memory_saving_gradients = hp.get_should_use_memory_saving_gradients()
should_only_train_transform_layers = hp.get_should_only_train_transform_layers()
optimizer = hp.get_optimizer()
should_overwrite = hp.get_should_overwrite()

session = gpt2.start_tf_sess()

print('beginning training')

gpt2.finetune(sess=session,
              dataset=input_file,
              steps=steps,
              model_name='',
              model_dir=model_dir,
              combine=combine,
              batch_size=batch_size,
              learning_rate=learning_rate,
              accumulate_gradients=accumulate_gradients,
              restore_from=restore_from,
              run_name=run_name,
              checkpoint_dir=checkpoint_directory,
              sample_every=sample_interval,
              sample_length=sample_length,
              sample_num=sample_number,
              multi_gpu=is_multi_gpu,
              save_every=save_interval,
              print_every=print_interval,
              max_checkpoints=max_checkpoints,
              use_memory_saving_gradients=should_use_memory_saving_gradients,
              only_train_transformer_layers=should_only_train_transform_layers,
              optimizer=optimizer,
              overwrite=should_overwrite)

print('training complete')
