import gpt_2_simple as gpt2
import hyperparameters as hp

checkpoint_directory = hp.get_checkpoint_directory()
sample_directory = hp.get_sample_directory()


def generate_response(text):
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess=sess,
                   checkpoint_dir=checkpoint_directory,
                   model_name=None,
                   model_dir='models',
                   multi_gpu=False)

    result = gpt2.generate(sess=sess,
                           run_name='run1',
                           checkpoint_dir=checkpoint_directory,
                           model_name=None,
                           sample_dir=sample_directory,
                           return_as_list=True,
                           truncate='<|endoftext|>',
                           destination_path=None,
                           sample_delim='=' * 20 + '\n',
                           prefix=f'<|startoftext|>\n{text}',
                           seed=None,
                           nsamples=1,
                           batch_size=1,
                           length=1023,
                           temperature=0.7,
                           top_k=0,
                           top_p=0.0,
                           include_prefix=False)
    print(f'prompt: {text}\nresult: {result[0]}')
    return result[0]
