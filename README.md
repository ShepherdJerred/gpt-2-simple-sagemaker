# gpt-2-simple-sagemaker
This is a wrapper around gpt-2-simple so that it can be easily used with AWS SageMaker. It allows you to upload your input and original GPT-2 models to S3 and use that for training. It is meant to first be uploaded to ECR and then used in a SageMaker training job.

## Usage
In SageMaker the following must be defined

### Hyperparameters
* steps (1000 is a good start)
* parameter_version (117M, 124M, 345M, 355M, 762M, 774M, 1558M)

### Input
#### GPT-2 Model
* Channel name: gpt-2
* Channel type: S3

#### Training Data
* Channel name: text
* Channel type: S3
* File name: input.txt

