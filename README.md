# gpt-2-simple-sagemaker
Train gpt-2-simple using AWS SageMaker

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

