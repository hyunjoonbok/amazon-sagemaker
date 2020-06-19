# Amazon-sagemaker

![logo](./img/sagemaker-logo.jpeg)


SageMaker uses Python SDK which is an open source library for training and deploying machine learning models on Amazon SageMaker.

With the SDK, you can train and deploy models using popular deep learning frameworks Apache MXNet and TensorFlow. You can also train and deploy models with Amazon algorithms, which are scalable implementations of core machine learning algorithms that are optimized for SageMaker and GPU training. If you have your own algorithms built into SageMaker compatible Docker containers, you can train and host models using these as well.

<hr>

## Motivation

This repository contains End-to-End Amazon Sagemaker usages from model builing to deployment with integrated AWS services. You should be able to get a firm grasp of how to use Amazon Sagemaker to build your service. Amazon Sagemaker allows us to use their own jupyter notebooks with tons of pre-built algorithms (or your own python script file) to perform the training and deploying in a very simple manner. I believe this is one of the needed skills to learn for any Data Scientists, not just for small organaztion, but for big corporates who wants to deploy model at scale and save incurring costs. This repo contains a numerious real-life notebooks created by [@hyunjoonbok](https://www.linkedin.com/in/hyunjoonbok/) utilizing codes from differnet sources (i.e. Amazon Sagemaker Tutorials)


## Table of contents
* [Setup](#Setup)
* [SageMaker Examples](#SageMaker)
* [Contact](#Contact)

<hr>

## Setup

First things first, you should set up authentication credentials. Credentials for your AWS account can be found in the IAM Console. You can create or use an existing user. Go to manage access keys and generate a new set of keys. For a new AWS user, go to [AWS website](https://aws.amazon.com/), and create an account. 

Then, you have to install AWS Command Line Interface (AWS CLI) depending on your OS. For downloading instructions, please refer to [AWS CLI Documentation](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html).

Finally, if you have the AWS CLI installed, then you can use it to configure your credentials file. Type in your CMD:

```
aws configure
```

Alternatively, you can create the credential file yourself. By default, its location is at ~/.aws/credentials:
```
[default]
aws_access_key_id = YOUR_ACCESS_KEY \ 
aws_secret_access_key = YOUR_SECRET_KEY \
```
You may also want to set a default region. This can be done in the configuration file. By default, its location is at ~/.aws/config: 
```
[default]
region=us-east-1
```

<hr>

Now you have to perform several more steps to actually use Amazon Sagemaker

### Create S3 Bucket

First, you have to create an example S3 Bucket for to use in training. 

Log-in to your AWS Management Console with credentials created above, serach for S3, selecet the region you want to create your storage on the top-right corner, and create a new bucket.

For a detailed guide on how to create a bucket, please refer to [AWS S3 doc](https://docs.aws.amazon.com/AmazonS3/latest/gsg/GetStartedWithS3.html)

### Install Boto3
Boto3 is the AWS SDK for Python. Boto3 makes it easy to integrate your Python application

Open up your Command Line Prompt, and type:

```
pip install boto3
```

### Install Sagemaker python-sdk

Open up your Command Line Prompt, and type:
- From PyPI

```
pip install sagemaker
```
- From Source
```
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

For detailed installing information, please refer to Sagemaker Python SDK [Github](https://github.com/aws/sagemaker-python-sdk)

<hr>

## SageMaker

These examples display unique functionality available in Amazon SageMaker. They cover a broad range of topics with different method that user can utilize inside SageMaker.

   #### [Image Classification on CIFAR-10 (transfer-learning)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Image%20Classification%20on%20CIFAR-10%20(transfer-learning)%20in%20Amazon%20SageMaker.ipynb) 
   <p>
    We use transfer learning mode in Sagemaker to fine-tune a pre-trained model (trained on imagenet data) to learn to classify a new dataset. The pre-trained model (ImageNet) will be fine-tuned using CIFAR-10 dataset. This covers a complete model-building cycle (dataloading, preprocessing, Hyperparmeter setting, Sagemaker training, Investigating loss, Testing, Model Deployment) using Amazon SageMaker.
	</p>
  
   #### [Movie recommendation with Factorization Machines](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Movie%20recommendation%20with%20Factorization%20Machines%20on%20Amazon%20SageMaker.ipynb) 
   <p>
    In this notebook, we are going to build a simple movie recommendation model with Factorization machine using Amazon Sagemaker. Covers full AWS integreation (using S3 to store objects, and booting up EC2 instance in local Jupyter Notebook)
	</p>  
   
   #### [Amazon SageMaker AutoPilot (ML Tabular Problem)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Amazon%20SageMaker%20AutoPilot%20(model%20building%20to%20deploy).ipynb) 
   <p>
    We cover SageMaker's AutoML feature, known as Autopilot, which automatically trains and tunes the best machine learning models for classification or regression, based on data while allowing to maintain full control and visibility. It takes care of all data preprocessing steps for users, so it is super useful when trying to quickly build a ML solution prototype. Here we look at the ML exmaple with tabular data to predict whether the client is enrolled in product(bank term deposit).
	</p>  

   #### [Advanced Amazon SageMaker AutoPilot (Hyperparameter tuning)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/(Advanced)%20Amazon%20SageMaker%20AutoPilot%20(Hyperparameter%20tuning).ipynb) 
   <p>
    Here we specifically look at how we could fine-tune hyperparamers using the same example in the basic Autopilot example above. It shows how Sagemakers' fast computing resources takes care of tuning much faster and cost-efficient than any other services. We can also inspect training and tuning jobs within Amazon SageMaker Experiments console (so it never goes away!). Also covers a complete model-building cycle.
	</p>  

   #### [Sagemaker Script mode (ML - XGboost))](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Sagemaker%20Script%20mode%20usage%20(ML%20-%20XGboost).ipynb) 
   <p>
    Introduce a script-mode in Sagemaker where a user can bring their own Python file. Script mode is a very useful technique that lets you easily run your existing code in Amazon SageMaker, with very little change in codes. This gives more flexibility in the traning without having to worry about building containers or managing the underlying infrastructure. This time, we tackle the simple Deep Learning problem (MNIST) with Tensorflow
	</p>  
  
   #### [Sagemaker Script mode (DL - Tensorflow)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Sagemaker%20Script%20mode%20usage%20(DL%20-%20Tensorflow).ipynb) 
   <p>
    The similar script mode codesets as above, but this time we look at the example of Deep Learning problem (MNIST) with Tensorflow.
	</p>    

##### For more information on Script mode, plese refer to [AWS Blog](https://aws.amazon.com/blogs/machine-learning/using-tensorflow-eager-execution-with-amazon-sagemaker-script-mode/)


<hr>


## Contact
Created by [@hyunjoonbok](https://www.linkedin.com/in/hyunjoonbok/) - feel free to contact me!


## To-Do 
- Consume official AWS sagemaker [github](https://github.com/awslabs/amazon-sagemaker-examples)
- Consume official AWS sagemaker script mode [github](https://github.com/aws-samples/amazon-sagemaker-script-mode)


## References 
- Julien Simon's [Gitlab](https://gitlab.com/juliensimon/dlnotebooks/-/tree/master/sagemaker)
- [Boto3 setup](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/sqs.html)
- Official amazon-sagemaker-examples [Github](https://github.com/awslabs/amazon-sagemaker examples/blob/master/advanced_functionality/README.md)
