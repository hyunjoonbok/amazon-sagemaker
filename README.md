# Amazon-sagemaker

![logo](./img/sagemaker-logo.jpeg)


SageMaker uses Python SDK which is an open source library for training and deploying machine learning models on Amazon SageMaker.

With the SDK, you can train and deploy models using popular deep learning frameworks Apache MXNet and TensorFlow. You can also train and deploy models with Amazon algorithms, which are scalable implementations of core machine learning algorithms that are optimized for SageMaker and GPU training. If you have your own algorithms built into SageMaker compatible Docker containers, you can train and host models using these as well.

<hr>

## Motivation

This repository contains End-to-End Amazon Sagemaker usages from model builing to deployment with integrated AWS services. You should be able to get a firm understading of how to use Amazon Sagemaker to build your service. Amazon Sagemaker allows us to use their own jupyter notebooks with tons of pre-built algorithms (or your own python script file) to perform the training and deploying in a very simple manner. I believe this is one of the needed skills to learn for any Data Scientists, not just for small organaztion, but for big corporates who wants to deploy model at scale and save incurring costs. This repo contains a numerious real-life notebooks created by [@hyunjoonbok](https://www.linkedin.com/in/hyunjoonbok/) utilizing codes from differnet sources (i.e. Amazon Sagemaker Tutorials)


## Table of contents
* [Setup](#Setup)
* [SageMaker Examples](#SageMaker)
* [Contact](#Contact)

<hr>

## Setup

First things first, you should set up authentication credentials. Credentials for your AWS account can be found in the IAM Console. You can create or use an existing user. Go to manage access keys and generate a new set of keys. For a new AWS user, go to [AWS website](https://aws.amazon.com/), and create an account. 

Then, you have to install AWS Command Line Interface (AWS CLI) depending on your OS. For downloading instructions, please refer to [AWS CLI Documentation](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html).

Finally, if you have the AWS CLI installed, then you can use it to configure your credentials file. 
For Windows, type below command to CMD(window+R), or Bash if you are Mac User

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

	
   #### [Unsupervised Anomaly Detection in Amazon SageMaker (IP address detection)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Unsupervised%20Anomaly%20Detection%20in%20Amazon%20SageMaker%20(IP%20address%20detection).ipynb) 
   <p>
    In this notebook, we use the Amazon SageMaker IP-Insights algorithm to train a model on synthetic data. We then use this model to perform inference on the data and show how to discover suspicious IP addresses. It ingests historical data as (entity, IPv4 Address) pairs and learns the IP usage patterns of each entity. When queried with an (entity, IPv4 Address) event, an Amazon SageMaker IP Insights model returns a score that infers how anomalous the pattern of the event is.
	</p>
	
   #### [FULL Anomaly Detection Coverage on Real Businses Problem with AWS SageMaker](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/FULL%20Anomaly%20Detection%20Coverage%20on%20Real%20Businses%20Problem%20with%20AWS%20SageMaker.ipynb) 
   <p>
    This notebook shows how to train an Amazon SageMaker model to flag anomalous Medicare claims and target them for further investigation on suspicion of fraud. THe solution is to create a model to flag suspicious claims. The difference between data normality and abnormality is often not clear. Anomaly detection methods could be application-specific. For example, in clinical data, a small deviation could be an outlier, but in a marketing application, you need a significant deviation to justify an outlier. Noise in data may appear as deviations in attribute values or missing values. Noise may hide an outlier or flag deviation as an outlier. Providing clear justification for an outlier may be difficult. AWS Sagemaker as a fully managed service to go through complete model building.
	</p>	

   #### [Customer Churn Prediction using XGBoost with Amazon SageMaker](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Customer%20Churn%20Prediction%20using%20XGBoost%20with%20Amazon%20SageMaker.ipynb) 
   <p>
    This notebook describes using machine learning (ML) for the automated identification of unhappy customers, also known as customer churn prediction. ML models rarely give perfect predictions though, so this notebook is also about how to incorporate the relative costs of prediction mistakes when determining the financial outcome of using ML. We use an example of churn that is familiar to all of us–leaving a mobile phone operator. Seems like I can always find fault with my provider du jour! And if my provider knows that I’m thinking of leaving, it can offer timely incentives–I can always use a phone upgrade or perhaps have a new feature activated–and I might just stick around. Incentives are often much more cost effective than losing and reacquiring a customer.
	</p>
	
  
   #### [Recommendation System on Amazon SageMaker - Beginner (Factorization Machine)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Recommendation%20System%20on%20Amazon%20SageMaker%20-%20Beginner%20(Factorization%20Machine).ipynb) 
   <p>
    In this notebook, we are going to build a simple movie recommendation model with Factorization-Machine algorithm using Amazon Sagemaker. Covers full AWS integreation (using S3 to store objects, and booting up EC2 instance in local Jupyter Notebook). 
	</p> 
	
   #### [Recommendation System on Amazon SageMaker - Advanced (Factorization Machine)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Recommendation%20System%20on%20Amazon%20SageMaker%20-%20Advanced%20(Factorization%20Machine).ipynb) 
   <p>
    This notebook incorportates advanced methods to build a recommendation model in Amazon SageMaker. We are going utilize Apache Airflow with Amazaon SageMaker, and this model would give recommendataion based on the Amazon customer ratings on over 160K digital videos.
	</p>  	
   
   #### [Recommendation System on Amazon SageMaker (ObjectToVec)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Recommendation%20System%20on%20Amazon%20SageMaker%20(ObjectToVec).ipynb) 
   <p>
    This notebook covers another algorithm Object2Vec, which is a highly customizable multi-purpose algorithm that can learn embeddings of pairs of objects, to build a recommendation model in SageMaker. We process the data with Spark, train it with XGBoost and deploy as Inference Pipeline
	</p>  

   #### [Introduction to using Optuna for Hyper Parameter Optimization with PyTorch and MNIST on Amazon SageMaker](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/%20Introduction%20to%20using%20Optuna%20for%20Hyper%20Parameter%20Optimization%20with%20PyTorch%20and%20MNIST%20on%20Amazon%20SageMaker.ipynb) 
   <p>
    This notebook covers how we perform HyperParameter Optimization using Optuna and its reference architecture in Amazon SageMaker. Amazon SageMaker supports various frameworks and interfaces such as TensorFlow, Apache MXNet, PyTorch, scikit-learn, Horovod, Keras, and Gluon. The service offers ways to build, train, and deploy machine learning models to all developers and data scientists. Amazon SageMaker offers managed Jupyter Notebook and JupyterLab as well as containerized environments for training and deployment. The service also offers an Automatic Model Tuning with Bayesian HPO feature by default.
	</p>  
      
   #### [Amazon SageMaker AutoPilot (ML Tabular Problem)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Amazon%20SageMaker%20AutoPilot%20(model%20building%20to%20deploy).ipynb) 
   <p>
    We cover SageMaker's AutoML feature, known as Autopilot, which automatically trains and tunes the best machine learning models for classification or regression, based on data while allowing to maintain full control and visibility. It takes care of all data preprocessing steps for users, so it is super useful when trying to quickly build a ML solution prototype. Here we look at the ML exmaple with tabular data to predict whether the client is enrolled in product(bank term deposit).
	</p>  

   #### [Advanced Amazon SageMaker AutoPilot (Hyperparameter tuning)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/(Advanced)%20Amazon%20SageMaker%20AutoPilot%20(Hyperparameter%20tuning).ipynb) 
   <p>
    Here we specifically look at how we could fine-tune hyperparamers using the same example in the basic Autopilot example above. It shows how Sagemakers' fast computing resources takes care of tuning much faster and cost-efficient than any other services. We can also inspect training and tuning jobs within Amazon SageMaker Experiments console (so it never goes away!). Also covers a complete model-building cycle.
	</p> 
	
	
   #### [Abalone Age Prediction (regression problem) using XGBoost with Amazon SageMaker, Spark Pipeline, and AWS Glue](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Predict%20the%20age%20of%20Abalone%20(regression%20problem)%20with%20Amazon%20SageMaker%2C%20Spark%20Pipeline%2C%20and%20AWS%20Glue%20.ipynb) 
   <p>
    This notebook shows how to build a prediction model to determine age of an Abalone (a kind of shellfish) from its physical measurements, using Feature processing with Spark, training with XGBoost and deploying as Inference Pipeline. In this notebook, we use Amazon Glue to run serverless Spark. Though the notebook demonstrates the end-to-end flow on a small dataset, the setup can be seamlessly used to scale to larger datasets. We'll use SparkML to process the dataset (apply one or many feature transformers) and upload the transformed dataset to S3 so that it can be used for training with XGBoost.
	</p>		
	
   #### [Text classification (classification problem) using BlazingText with Amazon SageMaker, Spark Pipeline, and AWS Glue](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Text%20classification%20model%20using%20BlazingText%20with%20Amazon%20SageMaker%2C%20Spark%20Pipeline%2C%20and%20AWS%20Glue%20.ipynb) 
   <p>
    In this example, we will train the text classification model using SageMaker BlazingText algorithm on the DBPedia Ontology Dataset. Many of the training step resembles the 'Predict age of Alabone' notebook, so explanations are limited in this example, and we dive into code directly. The dataset is constructed by picking 14 unique classes. It has 560,000 training samples and 70,000 testing samples. The fields we used for this dataset contain title and abstract of each Wikipedia article.
	</p>		

   #### [Fraud Detection Modeling on Amazon SageMaker - Advanced](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Fraud%20Detection%20Modeling%20on%20Amazon%20SageMaker%20-%20Advanced.ipynb) 
   <p>
    We build a Fraud Dection model using a credit card usage data, provided by Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles). We go through full cycle of model building (data loading, cleaning, modeling, deploying, inference). Fraud Detection Using Machine Learning enables you to execute automated transaction processing on an example dataset or your own dataset. The includes ML model detects potentially fraudulent activity and flags that activity for review. 
	</p>  

   #### [Forecast with Amazon SageMaker DeepAR 1 (Speed-violation Prediction)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Forecast%20with%20Amazon%20SageMaker%20DeepAR%201%20(Speed-violation%20Prediction).ipynb) 
   <p>
    This notebook shows time series forecasting using the Amazon SageMaker DeepAR algorithm by analyzing city of Chicago’s Speed Camera Violation dataset. The data is provided by Chicago Gov Data Portal. We also go through full cycle of model building (data loading, cleaning, modeling, deploying, inference) in SageMaker using Python SDK. 
	</p>  

   #### [Forecast with Amazon SageMaker DeepAR 2 (energy consumption Prediction)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Forecast%20with%20Amazon%20SageMaker%20DeepAR%202%20(Speed-violation%20Prediction).ipynb) 
   <p>
    This notebook shows how to use DeepAR on SageMaker for predicting energy consumption of 370 customers over time. Particularly we will see the power of SageMaker where it trains a model, deploy it, and make requests to the deployed model to obtain forecasts interactively. Also, we will utilize DeepAR's advanced features like: missing values, additional time features, non-regular frequencies and category information. Same her, we go through full cycle of model building (data loading, cleaning, modeling, deploying, inference) in SageMaker using Python SDK. 
	</p>  

   #### [Forecast with Amazon SageMaker DeepAR 3 (Synthetic data)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Forecast%20with%20Amazon%20SageMaker%20DeepAR%203%20(Synthetic%20data).ipynb) 
   <p>
    Often times, you encounter more complex types of data when building a time-series model. This notebook demonstrates how to prepare a synthetic dataset of time series for training DeepAR and how to use the trained model for inference. DeepAR builds a single model for all time-series and tries to identify similarities across them. Intuitively, this sounds like a good idea for temperature time-series as we could expect them to exhibit similar patterns year after year. 
	</p>  
	

   #### [Image Classification on CIFAR-10 (transfer-learning)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Image%20Classification%20on%20CIFAR-10%20(transfer-learning)%20in%20Amazon%20SageMaker.ipynb) 
   <p>
    We use transfer learning mode in Sagemaker to fine-tune a pre-trained model (trained on imagenet data) to learn to classify a new dataset. The pre-trained model (ImageNet) will be fine-tuned using CIFAR-10 dataset. This covers a complete model-building cycle (dataloading, preprocessing, Hyperparmeter setting, Sagemaker training, Investigating loss, Testing, Model Deployment) using Amazon SageMaker.
	</p>
	

   #### [Object Detection using Amazon SageMaker (RecordIO format)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Object%20Detection%20using%20Amazon%20SageMaker%20(RecordIO%20format).ipynb) 
   <p>
    This notebook shows end-to-end example introducing the Amazon SageMaker Object Detection algorithm. In this demo, we will demonstrate how to train and to host an object detection model on the Pascal VOC dataset using the Single Shot multibox Detector (SSD) algorithm. In doing so, we will also demonstrate how to construct a training dataset using the RecordIO format as this is the format that the training job will consume. We go through full cycle of model building. 
	</p>  

   #### [Object Detection using Amazon SageMaker (Incremental Training)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Object%20Detection%20using%20Amazon%20SageMaker%20(Incremental%20Training).ipynb) 
   <p>
    In this example, we will show you how to train an object detector by re-using a model you previously trained in the SageMaker. With this model re-using ability, you can save the training time when you update the model with new data or improving the model quality with the same data. In the first half of this notebook (Intial Training), we will follow the training with RecordIO format example to train a object detection model on the Pascal VOC dataset. In the second half, we will show you how you can re-use the trained model and improve its quality without repeating the entire training process. 
	</p>  
	
   #### [Object Detection using Amazon SageMaker - Advanced (Transfer Learning with ResNet50 with SSD)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Object%20Detection%20using%20Amazon%20SageMaker%20-%20Advanced%20(Transfer%20Learning%20with%20ResNet50%20with%20SSD).ipynb) 
   <p>
    Introduction Object detection is the process of identifying and localizing objects in an image. A typical object detection solution takes an image as input and provides a bounding box on the image where an object of interest is found. It also identifies what type of object the box encapsulates. To create such a solution, we need to acquire and process a traning dataset, create and setup a training job for the alorithm so that it can learn about the dataset. Finally, we can then host the trained model in an endpoint, to which we can supply images. This notebook is an end-to-end example showing how the Amazon SageMaker Object Detection algorithm can be used with a publicly available dataset of bird images. Amazon SageMaker's object detection algorithm uses the Single Shot multibox Detector (SSD) algorithm, and this notebook uses a ResNet base network with that algorithm.
	</p> 	

   #### [Fine-tuning and Deplying PyTorch BERT model with Amazon Elastic Inference on AWS SageMaker](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Fine-tuning%20and%20Deplying%20PyTorch%20BERT%20model%20with%20Amazon%20Elastic%20Inference%20on%20AWS%20SageMaker.ipynb) 
   <p>
    Text classification is a technique for putting text into different categories, and has a wide range of applications. This post demonstrates how to use Amazon SageMaker to fine-tune a PyTorch BERT model and deploy it with Elastic Inference
	</p> 

   #### [Deploying a Full-Stack visual search application with SageMaker and Tensorflow](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Deploying%20a%20full-stack%20visual%20search%20application.ipynb) 
   <p>
    In this notebook, we'll build a model to classify the clothes that are similar to the input using a CNN and Elasticsearchon KNN algorithm. Visual image search is used in interfaces where instead of asking for something by voice or text, you show a photographic example of what you are looking for. With KNN, from each image you extract 2,048 feature vectors from a pre-trained Resnet50 model hosted in Amazon SageMaker. Each vector is stored to a KNN index in an Amazon ES domain. The following screenshot illustrates the workflow for creating KNN index.
	</p> 


   #### [US Census Segmentation using PCA and K-means clustering](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Data%20Segmentation%20using%20PCA%20and%20K-means%20clustering.ipynb) 
   <p>
    Typically for machine learning problems, clear use cases are derived from labelled data. For example, based on the attributes of a device, such as its age or model number, we can predict its likelihood of failure. We call this supervised learning because there is supervision or guidance towards predicting specific outcomes. However, in the real world, there are often large data sets where there is no particular outcome to predict, where clean labels are hard to define. It can be difficult to pinpoint exactly what the right outcome is to predict. This technique can be applied by businesses in customer or user segmentation to create targeted marketing campaigns. So this notebook demonstrates how we can access the underlying models that are built within Amazon SageMaker to extract useful model attributes.
	</p>
	
	
   #### [Bring Your Own Locally Trained ML/DL Model in SageMaker (XGBoost)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Bring%20Your%20Own%20Locally%20Trained%20ML%26DL%20Model%20in%20SageMaker%20(XGBoost).ipynb) 
   <p>
    This notebook shows how to train an Xgboost model in scikit-learn and then inject it into Amazon SageMaker's first party XGboost container for scoring. This addresses the usecase where a customer has already trained their model outside of Amazon SageMaker, but wishes to host it for predictions within Amazon SageMaker. Amazon SageMaker includes functionality to support a hosted notebook environment, distributed, serverless training, and real-time hosting. We think it works best when all three of these services are used together, but they can also be used independently. Some use cases may only require hosting. Maybe the model was trained prior to Amazon SageMaker existing, in a different service. Please note that scikit-learn XGBoost model is compatible with SageMaker XGBoost container, whereas other gradient boosted tree models (such as one trained in SparkML) are not.
	</p>	
	

   #### [Sagemaker Script mode (ML - XGboost)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Sagemaker%20Script%20mode%20usage%20(ML%20-%20XGboost).ipynb) 
   <p>
    Introduce a script-mode in Sagemaker where a user can bring their own Python file. Script mode is a very useful technique that lets you easily run your existing code in Amazon SageMaker, with very little change in codes. This gives more flexibility in the traning without having to worry about building containers or managing the underlying infrastructure. This time, we tackle the simple Deep Learning problem (MNIST) with Tensorflow
	</p>  
  
   #### [Sagemaker Script mode (DL - Tensorflow)](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Sagemaker%20Script%20mode%20usage%20(DL%20-%20Tensorflow).ipynb) 
   <p>
    The similar script mode codesets as above, but this time we look at the example of Deep Learning problem (MNIST) with Tensorflow.
	</p>   
	
   #### [Automating Machine Learning Workflows with Amazon Glue, Amazon SageMaker and AWS Step Functions](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/Automating%20Machine%20Learning%20Workflows%20with%20Amazon%20Glue%2C%20Amazon%20SageMaker%20and%20AWS%20Step%20Functions.ipynb) 
   <p>
    Automating machine learning workflows helps to build repeatable and reproducible machine learning models. It is a key step of in putting machine learning projects in production as we want to make sure our models are up-to-date and performant on new data. We will use the AWS services mentioned above to develop and automate a machine learning workflow with PySpark on AWS Glue for data preparation and processing, and Amazon SageMaker for model training and batch predictions. This will help any aspiring data scientists would acquire necessary skills to use produce a full ML workflow
	</p>  	
	
   #### [A/B Testing ML models in Production using AWS SageMaker](https://github.com/hyunjoonbok/amazon-sagemaker/blob/master/A%26B%20Testing%20ML%20models%20in%20Production%20using%20AWS%20SageMaker.ipynb) 
   <p>
    In this notebook, we will evaluate models by invoking specific variants of the model, and release a new model by specifying traffic distribution. In production ML workflows, we frequently have to our their models, such as by performing Automatic Model Tuning, training on additional or more-recent data, and improving feature selection. Performing A/B testing between a new model and an old model with production traffic can be an effective final step in the validation process for a new model. In A/B testing, you test different variants of your models and compare how each variant performs relative to each other. Amazon SageMaker enables you to test multiple models or model versions behind the same endpoint using production variants. Each production variant identifies a machine learning (ML) model and the resources deployed for hosting the model. You can distribute endpoint invocation requests across multiple production variants by providing the traffic distribution for each variant, or you can invoke a specific variant directly for each request.
	</p>  	
	

##### For more information on Script mode, plese refer to [AWS Blog](https://aws.amazon.com/blogs/machine-learning/using-tensorflow-eager-execution-with-amazon-sagemaker-script-mode/)


<hr>

## Contact
Created by [@hyunjoonbok](https://www.linkedin.com/in/hyunjoonbok/) - feel free to contact me!


## To-Do 
- Serverless SageMaker Training and Deployment Orchestration [github](https://github.com/aws-samples/serverless-sagemaker-orchestration)
- Build End-to-End Machine Learning (ML) Workflows with Amazon SageMaker and Apache Airflow [github](https://github.com/aws-samples/sagemaker-ml-workflow-with-apache-airflow), [AWS Blog](https://aws.amazon.com/blogs/machine-learning/build-end-to-end-machine-learning-workflows-with-amazon-sagemaker-and-apache-airflow/)
- Official AWS sagemaker [github](https://github.com/awslabs/amazon-sagemaker-examples)
- Official AWS sagemaker script mode [github](https://github.com/aws-samples/amazon-sagemaker-script-mode)


### References 
- Julien Simon's [Gitlab](https://gitlab.com/juliensimon/dlnotebooks/-/tree/master/sagemaker) 
- [Boto3 setup](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/sqs.html) 
- Official amazon-sagemaker-examples [Github](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/README.md)
- SageMaker with Airflow [Github](https://github.com/aws-samples/sagemaker-ml-workflow-with-apache-airflow)
- AWS Machine Learning Blog [Link](https://aws.amazon.com/ko/blogs/machine-learning/page/3/)
