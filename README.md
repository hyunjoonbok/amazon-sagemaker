# amazon-sagemaker



# Set-up
Before you can begin using Boto3, you should set up authentication credentials. Credentials for your AWS account can be found in the IAM Console. You can create or use an existing user. Go to manage access keys and generate a new set of keys.

If you have the AWS CLI installed, then you can use it to configure your credentials file:
```
aws configure
```

Alternatively, you can create the credential file yourself. By default, its location is at ~/.aws/credentials:

[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
You may also want to set a default region. This can be done in the configuration file. By default, its location is at ~/.aws/config:

[default]
region=us-east-1


For more information on how to create a bucket and perfomr initial AWS set-up, 
please refer to https://docs.aws.amazon.com/AmazonS3/latest/gsg/GetStartedWithS3.html

### Install Boto3

```
pip install boto3
```

### Install Sagemaker python-sdk
```
pip install sagemaker
```
From PyPI
```
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```
From Source

For detailed information, please refer to [official github](https://github.com/aws/sagemaker-python-sdk)


> Example codes are taken often times from Julien Simon's [Gitlab](https://gitlab.com/juliensimon/dlnotebooks/-/tree/master/sagemaker)
> More examples are in 'https://boto3.amazonaws.com/v1/documentation/api/latest/guide/sqs.html'
