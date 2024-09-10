# LightGBM model using SageMaker Inference

This repository is to demo how to use SageMaker Inference to host LightGBM models. We shall build a custom inference container image and use it to host models. Especially, we are using [multi-model-server](https://github.com/awslabs/multi-model-server/tree/master) to host the model. There is an alternative option to use FastAPI (refer to [FastAPI exampl](https://testdriven.io/blog/fastapi-machine-learning/)).

## Steps

### Build a docker image and may push it to a ECR repo

```shell
cd docker/mms

export REGION_CODE=us-east-1
export ACCOUNT_ID=##...#

# build the docker image
docker build -t lightgbm-breat-cancer-classifier .

# authentication and login to ECR
aws ecr get-login-password --region $REGION_CODE | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION_CODE.amazonaws.com

# tagging
docker tag lightgbm-breat-cancer-classifier:latest $ACCOUNT_ID.dkr.ecr.$REGION_CODE.amazonaws.com/lightgbm-breat-cancer-classifier:latest

# pushing
docker push $ACCOUNT_ID.dkr.ecr.$REGION_CODE.amazonaws.com/lightgbm-breat-cancer-classifier:latest
```

### Configure JupyterNotebook runtime

Install dependencies:

```shell
pip install -r requirements.txt
```

Provide below properties in `.env` under project folder. 

```shell
AWS_PROFILE=
AWS_DEFAULT_REGION=
IAM_ROLE=arn:aws:iam::{ACCOUNT_ID}:role/service-role/{SAGEMAKER_ROLE_FOR_HOSTING}
INFERENCE_IMAGE_URI={ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/lightgbm-breat-cancer-classifier:latest
```

### Execute the model training, hosting and evaluation

Run through JupyterNotebook [Hosting LightGBM Model using SageMaker Inference.ipynb](./Hosting%20LightGBM%20Model%20using%20SageMaker%20Inference.ipynb) on a runtime with IAM permission that can access SageMaker default S3 bucket (or custom bucket), and using SageMaker Inference.