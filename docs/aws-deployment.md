# Deploying NEXUS AI on AWS

Reference architecture:

```
                ┌─────────────────────┐
  Users ──────▶ │ Application Load    │
                │ Balancer (HTTPS)    │
                └─────────┬───────────┘
                          │
                ┌─────────▼───────────┐       ┌──────────────────┐
                │ ECS Fargate service │──────▶│ S3 bucket        │  original documents
                │ (nexus-ai container)│       │ (SSE encrypted)  │
                └───┬─────────────┬───┘       └──────────────────┘
                    │             │
        ┌───────────▼───┐   ┌─────▼──────────────┐
        │ RDS PostgreSQL│   │ Secrets Manager    │  LLM API key, DB password
        └───────────────┘   └────────────────────┘
                    │
        ┌───────────▼─────────────┐
        │ MLflow server (EC2/ECS) │  experiment tracking, model artifacts in S3
        └─────────────────────────┘
```

## 1. Build and push the image to ECR

```bash
AWS_REGION=ap-south-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/nexus-ai

aws ecr create-repository --repository-name nexus-ai --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $REPO
docker build -t $REPO:latest .
docker push $REPO:latest
```

## 2. Create the data layer

```bash
# Document bucket (private, encrypted)
aws s3api create-bucket --bucket nexus-ai-documents-$ACCOUNT --region $AWS_REGION \
  --create-bucket-configuration LocationConstraint=$AWS_REGION
aws s3api put-public-access-block --bucket nexus-ai-documents-$ACCOUNT \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# PostgreSQL (use a private subnet group in production)
aws rds create-db-instance --db-instance-identifier nexus-db --engine postgres --engine-version 16 \
  --db-instance-class db.t4g.micro --allocated-storage 20 \
  --master-username nexus --manage-master-user-password --db-name nexus

# Secrets
aws secretsmanager create-secret --name nexus/anthropic-api-key --secret-string "sk-ant-..."
```

## 3. IAM task role

The ECS task role needs only:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::nexus-ai-documents-ACCOUNT/documents/*" }
  ]
}
```

The task *execution* role needs `secretsmanager:GetSecretValue` on the `nexus/*` secrets.

## 4. ECS Fargate service

Use `deploy/ecs-task-definition.json` (replace the `ACCOUNT`, `REGION` and ARN placeholders), then:

```bash
aws ecs register-task-definition --cli-input-json file://deploy/ecs-task-definition.json
aws ecs create-cluster --cluster-name nexus
aws ecs create-service --cluster nexus --service-name nexus-ai --task-definition nexus-ai \
  --desired-count 2 --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=DISABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=nexus-ai,containerPort=5000"
```

Point the ALB health check at `/api/health`.

## 5. MLflow on AWS

Run an MLflow server with PostgreSQL as the backend store and S3 for artifacts:

```bash
mlflow server --host 0.0.0.0 --port 5001 \
  --backend-store-uri postgresql+psycopg://nexus:PASSWORD@nexus-db.xxxx.rds.amazonaws.com/mlflow \
  --artifacts-destination s3://nexus-ai-documents-ACCOUNT/mlflow
```

Then train against it from anywhere:

```bash
MLFLOW_TRACKING_URI=http://<mlflow-host>:5001 python -m training.train_classifier
```

## Environment variables for the container

| Variable | Example |
|---|---|
| `DATABASE_URL` | `postgresql+psycopg://nexus:***@nexus-db.xxxx.rds.amazonaws.com:5432/nexus` |
| `STORAGE_BACKEND` | `s3` |
| `S3_BUCKET` | `nexus-ai-documents-123456789012` |
| `AWS_REGION` | `ap-south-1` |
| `LLM_PROVIDER` / `LLM_MODEL` | `anthropic` / `claude-sonnet-5-5` |
| `ANTHROPIC_API_KEY` | from Secrets Manager |
| `NEXUS_API_KEY` | from Secrets Manager (protects the API) |
| `MLFLOW_TRACKING_URI` | `http://mlflow.internal:5001` |
