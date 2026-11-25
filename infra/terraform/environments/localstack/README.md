# Local AWS Sandbox (Terraform + LocalStack Community)

Aprovisiona localmente a coste `$0` los recursos AWS de PraxisML sobre **LocalStack Community** (`http://localhost:4566`):
- Buckets S3 (`artifacts`, `dvc`, `lakehouse`) con versionado
- Tabla DynamoDB de lock para Delta Lake
- Rol IAM de runtime para tenants (`STS AssumeRole`)
- Secretos de runtime en Secrets Manager
- Colas SQS (+ DLQ) y tópico SNS de alertas

## Uso rápido

```bash
# 1. Levantar LocalStack Community (perfil aws-local)
docker compose --profile aws-local up -d localstack

# 2. Inicializar y aplicar Terraform contra LocalStack
terraform -chdir=infra/terraform/environments/localstack init
terraform -chdir=infra/terraform/environments/localstack apply -auto-approve

# 3. Configurar backend para apuntar a LocalStack S3
export STORAGE_BACKEND=s3
export AWS_ENDPOINT_URL_S3=http://localhost:4566
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export S3_BUCKET_NAME=praxisml-local-artifacts
```
