terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "localstack_endpoint" {
  type    = string
  default = "http://localhost:4566"
}

variable "bucket_prefix" {
  type    = string
  default = "praxisml-local"
}

provider "aws" {
  region                      = var.aws_region
  access_key                  = "test"
  secret_key                  = "test"
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  s3_use_path_style           = true

  endpoints {
    s3             = var.localstack_endpoint
    dynamodb       = var.localstack_endpoint
    iam            = var.localstack_endpoint
    sts            = var.localstack_endpoint
    sqs            = var.localstack_endpoint
    sns            = var.localstack_endpoint
    secretsmanager = var.localstack_endpoint
  }
}

module "s3_lakehouse" {
  source        = "../../modules/s3_lakehouse"
  bucket_prefix = var.bucket_prefix
}

module "delta_lock" {
  source     = "../../modules/delta_lock_dynamodb"
  table_name = "${var.bucket_prefix}-delta-lock"
}

module "iam_tenant_roles" {
  source = "../../modules/iam_tenant_roles"
  bucket_arns = [
    for name in values(module.s3_lakehouse.bucket_names) : "arn:aws:s3:::${name}"
  ]
}

module "secrets" {
  source      = "../../modules/secrets"
  secret_name = "praxisml/localstack/runtime"
  secret_payload = {
    STORAGE_BACKEND      = "s3"
    AWS_ENDPOINT_URL_S3  = var.localstack_endpoint
    S3_BUCKET_NAME       = module.s3_lakehouse.bucket_names["artifacts"]
    DVC_S3_BUCKET        = module.s3_lakehouse.bucket_names["dvc"]
    LAKEHOUSE_S3_BUCKET  = module.s3_lakehouse.bucket_names["lakehouse"]
  }
}

module "messaging" {
  source       = "../../modules/messaging"
  queue_prefix = var.bucket_prefix
}

output "s3_buckets" {
  value = module.s3_lakehouse.bucket_names
}

output "delta_lock_table" {
  value = module.delta_lock.table_name
}

output "tenant_role_arn" {
  value = module.iam_tenant_roles.role_arn
}

output "events_queue_url" {
  value = module.messaging.queue_url
}

output "alerts_topic_arn" {
  value = module.messaging.alerts_topic_arn
}
