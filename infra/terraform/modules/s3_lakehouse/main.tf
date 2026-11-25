variable "bucket_prefix" {
  type        = string
  description = "Prefijo para los buckets S3 de PraxisML."
}

variable "enable_versioning" {
  type    = bool
  default = true
}

locals {
  buckets = {
    artifacts = "${var.bucket_prefix}-artifacts"
    dvc       = "${var.bucket_prefix}-dvc"
    lakehouse = "${var.bucket_prefix}-lakehouse"
  }
}

resource "aws_s3_bucket" "this" {
  for_each      = local.buckets
  bucket        = each.value
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "this" {
  for_each = var.enable_versioning ? aws_s3_bucket.this : {}
  bucket   = each.value.id

  versioning_configuration {
    status = "Enabled"
  }
}

output "bucket_names" {
  value = { for k, b in aws_s3_bucket.this : k => b.bucket }
}
