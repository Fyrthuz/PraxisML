variable "role_name" {
  type    = string
  default = "praxisml-tenant-runtime-role"
}

variable "bucket_arns" {
  type = list(string)
}

data "aws_iam_policy_document" "assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::000000000000:root"]
    }
  }
}

resource "aws_iam_role" "tenant_runtime" {
  name               = var.role_name
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

data "aws_iam_policy_document" "s3_access" {
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]
    resources = flatten([
      for arn in var.bucket_arns : [arn, "${arn}/*"]
    ])
  }
}

resource "aws_iam_role_policy" "tenant_s3" {
  name   = "${var.role_name}-s3"
  role   = aws_iam_role.tenant_runtime.id
  policy = data.aws_iam_policy_document.s3_access.json
}

output "role_arn" {
  value = aws_iam_role.tenant_runtime.arn
}
