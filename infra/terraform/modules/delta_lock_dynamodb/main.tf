variable "table_name" {
  type        = string
  default     = "praxisml-delta-log-lock"
  description = "Tabla DynamoDB usada por Delta Lake para concurrencia ACID sobre S3."
}

resource "aws_dynamodb_table" "delta_lock" {
  name         = var.table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "tablePath"
  range_key    = "fileName"

  attribute {
    name = "tablePath"
    type = "S"
  }

  attribute {
    name = "fileName"
    type = "S"
  }
}

output "table_name" {
  value = aws_dynamodb_table.delta_lock.name
}
