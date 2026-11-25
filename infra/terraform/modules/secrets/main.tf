variable "secret_name" {
  type    = string
  default = "praxisml/local/app-secrets"
}

variable "secret_payload" {
  type      = map(string)
  sensitive = true
}

resource "aws_secretsmanager_secret" "app" {
  name                    = var.secret_name
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "app" {
  secret_id     = aws_secretsmanager_secret.app.id
  secret_string = jsonencode(var.secret_payload)
}

output "secret_arn" {
  value = aws_secretsmanager_secret.app.arn
}
