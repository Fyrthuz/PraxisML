variable "queue_prefix" {
  type    = string
  default = "praxisml"
}

resource "aws_sqs_queue" "domain_events_dlq" {
  name = "${var.queue_prefix}-domain-events-dlq"
}

resource "aws_sqs_queue" "domain_events" {
  name = "${var.queue_prefix}-domain-events"
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.domain_events_dlq.arn
    maxReceiveCount     = 3
  })
}

resource "aws_sns_topic" "alerts" {
  name = "${var.queue_prefix}-alerts"
}

output "queue_url" {
  value = aws_sqs_queue.domain_events.url
}

output "alerts_topic_arn" {
  value = aws_sns_topic.alerts.arn
}
