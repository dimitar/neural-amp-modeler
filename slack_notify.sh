#!/usr/bin/env bash
# Send a message to Slack via incoming webhook
# Usage: bash slack_notify.sh "message text"
curl -s -X POST -H 'Content-type: application/json' \
  --data "{\"text\": \"$1\"}" \
  "${SLACK_WEBHOOK_URL}" > /dev/null
