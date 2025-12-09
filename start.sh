#!/bin/bash

GIT_COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
TENSORBOARD_START_PORT=6006
TENSORBOARD_END_PORT=6015
export GIT_COMMIT_HASH
export TENSORBOARD_START_PORT
export TENSORBOARD_END_PORT

docker compose -f docker-compose.yaml up -d --build
