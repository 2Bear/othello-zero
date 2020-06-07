#!/bin/sh
set -eu

CONTAINER_NAME=othello_zero

if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    docker container rm $CONTAINER_NAME >/dev/null 2>/dev/null && echo "Removed previous container $CONTAINER_NAME" || true
    docker run --name $CONTAINER_NAME --network host --gpus all -u $(id -u):$(id -g) -v $(pwd):/home/$(whoami)/$(basename $(pwd)) -it tf
else
    docker exec -it $CONTAINER_NAME bash
fi
