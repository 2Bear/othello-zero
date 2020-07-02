#!/bin/sh
set -eu

docker build --network=host --build-arg USER=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) -f Dockerfile -t tf .
