FROM tensorflow/tensorflow:latest-gpu as base

ARG USER=
ARG UID=
ARG GID=

RUN addgroup --gid $GID $USER
RUN adduser --disabled-password --gecos '' --uid $UID --gid $GID $USER
RUN usermod -aG sudo $USER

RUN apt update && apt install -y --no-install-recommends apt-utils
RUN apt update && apt install -y --no-install-recommends locales

RUN locale-gen en_US.UTF-8

RUN apt update && apt install -y --no-install-recommends \
      git \
      sudo \
      ack-grep \
      vim \
      curl

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ${UID}:${GID}
WORKDIR /home/$USER
