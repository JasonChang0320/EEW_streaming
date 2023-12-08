FROM ubuntu:latest

# install OpenJDK 17
RUN \
  apt-get update && \
  apt-get install -y openjdk-17-jdk && \
  apt-get install -y git && \
  rm -rf /var/lib/apt/lists/*

# install Python
RUN \
    apt-get update && \
    apt-get install -y python3 python3-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Python package install
COPY requirements.txt /tmp/
RUN python3 -m pip install --upgrade pip;
RUN python3 -m pip --no-cache-dir install --requirement /tmp/requirements.txt;

# Define working directory
CMD ["bash"]