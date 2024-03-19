# Start from the base image
FROM mcr.microsoft.com/devcontainers/base:jammy

# Update the package lists
RUN apt-get update

# Install Python and pip
RUN apt-get install -y python3 python3-pip

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -