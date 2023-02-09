FROM python:3.11

# Install build dependencies
RUN pip install --no-cache-dir poetry

# Change user to root
USER root

# Set working directory
WORKDIR /code

# Copy project files
COPY . /code

# Install dependencies
RUN poetry install --only main --no-root

CMD [ "poetry", "run", "test" ]
