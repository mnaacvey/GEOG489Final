# Base image. Slim to keep size down. Python 3.12 matches our pyproject.toml.
FROM python:3.12-slim

# Install the package and its Python dependencies.
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Create directories the user will mount their input and output into.
RUN mkdir -p /data /reports

# Default command. Users override the args at runtime.
ENTRYPOINT ["embedding-recommender"]
CMD ["--help"]
