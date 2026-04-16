#Dockerfile for model training (the same for all the models trained)
FROM python:3.13-slim  

# Install curl pour uv
RUN apt-get update && apt-get install -y curl --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* 

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

#Set environnement
COPY pyproject.toml .
COPY uv.lock .          
RUN uv sync --frozen 

COPY Config.py .
COPY src/ ./src/ 
COPY data_external/revenu_median_communes.csv .
