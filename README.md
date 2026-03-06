# BDC-Alptis
Project part of Ensae "Business Data Challenge" 2025-2026

# How to start:
Clone the repo and follow the following steps:
## Environement creation and activation
```bash
uv sync
source ./venv/bin/activate
```

## Data loading:
local or S3 Storage loading based on the variable "LOAD_FROM_S3" in Config

- To do a local loading :
Put all the data in a file named as the variable "RAW_DATA_DIR" in Config

- To do a SSP cloud loading: 
Create an `.env` file that contains the following variables : AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_S3_ENDPOINT, AWS_BUCKET_NAME, those variables are used to load the data.