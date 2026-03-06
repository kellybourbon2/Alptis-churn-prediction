# BDC-Alptis
Project part of Ensae "Business Data Challenge" 2025-2026

# How to start:
Clone the repo and follow the following steps:

## Environement creation and activation
```bash
uv sync
source ./venv/bin/activate
```
## Data Loading

Two loading modes are available depending on the variable `LOAD_FROM_S3` in `Config`

### Local Loading

1. Create a folder matching the `DATA_RAW_DIR` variable defined in `Config`
2. Organize your data files following this structure:
```
data/
├── training/
│   └── *.csv
├── validation/
│   └── *.csv
└── test/
    └── *.csv
```

### SSPCloud Loading (S3 Loading)

Data is loaded directly from the shared MinIO bucket on SSPCloud.

Create a `.env` file at the root of the project with the following variables:
```ini
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...
AWS_S3_ENDPOINT=...
AWS_BUCKET_NAME=...
```
>  These credentials can be found in your SSPCloud account under **My Account → Storage Connection**. Note that `AWS_SESSION_TOKEN` expires periodically and must be refreshed.