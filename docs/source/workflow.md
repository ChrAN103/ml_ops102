# Development Workflow

## Project Structure

The project follows a standard MLOps structure with clear separation of concerns:

```
ml_ops102/
├── src/mlops_project/     # Source code
│   ├── data.py            # Data preprocessing
│   ├── train.py           # Model training
│   ├── evaluate.py        # Model evaluation
│   ├── models.py          # Model architecture
│   ├── api.py             # FastAPI service
│   └── visualize.py       # Visualization utilities
├── tests/                 # Unit and integration tests
├── configs/               # Configuration files
├── data/                  # Data storage (raw/processed)
├── models/                # Trained model checkpoints
├── dockerfiles/           # Docker configurations
└── .github/workflows/     # CI/CD pipelines
```

## Development Lifecycle

### 1. Data Preprocessing

Process raw news data into PyTorch tensors:

```bash
uv run invoke preprocess-data
```

This reads from `data/raw/News.csv` and creates train/val/test splits in `data/processed/`.

### 2. Model Training

Train the LSTM model with configurable hyperparameters:

```bash
uv run invoke train
```

Training uses:

- **Hydra** for configuration management (`configs/config.yaml`)
- **PyTorch Lightning** for training orchestration
- **Weights & Biases** for experiment tracking
- **Lightning checkpointing** for model saving

Checkpoints are saved to `models/` and logs to `logs/training/`.

### 3. Model Evaluation

Evaluate trained models on the test set:

```bash
uv run invoke evaluate
```

Generates metrics and saves results to the reports directory.

### 4. Testing

Run the full test suite with coverage:

```bash
uv run invoke test
```

Test categories:

- **Unit tests**: `tests/test_data.py`, `tests/test_model.py`
- **Integration tests**: `tests/integrationtests/test_api.py`
- **Performance tests**: `tests/performancetests/locustfile.py`

### 5. Code Quality

The project uses pre-commit hooks for code quality:

```bash
uv run ruff format .
uv run ruff check . --fix
uv run pre-commit run --all-files
```

## Configuration Management

Hydra manages configurations in the `configs/` directory:

* `config.yaml` - Main configuration
* `config_cloud.yaml` - Cloud-specific settings
* `config_cpu.yaml` / `config_gpu.yaml` - Device-specific configs

Override parameters via command line:

```bash
uv run python src/mlops_project/train.py training.epochs=20 training.batch_size=64
```

## Version Control

### Data Versioning

DVC tracks large files:

```bash
dvc pull
dvc push
```

Tracked files:

- `data/raw/News.csv.dvc`
- `models/model.pt.dvc`

### Code Versioning

Git tracks code with automated workflows:

- **CI**: Linting and testing on push
- **CD**: Docker builds and cloud deployment on merge

## Task Management

Use invoke for common tasks:

```bash
uv run invoke --list
```

Available tasks:

- `preprocess-data` - Preprocess raw data
- `train` - Train model
- `evaluate` - Evaluate model
- `test` - Run tests
- `docker-build` - Build Docker images
- `build-docs` - Build documentation
- `serve-docs` - Serve documentation locally

## Monitoring & Logging

### Training Monitoring

Experiments are tracked with:

- **Weights & Biases**: Real-time metrics and visualizations
- **PyTorch Lightning logs**: Local CSV logs in `logs/training/`
- **Hydra outputs**: Per-run outputs in `outputs/YYYY-MM-DD_HH-MM-SS/`

### API Monitoring

The FastAPI service includes:

- **Prometheus metrics**: Endpoint `/metrics`
- **Health checks**: Endpoint `/health`
- **Request logging**: Structured JSON logs

## Docker Workflow

Build containers locally:

```bash
uv run invoke docker-build
```

Run training in Docker:

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models train:latest
```

Run API in Docker:

```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models api:latest
```

## Continuous Integration

GitHub Actions automatically:

1. Run linting on every push
2. Run tests on every push
3. Build Docker images on merge to main
4. Deploy to GCP Cloud Run on merge to main

View workflow status in repository badges.
