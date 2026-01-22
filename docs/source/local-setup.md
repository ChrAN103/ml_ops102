# Running Locally

## Prerequisites

- **Python**: 3.13 or compatible version
- **uv**: Fast Python package installer and runner
- **Git**: For version control
- **DVC**: For data version control (optional, for pulling data)

## Initial Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ChrAN103/ml_ops102.git
cd ml_ops102
```

### 2. Install Dependencies

The project uses `uv` for fast dependency management:

```bash
uv sync
```

This installs all dependencies specified in `pyproject.toml`.

### 3. Pull Data (Optional)

If using DVC-tracked data:

```bash
dvc pull
```

This downloads the raw dataset and any tracked models.

## Running the Pipeline

### Data Preprocessing

Convert raw CSV data to PyTorch tensors:

```bash
uv run invoke preprocess-data
```

Output files will be created in `data/processed/`:

- `train.pt` - Training set
- `val.pt` - Validation set
- `test.pt` - Test set

### Training

Train the model with default configuration:

```bash
uv run invoke train
```

Or run directly with custom parameters:

```bash
uv run python src/mlops_project/train.py training.epochs=10 training.learning_rate=0.001
```

Training outputs:

- **Models**: Saved to `models/` directory
- **Logs**: Saved to `logs/training/version_X/`
- **Outputs**: Per-run logs in `outputs/YYYY-MM-DD_HH-MM-SS/`

### Evaluation

Evaluate a trained model:

```bash
uv run invoke evaluate
```

### Running the API

Start the FastAPI service:

```bash
uv run python src/mlops_project/api.py
```

The API will be available at `http://localhost:8000`.

#### API Endpoints

- `POST /predict` - Predict if news is fake or real
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics

#### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"title": "Breaking News", "text": "This is a news article..."}'
```

## Development Tools

### Code Formatting

Format code with ruff:

```bash
uv run ruff format .
```

### Linting

Check and fix linting issues:

```bash
uv run ruff check . --fix
```

### Testing

Run all tests with coverage:

```bash
uv run invoke test
```

Run specific test files:

```bash
uv run pytest tests/test_data.py
uv run pytest tests/test_model.py
```

Run integration tests:

```bash
uv run pytest tests/integrationtests/
```

### Pre-commit Hooks

Setup pre-commit hooks:

```bash
uv run pre-commit install
```

Run hooks manually:

```bash
uv run pre-commit run --all-files
```

## Docker Development

### Building Images

Build all Docker images:

```bash
uv run invoke docker-build
```

Or build individually:

```bash
docker build -t train:latest -f dockerfiles/train.dockerfile .
docker build -t api:latest -f dockerfiles/api.dockerfile .
```

### Running Containers

#### Training Container

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models train:latest
```

#### API Container

```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models api:latest
```

## Configuration

### Main Configuration

Edit `configs/config.yaml` to change:

- Model hyperparameters
- Training settings
- Data paths
- Logging configuration

### Environment-Specific Configs

- `config_cpu.yaml` - CPU-optimized settings
- `config_gpu.yaml` - GPU-optimized settings
- `config_cloud.yaml` - Cloud deployment settings

### Using Different Configs

```bash
uv run python src/mlops_project/train.py --config-name=config_gpu
```

## Documentation

### Building Documentation

Build the documentation site:

```bash
uv run invoke build-docs
```

Output is generated in `build/` directory.

### Serving Documentation

Serve docs locally with live reload:

```bash
uv run invoke serve-docs
```

Visit `http://127.0.0.1:8000` to view the documentation.

## Troubleshooting

### Common Issues

**Import errors**: Ensure you're using `uv run` prefix for Python commands

**Missing data**: Run `dvc pull` or manually place data in `data/raw/`

**CUDA errors**: Switch to CPU config with `--config-name=config_cpu`

**Port conflicts**: Change API port with `--port` flag

### Getting Help

Check available invoke tasks:

```bash
uv run invoke --list
```

View task help:

```bash
uv run invoke --help <task-name>
```
