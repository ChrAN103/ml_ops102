# Missing Components from ml_ops102 Project

This document lists all missing components from your MLOps project compared to the course requirements.

## ✅ What You Have (Good Progress!)

- ✅ Git repository with proper structure
- ✅ CookieCutter project template structure
- ✅ PyTorch Lightning implementation (`LightningModule`, `LightningDataModule`)
- ✅ Dockerfiles (train, api, evaluate)
- ✅ Unit tests (test_data.py, test_model.py, test_api.py - though api.py is empty)
- ✅ Basic CI/CD (tests.yaml, linting.yaml)
- ✅ Pre-commit hooks (basic setup)
- ✅ DVC setup (dvc-gdrive, dvc-gs)
- ✅ Documentation structure (MkDocs)
- ✅ Typer CLI integration
- ✅ Invoke tasks

---

## ❌ Missing Components

### Week 1 Requirements

#### 1. **Hydra Configuration Management** ❌
- **Status**: `configs/` directory exists but is empty
- **Missing**:
  - Hydra configuration files (YAML)
  - `@hydra.main()` decorator in training script
  - Dynamic parameter management
  - Configuration composition
- **Action**: 
  - Add `hydra-core` to dependencies (already in uv.lock but not in pyproject.toml)
  - Create `configs/config.yaml` with hyperparameters
  - Refactor `train.py` to use Hydra
  - Use `hydra.utils.instantiate` for optimizer/other objects

#### 2. **Profiling** ❌
- **Missing**:
  - `cProfile` integration for performance analysis
  - `snakeviz` for visualization
  - PyTorch profiler integration
  - Performance optimization based on profiling results
- **Action**:
  - Add profiling to training/data loading pipeline
  - Use `cProfile` and `snakeviz` to identify bottlenecks
  - Add `torch-tb-profiler` for PyTorch profiling

#### 3. **Logging** ⚠️ (Partial)
- **Status**: Using PyTorch Lightning's `self.log()` but no application-level logging
- **Missing**:
  - `loguru` or Python `logging` module integration
  - Log file rotation
  - Structured logging for application events
- **Action**:
  - Replace `print()` statements with proper logging
  - Add `loguru` to dependencies
  - Configure log levels and file rotation

#### 4. **Weights & Biases (Wandb)** ❌
- **Status**: Using CSVLogger instead of Wandb
- **Missing**:
  - `wandb` package (not in dependencies)
  - `wandb.init()` and `wandb.log()` integration
  - Wandb logger in PyTorch Lightning Trainer
  - Artifact logging (models, data)
  - Hyperparameter sweeps
  - Model registry integration
- **Action**:
  - Add `wandb` to dependencies
  - Replace `CSVLogger` with `WandbLogger`
  - Log metrics, images, histograms
  - Set up model registry with aliases (`staging`, `production`)

#### 5. **Pre-commit Hooks Enhancement** ⚠️ (Basic)
- **Status**: Only basic hooks (trailing whitespace, YAML check)
- **Missing**:
  - `ruff` pre-commit hook
  - `mypy` pre-commit hook (if using type checking)
- **Action**:
  - Add `ruff-pre-commit` to `.pre-commit-config.yaml`
  - Run `pre-commit install`

---

### Week 2 Requirements

#### 6. **CI/CD Enhancements** ⚠️ (Partial)
- **Status**: Basic tests and linting workflows exist
- **Missing**:
  - Caching in GitHub Actions (pip/uv cache)
  - Multi-Python version testing (only 3.13)
  - Multi-PyTorch version testing
  - DVC integration in CI (for data tests)
  - GCP authentication in CI
- **Action**:
  - Add caching to workflows
  - Add matrix strategy for Python versions
  - Add DVC setup step in CI
  - Add GCP authentication if needed

#### 7. **CML Workflows** ❌
- **Missing**:
  - Data-triggered workflow (when `data/` changes)
  - Model registry-triggered workflow (when Wandb model gets `staging` alias)
  - `cml` framework integration
  - Performance tests on staged models
- **Action**:
  - Create `.github/workflows/cml_data.yaml` triggered by `data/` changes
  - Create `.github/workflows/cml_model.yaml` triggered by Wandb webhooks
  - Use `cml comment create` for PR comments

#### 8. **GCP Integration** ❌
- **Missing**:
  - GCP Cloud Storage bucket setup
  - DVC remote configured to GCP bucket
  - Cloud Build configuration (`cloudbuild.yaml`)
  - Cloud Build trigger workflow
  - Artifact Registry setup
  - Vertex AI or Compute Engine training jobs
- **Action**:
  - Create GCP bucket and configure DVC remote
  - Create `cloudbuild.yaml` for Docker image builds
  - Set up Cloud Build trigger
  - Create Vertex AI custom job configuration

#### 9. **FastAPI Implementation** ❌
- **Status**: `api.py` file exists but is empty
- **Missing**:
  - FastAPI app with endpoints
  - Model loading and inference logic
  - Request/response models (Pydantic)
  - Error handling
  - Health check endpoint
- **Action**:
  - Implement FastAPI app in `api.py`
  - Add `/predict` endpoint
  - Add `/health` endpoint
  - Containerize and test locally

#### 10. **Cloud Deployment** ❌
- **Missing**:
  - Cloud Run deployment configuration
  - Cloud Functions deployment (if using serverless)
  - Environment variables/secrets management
  - Volume mounts for model storage
- **Action**:
  - Deploy FastAPI to Cloud Run
  - Configure environment variables
  - Set up model storage (Cloud Storage or Artifact Registry)

#### 11. **API Testing** ❌
- **Status**: `test_api.py` exists but is empty (API doesn't exist yet)
- **Missing**:
  - `httpx` or `TestClient` tests
  - Integration tests for API endpoints
  - CI workflow for API tests
- **Action**:
  - Write API tests using `fastapi.testclient.TestClient`
  - Add API test workflow to CI

#### 12. **Load Testing** ❌
- **Missing**:
  - `locust` package (not in dependencies)
  - `locustfile.py` with load test scenarios
  - Load testing in CI/CD
- **Action**:
  - Add `locust` to dependencies
  - Create `locustfile.py`
  - Add load testing step to CI (test deployed API)

#### 13. **ONNX or BentoML Deployment** ❌
- **Missing**:
  - ONNX model export
  - `onnxruntime` integration (not in dependencies)
  - ONNX inference endpoint
  - OR BentoML service implementation
- **Action**:
  - Add `onnxruntime` to dependencies
  - Export PyTorch model to ONNX format
  - Create ONNX inference endpoint
  - Benchmark PyTorch vs ONNX inference speed

#### 14. **Frontend** ❌
- **Missing**:
  - `streamlit` package (not in dependencies)
  - Streamlit app for user interface
  - Integration with deployed API
- **Action**:
  - Add `streamlit` to dependencies
  - Create Streamlit frontend
  - Deploy to Cloud Run
  - Connect to backend API

---

### Week 3 Requirements

#### 15. **Data Drift Detection** ❌
- **Missing**:
  - `evidently` package (not in dependencies)
  - Data drift detection implementation
  - Drift detection API endpoint
  - Data collection from deployed application
- **Action**:
  - Add `evidently` to dependencies
  - Implement drift detection using Evidently
  - Deploy drift detection API to Cloud Run
  - Set up data collection pipeline

#### 16. **System Monitoring** ❌
- **Missing**:
  - `prometheus-fastapi-instrumentator` (not in dependencies)
  - Prometheus metrics (Counter, Gauge, Histogram, Summary)
  - `/metrics` endpoint
  - Cloud Monitoring integration
  - Alert policies
- **Action**:
  - Add `prometheus-fastapi-instrumentator` to dependencies
  - Instrument API with Prometheus metrics
  - Set up Cloud Monitoring
  - Create alert policies in GCP

#### 17. **Model Optimization** ❌
- **Missing**:
  - Quantization implementation
  - Pruning implementation
  - Knowledge distillation (optional)
  - Performance benchmarking
- **Action**:
  - Implement PyTorch quantization
  - Implement pruning
  - Benchmark optimized models
  - Compare inference speed and model size

#### 18. **Distributed Training** ⚠️ (Optional but recommended)
- **Status**: PyTorch Lightning supports it, but not configured
- **Missing**:
  - Multi-GPU training setup
  - DDP (Distributed Data Parallel) configuration
- **Action**:
  - Configure `accelerator="gpu"` and `devices` for multi-GPU
  - Test distributed training if GPUs available

#### 19. **Distributed Data Loading** ⚠️ (Can be optimized)
- **Status**: Basic DataLoader exists
- **Missing**:
  - Optimization of `num_workers`
  - `pin_memory` configuration
  - Performance analysis
- **Action**:
  - Experiment with `num_workers` parameter
  - Add `pin_memory=True` for GPU training
  - Profile data loading performance

---

### Extra Requirements

#### 20. **Documentation Deployment** ⚠️ (Structure exists, not deployed)
- **Status**: MkDocs structure exists
- **Missing**:
  - GitHub Pages deployment workflow
  - Auto-deployment on push to main
- **Action**:
  - Create `.github/workflows/deploy_docs.yaml`
  - Configure GitHub Pages
  - Deploy documentation

#### 21. **Architectural Diagram** ❌
- **Missing**:
  - MLOps pipeline diagram
  - Architecture overview
- **Action**:
  - Create diagram (using draw.io or similar)
  - Add to `reports/figures/`
  - Include in documentation

---

## 📦 Missing Dependencies

Add these to `pyproject.toml`:

```toml
dependencies = [
    # ... existing dependencies ...
    "hydra-core>=1.3.2",           # Configuration management
    "wandb>=0.23",                  # Experiment tracking
    "loguru>=0.7.0",                # Logging
    "snakeviz>=2.2.2",              # Profiling visualization
    "torch-tb-profiler>=0.4.3",    # PyTorch profiling
    "onnxruntime>=1.23.2",         # ONNX runtime
    "locust>=2.42.3",               # Load testing
    "streamlit>=1.51",              # Frontend
    "evidently>=0.7.16",            # Data drift detection
    "prometheus-fastapi-instrumentator>=7.1",  # Prometheus metrics
    "httpx>=0.28.1",                # HTTP client for testing
]
```

---

## 🎯 Priority Order

### High Priority (Core Requirements)
1. **Hydra configuration** - Essential for experiment management
2. **Wandb integration** - Required for experiment tracking
3. **FastAPI implementation** - Core deployment component
4. **Cloud deployment** - Required for Week 2
5. **API testing** - Required for Week 2

### Medium Priority (Important Features)
6. **CML workflows** - Data and model registry triggers
7. **GCP integration** - Cloud Storage, Cloud Build, Vertex AI
8. **Load testing** - Required for Week 2
9. **ONNX deployment** - Specialized ML deployment
10. **Frontend** - User interface

### Lower Priority (Advanced Features)
11. **Data drift detection** - Week 3 requirement
12. **System monitoring** - Week 3 requirement
13. **Model optimization** - Week 3 requirement
14. **Profiling** - Should be done but not blocking
15. **Documentation deployment** - Extra requirement

---

## 📝 Quick Start Checklist

- [ ] Add missing dependencies to `pyproject.toml`
- [ ] Implement Hydra configuration in `configs/`
- [ ] Integrate Wandb in training script
- [ ] Implement FastAPI app in `api.py`
- [ ] Write API tests
- [ ] Set up GCP bucket and DVC remote
- [ ] Create Cloud Build configuration
- [ ] Deploy API to Cloud Run
- [ ] Add load testing with Locust
- [ ] Implement ONNX export and serving
- [ ] Create Streamlit frontend
- [ ] Set up data drift detection
- [ ] Add Prometheus metrics
- [ ] Deploy documentation to GitHub Pages

---

## 🔗 Reference Links

- [DTU MLOps Course](https://github.com/SkafteNicki/dtu_mlops)
- [Course Checklist](.github/agents/dtu_mlops_agent.md)
- [MLOps Learning Checklist](../MLOPS_LEARNING_CHECKLIST.md)
