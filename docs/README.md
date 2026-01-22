Documentation
----------

This project uses [MkDocs](http://www.mkdocs.org/) with the Material theme for documentation.

## Viewing Documentation

**Online**: Visit [https://chran103.github.io/ml_ops102/](https://chran103.github.io/ml_ops102/)

**Locally**: Run `uv run invoke serve-docs` and visit http://127.0.0.1:8000

## Building Documentation

Build locally:

```bash
uv run invoke build-docs
```

Or directly:

```bash
uv run mkdocs build --config-file docs/mkdocs.yaml
```

## Serving Documentation Locally

```bash
uv run invoke serve-docs
```

Or directly:

```bash
uv run mkdocs serve --config-file docs/mkdocs.yaml
```

## GitHub Pages Deployment

Documentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch.

The deployment workflow:
1. Detects changes in `docs/**`
2. Builds the documentation with MkDocs
3. Deploys to the `gh-pages` branch using `mkdocs gh-deploy`

See `.github/workflows/docs.yaml` for the deployment configuration.

### Manual Deployment

To manually deploy documentation to GitHub Pages:

```bash
uv run mkdocs gh-deploy --config-file docs/mkdocs.yaml --force
```

## Editing Documentation

All documentation source files are in `docs/source/`:
- `index.md` - Home page
- `workflow.md` - Development workflow
- `local-setup.md` - Local setup guide
- `cloud-setup.md` - Cloud deployment guide

Configuration is in `docs/mkdocs.yaml`.
