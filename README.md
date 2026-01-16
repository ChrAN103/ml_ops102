# MLOps project

Project for Machine Learning Operations course (02476) by group 102

Julius Gregers Gliese Winkel, s234862 \
Rune Daugaard Harlyk, s234814 \
Christian Amtoft Nickelsen, s234863 \
Joseph An Duy Nguyen, s234826 

### Status

[![Unit Tests](https://github.com/ChrAN103/ml_ops102/actions/workflows/tests.yaml/badge.svg)](https://github.com/ChrAN103/ml_ops102/actions/workflows/tests.yaml)
[![Code linting](https://github.com/ChrAN103/ml_ops102/actions/workflows/linting.yaml/badge.svg)](https://github.com/ChrAN103/ml_ops102/actions/workflows/linting.yaml)

### Overall goal of the project

The goal of this project is to implement a simple deep neural network for fake news detection. The goal is to understand the end-to-end machine learning lifecycle and not on model complexity.

### What framework are you going to use, and you do you intend to include the framework into your project?

Pytorch will be used as the deep learning framework. It will be included in the project via dependencies in [pyproject.toml](pyproject.toml).

### What data are you going to run on (initially, may change)

The dataset is sourced from [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/fake-news-detection-using-machine-learning/). It contains 44919 rows each with 5 columns that consists of title, text, subject, date and class. Title and text will be used as input features and class, which is binary 0 for fake news and 1 for news, as the target.

### What models do you expect to use

We expect to use a simple Recurrent Neural Network (RNN) Long Short-Term Memory (LSTM) layers. 



## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
