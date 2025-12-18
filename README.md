# Iris Classifier (Decision Tree)

## Overview

End‑to‑end ML example. Trains a decision tree classifier on the classic Iris dataset using scikit‑learn. Notebook in `notebooks/main-notebook.ipynb` contains exploratory data analysis and visualization. `src/train.py` contians a CLI tool that trains the model. `tests/test_train.py` contains tests for the training script.

## Quick start

### Using pip

```bash
git clone https://github.com/salty511/iris-classifier.git
cd iris-classifier
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 src/train.py
```

## Usage

```bash
python3 src/train.py --help
```

Shows the command line options for training a model. The default values are:

--test-size 0.2
--random-state 42

```bash
python3 src/train.py
```

Trains a decision tree classifier on the Iris dataset and saves the confusion martrix figure to `outputs/confusion_matrix.png`.

## Testing

```bash
python3 -m pytest
```

Runs all tests in the `tests` directory. test_train.py uses parameterized tests to test multiple values for test_size, asserting that the accuracy is greater than 95%.
