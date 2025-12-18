# Iris Classifier (Decision Tree)

## Overview

End‑to‑end ML example from Digital Marketing Mastery Module → builds a decision‑tree classifier on the classic Iris dataset using scikit‑learn.

## Quick start

### Using pip

```bash
git clone https://github.com/salty511/iris-classifier.git
cd iris-classifier
python -m venv .venv && source venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

## Testing

```bash
python -m pytest
```

Runs all tests in the `tests` directory. test_train.py uses parameterized tests to test multiple values for test_size and random_state, asserting that the accuracy is greater than 95%.
