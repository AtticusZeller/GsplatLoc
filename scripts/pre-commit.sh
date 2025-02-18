#! /usr/bin/env bash

cd ..
# Install requirements by poetry
poetry install

# Run pre-commit tests
# poetry run pre-commit autoupdate
# poetry run pre-commit clean
# poetry run pre-commit install
poetry run pre-commit run --all-files
pytest tests
# Generate coverage report --cov=./ --cov-report=xml --cov-report=html -vv
# poetry run pytest  --cov=./ --cov-report=xml --cov-report=html -vv
