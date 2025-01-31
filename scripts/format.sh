#!/usr/bin/env bash
set -x

ruff check src tests --fix
ruff format src tests
