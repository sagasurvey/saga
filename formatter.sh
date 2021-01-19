#!/bin/bash
DIR=$(dirname "$0")
black -l120 -tpy36 "$DIR"
isort "$DIR"
