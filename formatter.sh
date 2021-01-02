#!/bin/bash
DIR=$(dirname "$0")
black -l100 -tpy36 "$DIR"
isort "$DIR"
