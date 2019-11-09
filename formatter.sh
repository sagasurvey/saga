#!/bin/bash
DIR=`dirname "$0"`
black -tpy36 $DIR
isort -rc $DIR
