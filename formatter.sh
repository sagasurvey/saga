#!/bin/bash
DIR=`dirname "$0"`
black -tpy35 $DIR
isort -rc $DIR
