#!/bin/bash

venv="deep_fm-virtualenv"

echo "building virtualenv: $venv"

hash virtualenv
if [ "$?" != "0" ];
  then
    pip install virtualenv;
fi

virtualenv $venv

echo "installing deep_fm"
$venv/bin/pip install -e .


echo "===================="
echo "===================="
echo "===================="


echo "being by activating the virtualenv or running:"
echo "`$venv/bin/jupyter notebook`
