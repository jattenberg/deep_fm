#!/bin/bash
set -e

venv="venv"

hash virtualenv
if [ "$?" != "0" ];
then
    echo "installing virtualenv"
    pip install virtualenv;
fi

echo "building virtualenv: $venv"

virtualenv $venv

echo "installing deep_fm"
$venv/bin/pip install -e .
$venv/bin/jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip
$venv/bin/jupyter nbextension enable jupyter-black-master/jupyter-black

echo "===================="
echo "===================="
echo "===================="


echo "being by activating the virtualenv or running:"
echo "$venv/bin/jupyter notebook"
