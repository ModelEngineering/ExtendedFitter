#!/bin/bash
# Setup the virtual environment to test a PyPI distribution
# $1 - repo directory
REPOPATH=~/home/Technical/repos
pushd ${REPOPATH}
REPODIR=$1
DIR=${REPOPATH}/testing_${REPODIR}
if [ -d $DIR ] 
then
    rm -rf $DIR
    echo "Deleting existing $DIR"
fi
python3 -m venv ${DIR}
source ${DIR}/bin/activate
pip install --upgrade pip
pip install fitterpp
echo "Testing the install"
cd ${REPODIR}
nosetests tests
