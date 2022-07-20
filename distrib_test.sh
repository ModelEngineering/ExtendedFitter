#!/bin/bash
# Setup the virtual environment to test a PyPI distribution
cd $HOME
DIR=testing_fitterpp
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
cd ~/fitterpp
nose2 tests
