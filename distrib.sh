#!/bin/bash
# Create a new distribution.
rm -rf dist
#python setup.py sdist
python -m build
twine upload dist/*.*
