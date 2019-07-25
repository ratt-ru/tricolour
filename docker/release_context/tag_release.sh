#!/bin/bash
if [ -z ${1+x} ]; 
then 
    echo "Expected release tag as argument";
    exit 1
fi

echo "TAGGING NEW RELEASE AS ${1}"
rm -rf ./rel && rm -rf ./build && rm -rf ./dist && rm -rf ./tmpvenv && rm -f *.spec && rm -f *.log && \
rsync -av --progress ../../ ./rel --exclude docker --exclude .git --exclude '__pycache__' --exclude '*.pyc' --exclude '*.pyo' && \
virtualenv -p python3 tmpvenv && . tmpvenv/bin/activate && pip install ./rel[obfuscation] && pip uninstall -y tricolour && \
pyinstaller rel/tricolour/apps/tricolour/tricolourexe.py && \
mkdir dist/tricolourexe/tricolour && cp -r rel/tricolour/conf dist/tricolourexe/tricolour/conf && cp -r rel/tricolour/data dist/tricolourexe/tricolour/data && \
rm -rf rel && docker build -f release.docker -t ${1} . && \
rm -rf ./rel && rm -rf ./build && rm -rf ./dist && rm -rf ./tmpvenv && rm -f *.spec && rm -f *.log && \
echo "Successfully build tag ${1}" && exit 0

rm -rf ./rel && rm -rf ./build && rm -rf ./dist && rm -rf ./tmpvenv && rm -f *.spec && rm -f *.log && \
echo "Failed to build tag ${1}" && exit 1