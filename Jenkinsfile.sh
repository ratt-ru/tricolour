WORKSPACE_ROOT="$WORKSPACE/$BUILD_NUMBER"

# build
docker build -f ${WORKSPACE_ROOT}/projects/tricolour/docker/python36.docker -t tricolour.1804.py36:${BUILD_NUMBER} ${WORKSPACE_ROOT}/projects/tricolour/

#run tests
docker run --rm tricolour.1804.py36:${BUILD_NUMBER}
