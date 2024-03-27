set -e
set -u
set -x
WORKSPACE_ROOT="$WORKSPACE/$BUILD_NUMBER"
echo "Setting up build in $WORKSPACE_ROOT"
TEST_OUTPUT_DIR_REL=testcase_output
TEST_OUTPUT_DIR="$WORKSPACE_ROOT/$TEST_OUTPUT_DIR_REL"
mkdir $TEST_OUTPUT_DIR
TEST_DATA_DIR="$WORKSPACE/../../../test-data"


# build
docker build -f ${WORKSPACE_ROOT}/projects/tricolour/docker/python36.docker -t tricolour.1804.py36:${BUILD_NUMBER} ${WORKSPACE_ROOT}/projects/tricolour/
docker build -f ${WORKSPACE_ROOT}/projects/tricolour/docker/python38.docker -t tricolour.2004.py38:${BUILD_NUMBER} ${WORKSPACE_ROOT}/projects/tricolour/
docker build -f ${WORKSPACE_ROOT}/projects/tricolour/docker/python310.docker -t tricolour.2204.py310:${BUILD_NUMBER} ${WORKSPACE_ROOT}/projects/tricolour/

#run tests
tar xvf $TEST_DATA_DIR/acceptance_test_data.tar.gz -C $TEST_OUTPUT_DIR
TEST_MS_REL=1519747221.subset.ms

# test 3.8
docker run \
    --rm \
    -v $TEST_OUTPUT_DIR:/testdata \
    --env TRICOLOUR_TEST_MS=/testdata/$TEST_MS_REL \
    --workdir /code \
    --entrypoint /bin/sh \
    tricolour.2004.py38:${BUILD_NUMBER} \
    -c "python3 -m pytest --flake8 -s -vvv ."

rm -rf $TEST_OUTPUT_DIR/TEST_MS_REL
tar xvf $TEST_DATA_DIR/acceptance_test_data.tar.gz -C $TEST_OUTPUT_DIR

# test 3.10
docker run \
    --rm \
    -v $TEST_OUTPUT_DIR:/testdata \
    --env TRICOLOUR_TEST_MS=/testdata/$TEST_MS_REL \
    --workdir /code \
    --entrypoint /bin/sh \
    tricolour.2204.py310:${BUILD_NUMBER} \
    -c "python3 -m pytest --flake8 -s -vvv ."
