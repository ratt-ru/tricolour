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

#run tests
tar xvf $TEST_DATA_DIR/acceptance_test_data.tar.gz -C $TEST_OUTPUT_DIR
TEST_MS_REL=1519747221.subset.ms

docker run \
    --rm \
    -v $TEST_OUTPUT_DIR:/testdata \
    --env TRICOLOUR_TEST_MS=/testdata/$TEST_MS_REL \
    --workdir /code \
    --entrypoint /bin/sh \
    tricolour.1804.py36:${BUILD_NUMBER} \
    -c "python -m pytest pytest --flake8 -s -vvv ."


