cd VietnameseTextNormalizer
export PYTHON3_PATH=/usr/local/Cellar/python@3.9/3.9.13_2/Frameworks/Python.framework/Versions/3.9
export PYTHON3_DEV_INCULE=$PYTHON3_PATH/include/python3.9
export PYTHON3_LIB_PATH=$PYTHON3_PATH/lib
export PYTHON3_LIB_NAME=python3.9
export GPP_COMPILER=g++
cp -f MakefilePython3MACOS Makefile
make -j

cd ..


ls "/usr/local/Cellar/python@3.9/3.9.13_2/Frameworks/Python.framework/Versions/3.9/"
