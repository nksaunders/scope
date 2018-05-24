# Miniconda
wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
export PATH=/home/travis/miniconda2/bin:$PATH
conda update --yes conda


conda create --yes -n test python=$TRAVIS_PYTHON_VERSION
conda install --yes numpy scipy matplotlib nose numba pandas
ln -sf $(which gcc) x86_64-conda_cos6-linux-gnu-gcc


# Conda Python
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create --yes -n test python=$TRAVIS_PYTHON_VERSION
conda install --yes numpy scipy matplotlib nose numba pandas
ln -sf $(which gcc) x86_64-conda_cos6-linux-gnu-gcc
conda activate test

pip install tqdm
pip install ipython
pip install sklearn
pip install scipy
pip install george
pip install astroml
pip install everest-pipeline

# Build the code
# Build the extension
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    python setup.py install
else
    CXX=g++-4.8 CC=gcc-4.8 python setup.py install
fi
