# installation:
* clone the repo
  * `$ git clone https://github.com/Sheypex/ba_proto.git ba`
* install conda (see the [documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) for reference): 
  * `$ curl https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh > tmp.sh` (the current version can be found at [anaconda.com/products/individual](https://www.anaconda.com/products/individual). E.g. the 64-Bit (x86) Installer for Linux)
  * `$ chmod +x tmp.sh`
  * `$ ./tmp.sh`
  * `$ rm tmp.sh`
  * `$ conda init`
* install python dependencies:
  * `$ conda config --add channels conda-forge`
  * `$ conda create -n py --file ba/py/dependencies.conda` (depends on where you cloned to)
  * `$ conda activate py`
* install other dependencies:
  * `$ apt install libpq-dev graphviz graphviz-dev`
* done. you should now be able to run:
  * `$ python linRegPred.py ...`
