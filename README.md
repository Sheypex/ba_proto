# installation:
* clone the repo
  * `$ git clone https://github.com/Sheypex/ba_proto.git ba`
* install conda: 
  * `$ curl https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh > tmp.sh`
  * `$ chmod +x tmp.sh`
  * `$ ./tmp.sh`
  * `$ rm tmp.sh`
  * `$ conda init`
* install dependencies:
  * `$ conda config --add channels conda-forge`
  * `$ conda create -n py --file ba/py/dependencies.conda` (depends on where you cloned to)
  * `$ conda activate py`
* install other dependencies:
  * `$ apt install libpq-dev graphviz graphviz-dev`
* done. you should now be able to run:
  * `$ python linRegPred.py ...`
