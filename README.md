# installation:
* clone the repo
* install conda: $ curl https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh > tmp.sh
* chmod +x tmp.sh
* ./tmp.sh
* $ conda init
* $ cd ba/py (or wherever you cloned to)
* $ conda config --add channels conda-forge
* $ conda create -n py --file dependencies.conda
* $ conda activate py
* $ apt install libpq-dev graphviz graphviz-dev
* $ python linRegPred.py ...
