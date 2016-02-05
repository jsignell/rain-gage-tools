# rain-gage-tools

The goal of this project is to create a comprehensive set of tools for reading, quality checking, and analyzing data from rain gage 
networks. The emphasis is on quickly generating, interactve plots that speed up data processing and reduce duplication of effort for 
each new dataset. Of special interest is the intercomparison of computed radar rainfall and gage rainfall. 

These tools are being developed by the Hydrometeorology Group in the Civil Environmental Engineering Department at Princeton University 
(<http://hydrometeorology.princeton.edu/>) 


## Installation

The easiest way to install this package with all its dependencies involves using conda (<http://conda.pydata.org/miniconda>):

    $ conda config --add channels r jsignell
    $ conda env create -f environment.yml
    $ python setup.py install

#### Troubleshooting R

I have built conda-recipes for the r packages that are needed, but if for some reason the r based tools don't work, you can go in and manually install FNN, sp, intervals, spacetime, and gstat. The easiest way that I have found to do that is by downloading each package from source and then opening R and running:

    > install.packages("./source_package.tar.gz", repos=NULL) 
  
At the moment the dependencies in the environment.yml folder are only verified for linux. 

#### JSAnimation

The movies depend on a python to html animator created by Jake Vanderplas. In order for the movies to work at the moment you need to run. 
    
    $ git clone https://github.com/jakevdp/JSAnimation.git
    $ python JSAnimation/setup.py install
    
