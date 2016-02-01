# rain-gage-tools

The goal of this project is to create a comprehensive set of tools for reading, quality checking, and analyzing data from rain gage 
networks. The emphasis is on quickly generating, interactve plots that speed up data processing and reduce duplication of effort for 
each new dataset. Of special interest is the intercomparison of computed radar rainfall and gage rainfall. 

These tools are being developed by the Hydrometeorology Group in the Civil Environmental Engineering Department at Princeton University 
(<http://hydrometeorology.princeton.edu/>) 


## Installation

The easiest way to install this package with all its dependencies involves using conda (<http://conda.pydata.org/miniconda>):

    $ conda env create -f environment.yml
    $ python setup.py install
    
As of right now, the r based tools won't work until you go in and manually install FNN, sp, intervals, spacetime, and gstat
The easiest way that I have found to do that is by downloading each package from source and then running

    $ R

Once R opens successfully, for each package run:

    > install.packages("./source_package.tar.gz", repos=NULL) 
  
I am working on making conda recipes though for these packages, so hopefully I will get it worked out soon and you won't need
to install these less common packages manually. At the moment the dependencies in the environment.yml folder are only verified for linux. 
