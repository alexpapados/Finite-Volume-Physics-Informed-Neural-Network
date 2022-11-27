# Finite Volume Physics-Informed Neural Networks for Compressible Flow & Hyperbolic Conservation Laws
## Author: Alexandros Papados ##
## FV-PINNs ##
FV-PINNs migrates away from using pre-built automatic differentiation kernels to differentiate the neural network w/r to the governing partial differential equations. 
Instead we utilize underlying finite volume schemes to calculate gradients of fluxs and classic time integration schemes such as RK-X methods. 
I coin this method Finite Volume Physics-Informed Neural Networks. Instead of the physics coming from the underlying PDE, the physics come from the
numerical discretization scheme.
## Setup ##
First, clone repository:

`git clone https://github.com/alexpapados/FV-PINNs/`

Once the repository is cloned locally, run:

`bash setup.sh`

If you do not have bash on your machine, try:

`chmod u+x setup.sh;
./setup.sh`

## Libraries ##
All FV-PINNs code was written using Python. The libraries used are:
* PyTorch 
* Pandas
* SciencePlots
* NumPy
* ScriPy
* Time

---------------------------------------------------------------------------------------------------------------------------------
Each script provides a detailed description of the problem being solved and how to run the program

## How to Run the Code ##
Preferably using an IDE such as PyCharm, and once all libraries are downloaded, users may simply run the code and each case as described in individual scripts.

