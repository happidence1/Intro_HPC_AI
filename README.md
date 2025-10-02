# Introduction to HPC and AI

## Code Snippets from Presentation - ModuLair

- clean up
```
cd $SCRATCH/intro_hpc_ai/notebooks
module purge
```
- create a Python virtual environment 
```
create_venv hpc_ai_env -d "Environment for Intro to HPC and AI" -t "GCCcore/12.2.0 Python/3.10.8"
```
- activate the virtual environment
```
source activate_venv hpc_ai_env
```
- install required packages
```
pip3 install jupyter
pip3 install pandas matplotlib
pip3 install scikit-learn
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
- deactivate the virtual environment
```
deactivate
```
