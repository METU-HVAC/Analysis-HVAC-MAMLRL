# HVAC Reinforcement Learning
This is the official code repository for the paper "Analysis of Model-Agnostic Meta-Reinforcement Learning on Automated HVAC Control".
Install the requirements using the following command:

```
pip install -r requirements.txt
```

To use this repository., you need to download sinergym and EnergyPplus first.
Follow the instructions on https://github.com/ugr-sail/sinergym/blob/main/INSTALL.md for installation

## Recomended installation
### Install EnergyPlus
Install EnergyPlus-23-1-0 version from github.
```
curl -L -o EnergyPlus-23.1.0-87ed9199d4-Linux-Ubuntu22.04-x86_64.sh https://github.com/NREL/EnergyPlus/releases/download/v23.1.0/EnergyPlus-23.1.0-87ed9199d4-Linux-Ubuntu22.04-x86_64.sh
```

Give permission for executable.
```
chmod +x EnergyPlus-23.1.0-87ed9199d4-Linux-Ubuntu22.04-x86_64.sh
```

Run the installer
```
./EnergyPlus-23.1.0-87ed9199d4-Linux-Ubuntu22.04-x86_64.sh 
```
After installating Energyplus, add environment variables and update python path. Add these lines to your ".bashrc" file. It should be located at "~/.bashrc"
```
    export PYTHONPATH=/path/to/energyplus:$PYTHONPATH
    export EPLUS_PATH=/path/to/energyplus
```
Re-login, go to new terminal or use below command to update it.

```
    source ~/.bashrc
```
### Install Sinergym

Download Sinergym repository

You need to install Sinergym with version 2.5.0 with Ubuntu 22.04. A table for Energyplus-Sinergym compatibility is found here:
https://ugr-sail.github.io/sinergym/compilation/main/pages/installation.html

Use Conda for environment management. Download Miniconda if you don't have it.Then;
```
cd sinergym
conda env create -f python_environment.yml
conda activate sinergym
```

To confirm the instalation, you can run "pytest tests/ -vv" on sinergym folder.

If al tests pass, you can download this repository on the sinergym folder.
```
    cd /path/to/sinergym
    git clone https://github.com/METU-HVAC/Analysis-HVAC-MAMLRL.git
```


After this, you need to add A403 environment configuration to your Sinergym environment by transferring the files from sinergym-addons to your sinerggym file.
