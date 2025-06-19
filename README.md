# Spacepoint SSV

A machine-learning based tool to identify a missed second shower from a Pi0 that was misclustered as part of a cosmic ray using Wire-Cell 3D cosmic spacepoints in MicroBooNE. This type of tool should be useful for single photon searches in MicroBooNE and other LArTPCs.

Similar idea to the Second Shower Veto (SSV) used in [Phys. Rev. Lett. 128, 111801 (2022)](https://doi.org/10.1103/PhysRevLett.128.111801).

# To set up the python environment:
```
python3 -m venv venv
source venv/bin/activate # run this again each time you interact with the code, the rest of the steps here are only needed once
# run the command corresponding to your machine and pip here: https://pytorch.org, for example `pip install torch torchvision torchaudio` on Mac
pip install ipywidgets ipykernel awkward-pandas uproot tqdm pandas numpy matplotlib plotly
```

# To preprocess the spacepoints from a root file:
```
python preprocess_spacepoints.py -f input_files/bdt_convert_superunified_bnb_ncpi0_full_spacepoints.root -n 100
```

# To run the training:
```
python train.py
```