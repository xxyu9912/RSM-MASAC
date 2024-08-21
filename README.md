# RSM-MASAC

This repository contains the implementation of the paper: [Communication-Efficient Soft Actor-Critic Policy Collaboration via Regulated Segment Mixture in the Internet of Vehicles](https://arxiv.org/abs/2312.10123).

Note that this is a research project and by definition is unstable. Please write to us if you find something not correct or strange. We are sharing the codes under the condition that reproducing full or part of codes must cite the related paper. 

## Setup

The installation steps are based on the simulation environment provided by [Flow](https://flow.readthedocs.io/en/latest/flow_setup.html#installing-flow-and-sumo).

**System Requirements:**

- Python version: 3.7.3
- PyTorch version: 1.7.0
- Operating System: Ubuntu 18.04

**Installation**

1. **Create and activate a Conda environment:**

    ```bash
    conda env create -f environment.yml
    conda activate flow
    ```

2. **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Install SUMO:**

    Run the provided script to install pre-compiled SUMO binaries and Python tools:

    ```bash
    scripts/setup_sumo_ubuntu1804.sh
    ```



## Run
To run baseline:
```bash
python baseline.py
```

To run the training process:

```bash
python train.py --output ./ 
```

For additional command-line options, refer to `train.py`.

