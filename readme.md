# ME5406 Example SAPP with Imitation Learning (IL)

This repository contains code for the course ME5406, focusing on single agent path planning using Reinforcement Learning (RL) and Imitation Learning (IL).

## Getting Started

To set up the environment, you can use the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

If the above method does not work, you can manually create the environment using the `requirements.txt` file:

```bash
conda create --name <env> --file requirements.txt
conda activate <env>
pip install --user pyglet==1.5.27 numpy==1.23.5 keras==2.4.2 imageio matplotlib scipy jupyter opencv-python gym
```

Replace `<env>` with your desired environment name.

## Usage

Once the environment is set up, you can start using the code for your path planning projects.

The main driver code is in the `ME5406_exampleSAPP_IL.ipynb` notebook, where you can define the rate of imitation learning (between 0 and 1). The `ME5406_exampleSAPP_IL-noVal.ipynb` notebook is the same code without valid action learning. Open any of these notebooks to begin experimenting with single agent path planning using RL and IL.
