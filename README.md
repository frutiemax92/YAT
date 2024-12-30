# YAT : Yet Another Trainer for diffusion models

This is Yet Another Trainer for diffusion models, currently supports those models:
1. SANA
2. Pixart-Sigma

## How to train

First, install the required packages.

    pip install --upgrade -r requirements.txt

Then configure your accelerate configuration.

    accelerate config

It will prompt you with an interactive menu, the most important options are your number of graphic cards and the mixed precision setting.
After that, prepare a config.yaml file, you can check the example in tests. 
Finally execute the script.

    accelerate launch train_sana.py --config config.yaml


