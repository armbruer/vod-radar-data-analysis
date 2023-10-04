# Parameter Distribution Extraction

## Dataset Download

See [DATASET-README](./DATASET-README.md) (default README of the dataset) and [VoD Website](https://intelligent-vehicles.org/datasets/view-of-delft/) for instructions on how to download and install the dataset.

**Please make sure to also setup the tracking ids. A label file needs to be separately downloaded for this and placed in the correct folder. Or disable the respective code in this repository.**

## Setup

Follow these steps to setup the repository:

**Warning:** Do not use anaconda or miniconda to setup this repository. It's horribly slow!**

Instead use [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html).

Installation guide for Arch Linux:

1. Clone

```shell
git clone git@gitlab.lrz.de:master-thesis8/delt-radar-data-extractor.git
cd delt-radar-data-extractor
```

2. Install micromamba (from AUR using your favorite helper, e.g. [aura](https://github.com/fosskers/aura))

```shell
sudo aura -A micromamba-bin
```

3. Create a python environment with the packages required for this repository using micromamba

```shell
micromamba create -n view-of-delft -f environment-update.yml -c conda-forge
```

4. Activate the environment 

As with any python virtual environment you need to run this everytime before running the code in this repo.

```shell
micromamba activate view-of-delft
#source /home/eric/micromamba/bin/activate view-of-delft
```

5. Alternatively to activating every time use VS-code

PyCharm does not support micromamba at the time of writing.

Install the `Python Extension Pack` and select the interpreter, i.e., the micromamba virtual environment you just created.

6. Directory setup:

Your directory layout needs to look like this:

```shell
❯ tree -L 1         
.
├── delt-radar-data-extractor # (repository)
└── view_of_delft_PUBLIC # (dataset)
```

7. Run [main.py](./extraction/main.py).
