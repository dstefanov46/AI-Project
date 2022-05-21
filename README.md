## Practical Assignment in the Subject *Artificial Intelligence*

This repository contains the project titled *Reinforcement Learning Trading Bot for Cryptocurrencies* created as part of
the subject *Artificial Intelligence*. 

The work was done on a **Windows 10 (x64)** machine, and to re-create the environment in which the project was developed, you need
to run:
```
conda env create -f environment.yml
```
The ``environment.yml`` file is part of this repo. And, to start working in the environment, simply run:
```
conda activate environment
```

Our work can be mainly divided in 3 files:
- ``libs.py``: in this file we import all the necessary libraries,
- ``helper_funcs.py``: here we have all the functions we need, both for the data preprocessing and the learning of the model,
- ``callback.py``: in this file we've created a **callback** class which saves the best-perfoming model up to a certain
point during the training.

In the folder ``data``, you can find the **.csv** files containing time series data about the cryptocurrencies we 
analyzed.

Lastly, if you would like to replicate the results we have obtained during the course of this project, please run the
code inside the **jupyter notebook** named ``report_result_replication.ipynb``. In the notebook, you'll find the code
for the entire hyperparameter grid search we perform. When run, this code stores the results in a file titled 
``results_{cc}.txt`` where ``cc`` can be of the 3 cryptocurrencies we investigated: ``ADA``, ``BTC`` and ``ETH``.

--------

For any suggestions on how to improve this work, please contact me at 
[ds2243@student.uni-lj.si](https://gmail.google.com/inbox/).
