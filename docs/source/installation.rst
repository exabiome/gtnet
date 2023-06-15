Installation
============

GTNet is available on the Python Package Index.

.. code:: bash

  pip install gtnet

GPU acceleration
----------------
GTNet uses `PyTorch <https://pytorch.org/>`_, so it is capable of GPU acceleration with CUDA. As long as
CUDA is available on your system, GTNet will detect if CUDA is available and make GPU acceleration available.

If your system is equipped with NVIDIA GPUs, but are unsure if CUDA is installed, we recommend installing PyTorch
and the `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ using Conda.

For example, if you would like to run PyTorch with CUDA Toolkit 11.8, you can run the following commands:

.. code:: bash

  conda create -n gtnet-env
  conda activate gtnet-env
  conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
  pip install gtnet
