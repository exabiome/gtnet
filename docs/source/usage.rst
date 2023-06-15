Running GTNet
=============

GTNet comes with multiple commands. The simplest way of running GTNet is to use the ``classify`` command.

.. code:: bash

  gtnet classify data/small.fna > data/small.tax.csv


GTNet steps
-----------
GTNet consists of two main steps: 1) get scored predictions of taxonoimc assignments and 2) filter
scored predictions. The previous command combines these two commands into a single command with a
default false-positive rate. The two steps have been separated into two commands for those who
want to experiment with different false-positive rates.

Getting predictions
^^^^^^^^^^^^^^^^^^^

To get predictinos for all sequences in a Fasta file, use the ``predict`` subcommand.

.. code:: bash

  gtnet predict data/small.fna > data/small.tax.raw.csv

The first time you run ``predict``, the model file will be downloaded and stored in the
same directory that the *gtnet* package is installed in. Therefore, for the this to be successful,
you must have write privileges on the directory that *gtnet* is installed in.

Filtering predictions
^^^^^^^^^^^^^^^^^^^^^

After getting predicted and scored taxonomic classifications, you can filter the raw classifications
to a desired false-positive rate.

.. code:: bash

  gtnet filter --fpr 0.05 data/small.tax.raw.csv > data/small.tax.csv


GPU acceleration
----------------
If CUDA is available on your system, the ``classify`` and ``predict`` commands will have the option ``-g/--gpu`` to enable
using the available GPU to accelerate neural network calculations.
