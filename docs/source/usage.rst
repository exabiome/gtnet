Running GTNet
=============

GTNet comes with multiple commands. The simplest way of running GTNet is to use the ``classify`` command.

.. code:: bash

  gtnet classify genome.fna > genome.tax.csv

This command generates one classification for the entire file, and should be used to get classification for metagenome bin.
Use the ``-s/--seqs`` flag to get classifications for the individual sequences in ``genome.fna``

.. Attention::
   The first time you run ``classify`` and ``predict`` (see below), the model file will be downloaded and stored in the same
   directory that the *gtnet* package is installed in. Therefore, for the this to be successful, you must have write privileges
   on the directory that *gtnet* is installed in.


.. code:: bash

  gtnet classify --seqs genome.fna > genome.seqs.tax.csv


The ``classify`` command can take multiple fasta files, and will produce line per file in the output. For example, the following
command will contain two lines:

.. code:: bash

  gtnet classify bin1.fna bin2.fna > bins.tax.csv


GTNet steps
-----------
GTNet consists of two main steps: 1) get scored predictions of taxonoimc assignments and 2) filter
scored predictions. The previous command combines these two commands into a single command with a
default false-positive rate. The two steps have been separated into two commands for those who
want to experiment with different false-positive rates.

Getting predictions
^^^^^^^^^^^^^^^^^^^

To get predictinos for all sequences in a Fasta file, use the ``predict`` subcommand. This command also accepts multiple fasta files
and the ``-s/--seqs`` argument for getting predictions for individual sequences.

.. code:: bash

  gtnet predict genome.fna > genome.tax.raw.csv

Filtering predictions
^^^^^^^^^^^^^^^^^^^^^

After getting predicted and scored taxonomic classifications, you can filter the raw classifications
to a desired false-positive rate.

.. code:: bash

  gtnet filter --fpr 0.05 genome.tax.raw.csv > genome.tax.csv

The ``filter`` command supports predictions for whole files and individual sequences.

GPU acceleration
----------------
If CUDA is available on your system, the ``classify`` and ``predict`` commands will have the option ``-g/--gpu`` to enable
using the available GPU to accelerate neural network calculations.
