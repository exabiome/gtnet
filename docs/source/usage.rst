Running GTNet
=============

Getting taxonomic classifications for all sequences in a Fasta file.

.. code:: python

  def my_function():
        "just a test"
              print 8/2

.. code:: bash
  gtnet classify data/small.fna > data/small.tax.csv


GTNet steps
-----------
GTNet consists of two main steps: 1) get scored predictions of taxonoimc assignments and 2) filter
scored predictions. The previous command combines these two commands into a single command with a
default false-positive rate. The two steps have been separated into two commands for those who
want to experiment with different false-positive rates.

Getting predictions for all sequences in a Fasta file.

.. code:: bash
  gtnet predict data/small.fna > data/small.tax.raw.csv

The first time you run *predict*, the model file will be downloaded and stored in the
same directory that the *gtnet* package is installed in. Therefore, for the this to be successful,
you must have write privileges on the directory that *gtnet* is installed in.

Filtering predictions

.. code:: bash
  gtnet filter --fpr 0.05 data/small.tax.raw.csv > data/small.tax.csv

