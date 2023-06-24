Trainig a GTNet model
=====================

Download data
-------------

GTDB metadata file
^^^^^^^^^^^^^^^^^^

- Download metadata file from GTDB

NCBI Genomes
^^^^^^^^^^^^

- Download genomes from NCBI

.. code:: bash

  deep-taxon ncbi-fetch --metadata metadata.csv ncbi

.. Attention:: To speed this process up, consider using the`-p/--processes` argument to parallelize.

Convert data to HDMF file for training
-------------------------------------

- Build training file by running `prepare-data` command.

Build training data file
^^^^^^^^^^^^^^^^^^^^^^^^

- Use --rep flag to only convert representative genomes

.. code:: bash

  deep-taxon prepare-data --rep --genomic ncbi metadata.csv r207.rep.h5

.. Attention:: To speed this process up, consider using the`-p/--procs` argument to parallelize calculating dataset size.

Build calibration data file
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Use --calib flag to only convert representative genomes

.. code:: bash

  deep-taxon prepare-data --calib --genomic ncbi metadata.csv r207.nonrep.calib.h5


Training GTNet Model
--------------------

.. code:: bash

  deep-taxon train training_config.yml r207.rep.h5 gtnet.ckpt

Running on Perlmutter
^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

  deep-taxon train-job --perlmutter -t 720 -n 8 -g 4 -P mXXXX -N train_r207 -a 8 -E prod_train -D r207 --csv -e 50 -o train r207.rep.h5 training_config.yml sbatch.sh --submit

This will prompt you for a message describing the run. This message will be written to `$PWD/jobs.log` along with other information about the
job, such as output directory and where standard output/error will be saved to. The log file, `$PWD/jobs.log` can be changed with the `-L/--log`
flag. The message can be passed in directly on the command line using the `-m/--message` flag.

Running inference
-----------------

.. code:: bash

  deep-taxon infer training_config.yml r207.nonrep.calib.h5 gtnet.ckpt


Useful flags


.. code:: bash
   -g/--gpus
   -p/maxprob INT


Inference is parallelizable with MPI i.e. `mpirun -n 4 deep-taxon infer ...`.

Additional flags can by found with `deep-taxon infer --help`.


Running on Perlmutter
^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

  deep-taxon infer-job --perlmutter -T 60 -F training_config.yml r207.nonrep.calib.h5 gtnet.ckpt sbatch.sh --submit


This will prompt you for a message describing the run. This message will be written to `$PWD/jobs.log` along with other information about the
job, such as output directory and where standard output/error will be saved to. The log file, `$PWD/jobs.log` can be changed with the `-L/--log`
flag. The message can be passed in directly on the command line using the `-m/--message` flag.

Additional flags can by found with `deep-taxon infer-job --help`.


Training calibration model
--------------------------

.. code:: bash

  deep-taxon conf-model --bins conf_model/bins outputs.h5
  deep-taxon conf-model conf_model/contigs outputs.h5


Building Deployment packages
----------------------------

.. code:: bash

  deep-taxon deploy-pkg r207.rep.h5 training_config.yml gtnet.ckpt conf_model/bins/metadata.json conf_model/contigs/metadata.json deploy_pkg


This will create a Zip archive, `deploy_pkg.zip`.

.. Attention:: The final argument, `deploy_pkg` must be named this. It is currently a dependency `here <https://github.com/exabiome/gtnet/>`_


Upload to OSF
-------------

Upload the Zip archive to `Open Science Framework <https://osf.io/....>`.


Updating code
-------------

Deploying the updated model for public use requires changing where the GTNet code looks to download a deployment package,
updating expected output for testing, and then cutting a release from this updated code.


Update GTNet
^^^^^^^^^^^^

Once the archive is, copy the download link. This can be found by clicking
on the upload Zip archive in the OSF file browser. Click the upper right ... icon. Right click on "Download" and click "Copy Link"

Set the class attribute `DeployPkg._deploy_pkg_url <https://github.com/exabiome/gtnet/blob/d25bd39027980b8ec3de20963790ff745fd79a88/src/gtnet/utils.py#L37`_ as this link.

Calculate the md5 checksum for `deploy_pkg.zip` and set the class attribute `DeployPkg._checksum  https://github.com/exabiome/gtnet/blob/d25bd39027980b8ec3de20963790ff745fd79a88/src/gtnet/utils.py#L39`_
to this hash.

Commit and push these changes to a branch.

Update test data
^^^^^^^^^^^^^^^^

Update the expected file outputs

- data/small.raw.test.csv
- data/small.seqs.raw.test.csv
- data/small.seqs.tax.test.csv
- data/small.tax.test.csv


Commit and push these to the branch you are using


Once tests have passed and code has been reviewed, merge changes to the main branch.

Cutting a release
^^^^^^^^^^^^^^^^^

Update your local copy of the main branch.
Create and push a signed tag from the current state of the main branch.

.. code:: bash
   git tag --sign -m \"$REPO $REL\" $REL origin/main; \
   git push origin $REL


From here, continuous deployment will build a wheel and test install from the wheel. If the install
test passes, a release will be made to Github and PyPI.
