Updating GTNet
==============

As the `GTDB taxonomy <https://gtdb.ecogenomic.org/>`_ is updated, GTNet will also need to be updated. This amounts
to retraining the network with the new taxonomy and updating the `gtnet software <https://github.com/exabiome/gtnet>`_
to use the new model and taxonomy.

Training a new model
--------------------
Software for training GTNet is available in the `deep-taxon <https://github.com/exabiome/deep-taxon>`_ repository.


Uploading to OSF
++++++++++++++++
Once a model is trained, calibrated, and packaged, the deployment package needs to be made publicly available. GTNet is
currently carried hosted on `OSF <https://osf.io/cwaqs/>`_.


Updating the gtnet software
---------------------------
After training a new model and packaging the model, the :py:class:`~gtnet.utils.DeployPkg` class will need to be
updated with the new URL and checksum of the new deployment package. This can be done starting around
`here <https://github.com/exabiome/gtnet/blob/b9ba8a4fb1a63affd9047005c92c12799df9c2b7/src/gtnet/utils.py#L36>`_
in the code.
