---
title: 'gtnet: A Python package for taxonomic labelling with the Genome Taxonomy Network'
tags:
  - Python
  - metagenomics
  - microbial taxonomy
  - taxonomic classification
  - neural network
authors:
  - name: Andrew J. Tritt
    orcid: 0000-0002-1617-449X
    equal-contrib: false
    affiliation: 1
affiliations:
 - name: Applied Math and Computational Research Division, Lawrence Berkeley National Laboratory, Berkeley, CA, USA
   index: 1
date: 04 March 2024
bibliography: paper.bib
---

# Summary

The field of metagenomics seeks to understand the genomic and functional diversity of microbial
communities. Modern metagenomic sequencing pipelines produce unlabelled genomic sequences at an
unprecedented rate. Processing of these sequences, i.e. contigs, involves labelling the taxonomy of these
contigs. In recent years, the metagenomics fields has been coalescing around the use of the Genome
Taxonomy Database  [@GTDB], a phylogenetically informed taxonomy for consistently labelling microbial taxa.
The Genome Taxonomy Network, `GTNet`, is a neural network capable of classifying metagenomic
contigs with taxonomic labels from the Genome Taxonomy Database.

# Statement of need

`gtnet` [@gtnet] is a Python package and command-line utility built on top of `GTNet`. The purpose of this software
is to make the predictive capabilities of the GTNet easily accessible to the meteagenomics community. 

In addition to deploying GTNet, the `gtnet` software seeks to address other outstanding issues in the
field. Many taxonomic classification tools are still released as source code in tarball formats, require
installation of third-party software that may no longer be maintained, or use application-specific output formats.
These issues make existing tools cumbersome and difficult to use. By leveraging the existing Python ecosystem, we 
seek to make a tool that is easier to use and version for the sake if user-friendliness and reproducibility. 

By releasing easily-installable and user-friendly software capable of generating GTDB taxonomies, we
hope to lower the technical barrier to wide adoption of standardized taxonomy across the metagenomics field.

# References
