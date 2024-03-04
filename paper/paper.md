---
title: 'gtnet: A Python package for taxonomic labelling with the Genome Taxonomy Network'
tags:
  - Python
  - metagenomics
  - deep learning
authors:
  - name: Andrew J. Tritt
    orcid: 0000-0002-1617-449X
    equal-contrib: false
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Kristofer Bouchard
    orcid: 0000-0002-1974-4603
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
 - name: Applied Math and Computational Research Division, Lawrence Berkeley National Laboratory, Berkeley, CA, USA
   index: 1
 - name: Scientific Data Division, Lawrence Berkeley National Laboratory, Berkeley, CA, USA.
   index: 2
 - name: Biological Systems and Engineering Division, Lawrence Berkeley National Laboratory, Berkeley, CA, USA
   index: 3
 - name: Helen Wills Neuroscience Institute and Redwood Center for Theoretical Neuroscience, University of California Berkeley, Berkeley, CA, USA.
   index: 4
date: 13 August 2017
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

`gtnet` [@gtnet] is a Python package and command-line utility built on top of `GTNet`. The purpose of thise software
is to make the predictive capabilities of the GTNet easily accesible to the meteagenomics community. 

In addition to deploying GTNet, the `gtnet` software seeks to address other outstanding issues in the
field. Many taxonomic classification tools are still released as source code in tarball formats, require
installation of third-party software that may no longer be maintained, or use application-specific output formats.
These issues make existing tools cumbersome and difficult to use. By leveraging the existing Python ecosystem, we 
seek to make a tool that is easier to use and version for the sake if user-friendliness and reproducibility. 

By releasing easily-installable and user-friendly software capable of generating GTDB taxonomies, we
hope to lower the technical barrier to wide adoption of standardized taxonomy across the metagenomics field.

# References
