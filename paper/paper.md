---
title: "HiPart: Hierarchical Divisive Clustering Toolbox"
tags:
  - Python
  - Clustering
  - High dimensionality
  - Machine Learning
authors:
  - name: Panagiotis Anagnostou
    orcid: 0000-0002-4775-9220
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name:  Sotiris Tasoulis
    orcid: 0000-0001-9536-4090
    equal-contrib: true
    affiliation: 1
  - name: Vassilis P. Plagianakos
    orcid: 0000-0002-4266-701X
    equal-contrib: true
    affiliation: 1
  - name: Dimitris Tasoulis
    equal-contrib: true
    affiliation: 2
affiliations:
 - name: Department of Computer Science and Biomedical Informatics, University of Thessaly, Greece
   index: 1
 - name: Signal Ocean SMPC, Greece
   index: 2
date: 8 November 2022
bibliography: paper.bib
---

# Summary

This paper presents the HiPart package, an open-source native Python library that provides efficient and interpret-able implementations of divisive hierarchical clustering algorithms. HiPart supports interactive visualizations for the manipulation of the execution steps allowing the direct intervention of the clustering outcome. This package is highly suited for Big Data applications as the focus has been given to the computational efficiency of the implemented clustering methodologies. The dependencies used are either Python build-in packages or highly maintained stable external packages. The software is provided under the MIT license. The package's source code and documentation can be found at [GitHub](https://github.com/panagiotisanagnostou/HiPart).


# Statement of need

A highly researched problem by a variety of research communities is the problem of clustering. However, high-dimensional clustering still constitutes a significant challenge, plagued by the *curse of dimensionality* [@hutzenthaler2020overcoming]. Hierarchical divisive algorithms developed in the recent years [@TASOULIS20103391; @pavlidis2016minimum; @hofmeyr2016clustering; @hofmeyr2019minimum; @hofmeyr2019ppci] have shown great potential for the particular case of high dimensional data,
incorporating dimensionality reduction iteratively within their algorithmic procedure. Additionally, they seem unique in providing a hierarchical format of the clustering result with low computational cost, in contrast to the commonly used but computationally demanding agglomerative clustering methods.

Although the discovery of a hierarchical format is crucial in many fields, such as bioinformatics [@luo2003hierarchical; @modena2014gene], to the best of our knowledge, this package is the first native Python implementation of divisive hierarchical clustering algorithms. We particularly focus on the "Principal Direction Divisive Clustering (PDDP)" algorithm [@boley1998principal] for its potential to effectively tackle the *curse of dimensionality* and its impeccable time performance [@TASOULIS20103391].

Simultaneously, we provide implementations of a complete set of hierarchical divisive clustering algorithms with a similar basis. These are the dePDDP [@TASOULIS20103391], the iPDDP [@TASOULIS20103391], the kM-PDDP [@zeimpekis2008principal], and the bisecting k-Means (BKM) [@savaresi2001performance]. We also provide additional features not included in the original developments of the aforementioned methodologies that make them appropriate for the discovery of arbitrary shaped or non-linear separable clusters. In detail, we incorporate kernel Principal Component Analysis (kPCA) [@Scholkopf99kernelprincipal] and Independent Component Analysis (ICA) [@hyvarinen2000independent; @tharwat2020independent] for the iterative dimensionality reduction steps.

As a result, the package provides a fully parameterized set of algorithms that can be applied in a diverse set of applications, for example, non-linear separable clusters, automated identification for the cluster number, and outlier control.


# Software Description

The HiPart (Hierarchical Partitioning) package is divided into three major sections:

  - Method implementation
  - Static Visualization
  - Interactive Visualization


## Method Implementation

The package employs an object-oriented approach for the implementation of the algorithms, similarly to that of [@JMLR:v23:21-0862], while incorporating design similarities with the scikit-learn library [@pedregosa2011scikit]. Meaning, a class instance executes each of the algorithms, and the class instance's attributes are the algorithm's hyper-parameters and results.

For the execution of the algorithms, the user needs to call either the `predict` or the `fit_predict` method of each algorithm's execution class. The algorithm parameterization can be applied at the constructor of their respective class.


## Static Visualization

Two static visualization methods are included. The first one is a 2-Dimensional representation of all the data splits generated by each algorithm during the hierarchical procedure. The goal is to provide an insight to the user regarding each node of the clustering tree and, subsequently, each step of the algorithm's execution.

The second visualization method is a dendrogram that represents the splits of all the divisive algorithms. The dendrogram's figure creation is implemented by the *SciPy* package, and it is fully parameterized as stated in the library.


## Interactive Visualization

In the interactive mode, we provide the possibility for stepwise manipulation of the algorithms. The user can choose a particular step (node of the tree) and manipulate the split-point on top of a two-dimensional visualization, instantly altering the clustering result. Each manipulation resets the algorithm's execution from that step onwards, resulting in a restructuring of the sub-tree of the manipulated node.


# Development Notes

For the development of the package, we complied with the **PEP8** style standards, and we enforced it with the employment of *flake8* command-line utility. To ensure the code's quality, we implemented tests using the *pytest* module to assert the correctness of the implementations. In addition, platform compatibility has been assured through extensive testing, and the package development in its entirety uses only well-established or native Python packages. The package has been released as
open-source software under the MIT License.  For more information regarding potential contributions or for the submission of an issue, or a request, the package is hosted as a repository on Github.


# Experiments and Comparisons

In this section, we provide clustering results with respect to the execution speed and clustering performance for the provided implementations. For direct comparison, we employ a series of well-established clustering algorithms. These are the k-Means [@likas2003global], the Agglomerative (AGG) [@ackermann2014analysis] and the OPTICS [@ankerst1999optics] of the scikit-learn [@pedregosa2011scikit] Python library and the fuzzy c-means (FCM) algorithm [@bezdek1984fuzzy] of the fuzzy-c-means [@dias2019fuzzy] Python package. Clustering performance is evaluated using the Normalized Mutual Information (NMI) score [@yang2016comparative].

Four widely used data sets from the field of bioinformatics are employed along with two popular data sets benchmark data set for text and image clustering, respectively:

```{=html}
<style>
#data>table>tbody, #data>table>tbody>tr>td {
  border-style: none;
}
</style>
```

:::{#data}

-   the Deng [@DengData],
-   the Baron [@baron2016single],
-   the TGCA Pan-cancer[^1] (Cancer),
-   the Chen [@chen2017single],
-   the USPS [@291440],
-   the BBC [@greene06icml],
:::
[^1]: https://www.doi.org/10.7303/syn300013

::: myimg
![Dendrogram figure for the Cancer data set with the use of the dePDDP algorithm and the dendrogram visualization module of the HiPart library. The line below the tree represents the colour of the original cluster each sample belongs.](dendrogram.pdf)
:::

All experiments took place on a server computer with Linux operating system, kernel version 5.11.0, with an Intel Core i7-10700K CPU \@ 3.80GHz and four DDR4 RAM dims of 32GB with 2133MHz frequency. Default parameters were used for the execution of all the algorithms, and the actual number of clusters was provided to algorithms as a parameter when required.

In [Table 1](#mytable)we present the mean performance of all methods with respect to execution time (time in secs) and NMI across 100 experiments. We observe that HiPart implementations perform exquisitely in terms of execution time while still being comparable with respect to clustering performance.

```{=html}
<style>
#floats {
  width: 100%;
  font-size: 0.645em
}
#floats::after {
  content: "";
  clear: both;
  display: table;
}
#mytable {
  display: inline;
  float: left;
  width: 47%;
  max-width: 47%;
}
#mytable>table>tbody>tr>td {
  margin: 0pt;
  padding-right: 1pt;
  padding-left: 1pt;
  padding-top: 0.02rem;
  padding-bottom: 0.02rem;
}

.myimg {
  display: inline;
  position: realativer;
  right: 0px;
  float: left;
  width: 52%;
  padding: 3.5% 0% 0% 2%;
}

.myimg>div>figure>p>img {
  width: 100%
}

.mycaption {
  font-size: 0.875rem;
  color: #6c757d;
}
</style>
```
::: {#floats}
::: {#mytable}
+------------+--------------------------+----------+-------------------------+-----------+
| Algorithm  | time (seconds)           | NMI      | time (seconds)          | NMI       |
+============+:========================:+:========:+:=======================:+:=========:+
|            | **Gene Expression Data**                                                  |
+------------+--------------------------+----------+-------------------------+-----------+
|            | **Deng (135, 12548)**               | **Baron (1886, 14878)**             |
+------------+--------------------------+----------+-------------------------+-----------+
| iPDDP      | 0.10                     | 0.76     | 1.16                    | 0.08      |
+------------+--------------------------+----------+-------------------------+-----------+
| dePDDP     | 0.14                     | 0.70     | 2.16                    | 0.53      |
+------------+--------------------------+----------+-------------------------+-----------+
| PDDP       | 0.15                     | 0.54     | 2.55                    | 0.53      |
+------------+--------------------------+----------+-------------------------+-----------+
| kM-PDDP    | 0.25                     | 0.61     | 3.81                    | 0.52      |
+------------+--------------------------+----------+-------------------------+-----------+
| BKM        | 0.50                     | 0.64     | 11.51                   | 0.52      |
+------------+--------------------------+----------+-------------------------+-----------+
| k-Means    | 0.14                     | 0.71     | 7.52                    | 0.48      |
+------------+--------------------------+----------+-------------------------+-----------+
| AGG        | 0.04                     | 0.72     | 14.54                   | 0.51      |
+------------+--------------------------+----------+-------------------------+-----------+
| OPTICS     | 27.78                    | 0.48     | 710.99                  | 0.13      |
+------------+--------------------------+----------+-------------------------+-----------+
| FCM        | 1.37                     | 0.68     | 163.63                  | 0.45      |
+------------+--------------------------+----------+-------------------------+-----------+
|            | **Cancer (801, 20531)**             | **Chen (14437, 23284)**             |
+------------+--------------------------+----------+-------------------------+-----------+
| iPDDP      | 0.90                     | 0.67     | 13.18                   | 0.30      |
+------------+--------------------------+----------+-------------------------+-----------+
| dePDDP     | 1.06                     | 0.93     | 20.71                   | 0.36      |
+------------+--------------------------+----------+-------------------------+-----------+
| PDDP       | 1.04                     | 0.74     | 37.52                   | 0.48      |
+------------+--------------------------+----------+-------------------------+-----------+
| kM-PDDP    | 1.27                     | 0.86     | 53.73                   | 0.48      |
+------------+--------------------------+----------+-------------------------+-----------+
| BKM        | 5.94                     | 0.88     | 255.85                  | 0.48      |
+------------+--------------------------+----------+-------------------------+-----------+
| k-Means    | 1.54                     | 0.98     | 249.72                  | 0.48      |
+------------+--------------------------+----------+-------------------------+-----------+
| AGG        | 3.09                     | 0.98     | 1218.68                 | 0.49      |
+------------+--------------------------+----------+-------------------------+-----------+
| OPTICS     | 266.39                   | 0.34     | 27089.35                | 0.00      |
+------------+--------------------------+----------+-------------------------+-----------+
| FCM        | 5.53                     | 0.53     | 5710.83                 | 0.26      |
+------------+--------------------------+----------+-------------------------+-----------+
|            | **Benchmark Data**                                                        |
+------------+--------------------------+----------+-------------------------+-----------+
|            | **USPS (4575, 256)**                | **BBC (2225, 21213)**               |
+------------+--------------------------+----------+-------------------------+-----------+
| iPDDP      | 0.02                     | 0.55     | 2.02                    | 0.60      |
+------------+--------------------------+----------+-------------------------+-----------+
| dePDDP     | 0.05                     | 0.65     | 1.70                    | 0.60      |
+------------+--------------------------+----------+-------------------------+-----------+
| PDDP       | 0.04                     | 0.60     | 2.57                    | 0.78      |
+------------+--------------------------+----------+-------------------------+-----------+
| kM-PDDP    | 0.17                     | 0.50     | 2.93                    | 0.74      |
+------------+--------------------------+----------+-------------------------+-----------+
| BKM        | 0.38                     | 0.58     | 9.92                    | 0.65      |
+------------+--------------------------+----------+-------------------------+-----------+
| k-Means    | 0.12                     | 0.72     | 4.68                    | 0.35      |
+------------+--------------------------+----------+-------------------------+-----------+
| AGG        | 1.15                     | 0.77     | 23.18                   | 0.64      |
+------------+--------------------------+----------+-------------------------+-----------+
| OPTICS     | 635.05                   | 0.04     | 1055.75                 | 0.06      |
+------------+--------------------------+----------+-------------------------+-----------+
| FCM        | 2.92                     | 0.58     | 4.73                    | 0.60      |
+------------+--------------------------+----------+-------------------------+-----------+
Table: Clustering results with respect to execution time and clustering performance.
:::

:::


# Conclusions and Future Work

We present a highly time-efficient clustering package with a suite of tools that give the capability of addressing problems in high-dimensional clustering problems. Also, the developed new visualization tools enhance understanding and identification of the underlying clustering data structure.

We plan to continuously expand the HiPart package in the future through the addition of more hierarchical algorithms and by providing even more options for dimensionality reduction, such as the use of recent projection pursuit methodologies [@pavlidis2016minimum; @hofmeyr2016clustering; @hofmeyr2019minimum; @hofmeyr2019ppci]. Our final aim is to establish the golden standard when considering hierarchical divisive clustering.


# Acknowledgments

This project has received funding from the Hellenic Foundation for Research and Innovation (HFRI), under grant agreement No 1901.

# References