---
title: 'DSSE: An environment for simulation of drone swarm maritime search and rescue missions'
tags:
  - Python
  - PettingZoo
  - reinforcement learning
  - multi-agent
  - drone swarms
  - maritime search and rescue
  - shipwrecked people
authors:
  - name: Renato Laffranchi Falcão
    orcid: 0009-0001-5943-0481
    corresponding: true
    equal-contrib: true
    affiliation: 1
  - name: Jorás
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
  - name: Pedro
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
  - name: Ricardo
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
  - name: Fabrício Jailson Barth
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
  - name: José Fernando
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 2
affiliations:
 - name: Insper, Brazil
   index: 1
 - name: Embraer, Brazil
   index: 2
date: XX April 2024
bibliography: paper.bib

---

# Summary



The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration).

# Statement of need

Maritime navigation plays a crucial role across various domains, including leisure activities and commercial fishing. However, maritime transportation is particularly significant as it accounts for 80% to 90% of global trade [@allianz]. Therefore, maritime safety is essential, demanding significant enhancements in search and rescue (SAR) missions. It is imperative that SAR missions minimize the search area and maximize the chances of locating the search object.

To achieve this objective, traditional SAR operations used methods
such as parallel track, crawl line, extended square, and sector searches
(IAMSAR, 2016; Koopman, 1957). Recently, there has been a surge in
research aimed at enhancing traditional search methods. 
Ramirez et al.(2011) = https://ieeexplore.ieee.org/document/6003509

`DSSE` is a Python package that provides a simulation environment using the PettingZoo interface with the purpose of training and evaluating single or multi-agent reinforcement learning algorithms. The API was designed to contribute to researches on the effectiveness of integrating reinforcement learning techniques into SAR path planning.

The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References