# Concept bottleneck models
<p>
  Built using
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-ee4c2c?style=flat&logo=pytorch&logoColor=white" />
  <a href="https://github.com/pyc-team/pytorch_concepts">
    <img src="https://img.shields.io/badge/PyC-990001?style=flat&logo=PyC&logoColor=white">
  </a>
</p>
Concept bottleneck models (CBMs) are deep learning models which are designed to be interpretable and intervenable. Instead of training a model which takes inputs and outputs a label, a CBM comprises of an encoder module which takes inputs and encodes these to a set of human interpretable concepts. From there, these concepts are fed into a predictor module which outputs the final labels. The concept layer can be used to explain which features of an input led to the final prediction, and can be intervened on, or corrected, to improve predictions on a particular input.  <br>


This repo contains example CBM(s) and associated evaluation metrics.  

[See the original CBM paper here](https://proceedings.mlr.press/v119/koh20a.html)
