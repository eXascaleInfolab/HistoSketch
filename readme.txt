This is the implementation of HistoSketch in MATLAB (see the following paper)

- Yang, Dingqi, Bin Li, Laura Rettig, and Philippe Cudré-Mauroux. "HistoSketch: Fast Similarity-Preserving Sketching of Streaming Histograms with Concept Drift." In ICDM’17.
https://exascale.info/assets/pdf/icdm2017_HistoSketch.pdf (latest version with typo corrected)

How to use (Tested on MATLAB 2014b, 2017a and 2017b):
1. Compile jenkinshash.c using mex: mex jenkinshash.c
2. Run experiment_simulation_abrupt_drift_K.m

Please cite our paper if you publish material using this code.

