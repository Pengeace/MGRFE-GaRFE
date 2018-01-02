## About this repository

Here is about "MGRFE: multilayer recursive feature elimination based on embedded genetic algorithm for cancer classification". The detailed information are available at the file *MGRFE.pdf*.

In this repository, you can find all 19 datasets for cancer classification used by MGRFE, our source code and the generated results.

## About MGRFE-GaRFE

We previously proposed MGRFE, a novel multilayer recursive feature elimination algorithm based on embedded variable length encoding genetic algorithm, which aims at selecting minimal discriminatory genes associated closely with the phenotypes in macro-array gene datasets. The work combined the evolutionary calculating of embedded genetic algorithm and explicit feature decline of recursive feature elimination as GaRFE, which is taken as the feature selection unit at each layer of MGRFE.

The mostly used total 19 benchmark micro-array datasets including multi-class and imbalanced datasets are divided into two large datasets (e.g., the *Dataset One* and *Dataset Two*) and used to validate the proposed method and make a comprehensive comparison with other popular feature selection methods for cancer classification. Many promising results were obtained by MGRFE on these datasets. MGRFE can reaches *Acc* 100% within just 5 genes on 10 (52.6%) of 19 datasets, and *Acc* higher than 90% within 10 genes on all 19 datasets. MGRFE also shows the robustness for multi-class datasets and imbalanced datasets according to *Sn*, *Sp*, *Avc*, and *MCC* metrics. Based on classification performance comparison with other 20 methods on the two large Datasets, our proposed method MGRFE is proved to be more superior than most of current popular feature selection methods for achieving better classification accuracy with smaller gene size.

Furthermore, the biological function analysis using literature mining for predicted bio-markers confirmed that the selected genes by MGRFE are biologically relevant to cancer phenotypes. 

MGRFE can represent a complementary feature selection algorithm for high-dimensional bio-data analysis and is significant for cancer diagnosis and further biomedical research.

## How to run

1. The codes in this project are in [Python](https://www.python.org/downloads/) 3.6. And following related python packages are also depended and should be installed.
  - [NumPy](http://www.numpy.org/)
  - [SciPy](https://www.scipy.org/)
  - [Pandas](http://pandas.pydata.org/)
  - [minepy](https://pypi.python.org/pypi/minepy)
  - [scikit_learn](http://scikit-learn.org/stable/)
2. Currently, you should edit the paths (e.g., the project path `D:\\codes\\python\\MGRFE\\` ) in the scripts based on your settings.
3. An example showing how to use the existing codes to deal with your own micro-array gene expression data set is available at the *Demo* folder.
4. Note that thorough comments are added in the scripts to improve readability and may offer you an better understanding about the algorithm and the implementation.

## Help

For any matters, just feel free to contact chengpengeace@gmail.com or start an issue instead.
