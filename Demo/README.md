## About the demo

The *Demo* folder contains some scripts to work as an example showing how to use the existing codes to deal with your micro-array gene expression dataset and obtain the target gene combination with excellent cancer phenotype classification ability.

The file *individual.py* contains the Individual class of GA individual and file *population.py* is about the Popolation class which implements the GA. *dataset.txt* is an example gene expression dataset for test and *main.py* is the main script for applying the methodology on a specified dataset.
To run the main sript, the python runtime environment and required python packages should be installed and some information about the dataset file is also required. 

## Prerequisites

1. Since the codes in this project are in [Python 3](https://www.python.org/downloads/), so you should make sure the availibility of appropriate runtime enviroment if you want to run the demo directly.
2. Following Python packages should also be installed.
  - [NumPy](http://www.numpy.org/)
  - [SciPy](https://www.scipy.org/)
  - [Pandas](http://pandas.pydata.org/)
  - [minepy](https://pypi.python.org/pypi/minepy)
  - [scikit_learn](http://scikit-learn.org/stable/)
  
## Run the demo

1. Replace the undetermined "xxx" in the main.py with appropriate information.
  * Path for import. In line 31, replace the appended path with the folder path containing the file individual.py and population.py in your PC.
  * Dataset name. In line 286, fill in the dataset name here.
  * Input and output path.
    - In line 291, put the input path of a specified dataset here, and for example, you can just offer the path of the *dataset.txt* file in your PC.
    - In line 292, here you should decide the output path where you'd like to place the output files.
  * Input file information. 
    - In line 294 and 295, fill in the name of positive sample and negtive sample in the input dataset file respectively. For example, in the *dataset.txt* file the two names are `POS` and `NEG` in the first line of the data.
    - In line 296, fill in the name of the gene or probe column, that is, the first string of the first column. For example, in the *dataset.txt* file it is `gene`.
    - In line 298, if dataset is balanced, put 0 here and 1 for imbalanced dataset. This influence the fitness calculation of GA individual, and by default, 1 is OK in all cases.
2. If you like, you can read the annotations about parameters in 3 scripts and adjust them as you wish. If not, just keep them unchanged is also a reliable method.
3. Just run the main.py and wait for the its end. Meanwhile, the real-time output of each GaRFE process and every layer of MGRFE is availible from the console.

## Get the results 
When main script finishes its run, you can get the outputs from the output path you set before. The best GA individuals which are optimal gene combinations in each GaRFE run, every layer of MGRFE and the whole MGRFE process are recorded in the corresponding files. The optimal gene combinations are sorted in a performence descending order accroding to their sizes and related classification metrics, and so the top ones in each file are the best gene combinations found in that stage. It should be mentioned that the list of gene number in each GA individual are the T-test rankings of selected genes and the gene of probe names of them could be found at end of the GaRFE output file.
