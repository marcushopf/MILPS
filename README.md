# MILPS
## About
In this repository, I provide data and code to run <b>M</b>ultiple <b>I</b>nstance <b>L</b>earning with <b>P</b>artly <b>S</b>upervision (<b>MILPS</b>), sometimes in the code also referred to as MIL<b>LDI</b> (= MIL with <b>L</b>ocal <b>D</b>ata <b>I</b>njection). The underlying attention mechanism used for MILPS is based on the model MILNET by Stefanos Angelidis and Mirella Lapata (2018) Multiple Instance Learning Networks for Fine-Grained Sentiment Analysis, Transactions of the Association for Computational Linguistics, 6:17–31 (http://dx.doi.org/10.1162/tacl_a_00002).

MILPS requires two steps:
-	data preprocessing for MILPS (cf. “Data_preprocessing.ipynb”)
-	apply MILPS (cf. “Controller_MILPS.ipynb”)

## Utilizing MILPS
For executing the scripts, I used ipynb files on Google Colab together with data on Google drive.

Data_preprocessing.ipynb
-	input: data_input_milldi.json is the raw data in the current working directory (cwd)
-	output: data_train_0.7.bin, data_dev_0.15.bin, data_test_0.15.bin in cwd+"MIL data"

Controller_MILPS.ipynb
-	input: data_train_0.7.bin, data_dev_0.15.bin, data_test_0.15.bin in cwd+"MIL data"
-	output: fine-grained sentiment classification for sentences of data_dev_0.15.bin, data_test_0.15.bin & evaluation results saved in

## Highlights
py_scripts/MIL_nn_utils.py: 
-	definition of MILRestsDataset, which is a custom child class of the PyTorch class "Dataset" that enables MIL

Data_preprocessing.ipynb:
-	tokenization with BertTokenizer from the transformers framework
-	estimation of maximum sequence length for sentences and reviews
-	inspection and verification of data structure of MILRestsDataset

Controller_MILPS.ipynb:
-	utilization of PyTorch for building and evaluation (training and testing) MILPS
-	BERT-based sentence encoder for MIL
-	Attention-based MIL for aggregation of sentence level sentiments to a review level sentiment (based on MILNET by Angelidis & Lapata (2018))
-	Extension of MIL with Partly fine-grained supervision on sentence level sentiment labels (i.e., MILPS)
