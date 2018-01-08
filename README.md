# SMS Classifier for LG

## Setting up environment

Set up python3 virtual env using:

```
virtualenv -p /usr/bin/python3 <envname>
```

example : virtualenv -p /usr/bin/python3 my_env

Use it by:

```
source <envname>/bin/activate
```

Install requirements using:

```
pip install -r requirements.txt
```

NLTK libraries:

```
python -m nltk.downloader 'averaged_perceptron_tagger'
python -m nltk.downloader 'wordnet'
python -m nltk.downloader 'punkt'
```

## Usage

First you'll have to generate dependency-relation files for your dataset.
To generate dependency files:

1. Download Stanford dependency parser from here https://nlp.stanford.edu/software/stanford-parser-full-2017-06-09.zip , if you don't have it already. Unzip it.

2. Set $CLASSPATH environment variable to the path to the Stanford dependency parser.
	export CLASSPATH=path/to/parser/stanford-parser-full-2017-06-09

3. Keep your training and testing files (tab seperated files, tsv) in a folder, say "dataset", with names train.tsv and test.tsv respectively.

	dataset/
		|	train.tsv
		|	test.tsv

4. Install these packages in python3 if not already installed:
	nltk
	pandas
	argparse
	pickle
	copy

5. Run gen_dep_relation.py with python3.
		python3 gen_dep_relation.py --data path/to/dataset/
	Don't forget the ending "/"

6. Dependency files will be stored in the dataset folder itself. Note: It may take long time depending upon the size of dataset.

Use 


```
source <envname>/bin/activate
```


```
python sms_class.py -h
```

for help on all the options
For a basic minimal run, do:
```
python sms_class.py --data <path_to_data_files/>
```

Remeber that the data files need to be train.tsv, tune.tsv, test.tsv. Also don't forget the ending "/" in the path.

Following feature files must be present in the sms_class.py directory for full support:
* fire.words - Fire related words, one word per line
* health.words - health related emergency words, one per line
* personal.words - personal related words, one per line
* wsd_features - allowable wsd synsets, <Word><TAB><stem_word><TAB><Synset>
## Saving features

Use
```
python sms_class.py --data path_to_data_folder/ --to_weka 
```
to save the train, tune, test files in weka format

## Saving and predicting from model

Save model as:
```
python sms_class.py --data path_to_data_foler/ --save
```
The above will generate a '.model' file
Predict from the above model file using:
```
python sms_class.py --data path_to_data_folder/ --model <model_file> --predict
```
