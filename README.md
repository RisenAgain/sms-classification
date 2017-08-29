# SMS Classifier for LG

## Setting up environment

Set up python3 virtual env using:

```
virtualenv -p /usr/bin/python3 <envname>
```

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

Use 
```
python sms_class.py -h
```

for help on all the options
For a basic minimal run, do:
```
python sms_class.py --data <path_to_data_files/>
```

Remeber that the data files need to be train.tsv, tune.tsv, test.tsv

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
