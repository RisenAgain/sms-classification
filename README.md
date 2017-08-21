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
