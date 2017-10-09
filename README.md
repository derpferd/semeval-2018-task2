# semeval-2018-task2

You can find the problem statement with details at [https://competitions.codalab.org/competitions/17344](https://competitions.codalab.org/competitions/17344).

## Setup
Simply run `./install.sh` to install all dependencies and do all required setup to get up and running on a clean install of **Ubuntu 16.04**. *Note: this script should work on other versions and linux systems but it has not been tested.*

## Running
Simply run `./runit.sh` to run and see the results of the program. `runit.sh` will train a model for both the English and Spanish train data sets and print out the results of each fold in 10 cross folds.

## System code
The code for the **Hopper** system can be found under the `hopper` directory along with all accompanying documentation. All the models are described in detail in the [readme](hopper/README.md) in that directory.

## Dependencies
- python >= 3.5
- nltk == 3.2.4
- scikit-learn == 0.19.0
- sklearn == 0.0
- scipy == 0.19.1

This project requires Python 3.  
The python module dependencies can be found in `requirements.txt`.  
Install them by running `pip install -r requirements.txt`. (**Note**: You may need to replace `pip` with `pip3` if python 3 is not your default.)

## Data

All the data can be found under the `data` directory.
- The mapping of numbers to Emojis are in the mappings directory.
- The trail data is in the trial directory.
  - This is ~50k tweets with labels.
  - The tweets are separated into English and Spanish
- The train data is in the train directory.
  - This is ~500k tweets with labels.
  - The tweets are separated into English and Spanish using the provided script. However upon reading the script it separates the tweets based on the emoji used instead on the language used in the tweet.

## Output

All the output for the system is stored in `output` directory.
Contents
 - `stage_1_train.out.txt`
   - Baseline Models trained and tested on the train data
     - Tested using 10 cross folds
 - `stage_1_trial.out.txt`
   - Baseline Models trained and tested on the trial data
     - Tested using 10 cross folds

## Stage 1 TODO (everything should be done by Wednesday night)
This is a todo list of everything we need for Stage 1 submission.
- [ ] Write `install.sh` (Not Assigned)
- [ ] Analyze results. (Sai) Due: 10/6
- [ ] Create baseline based on most frequent class (Dennis) Due: 10/9
- [ ] Write readme section about BernoulliNB Model. (Jon) Due: 10/9
- [ ] Write readme section about most frequent class baseline model. (Dennis) Due: 10/10
- [ ] Clarify Authorship (All)
- [ ] Final review, review each others work. (All) Due: 10/11

## Stage 2 TODO
- [ ] Check out Twitter embeddings at [https://github.com/fvancesco/acmmm2016](https://github.com/fvancesco/acmmm2016)
- [ ] Try normalization.
