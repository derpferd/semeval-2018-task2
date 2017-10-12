# semeval-2018-task2
Author: Jonathan Beaulieu

## Task
You can find the problem statement with details at [https://competitions.codalab.org/competitions/17344](https://competitions.codalab.org/competitions/17344).
This SemEval task is focused on emoji use in tweets. The task is to create a program which given a set of tweet text (without emojis) and the emoji which was used inside the tweet text can predict which emoji will be used in any tweet given the text of the tweet. For this task we are not allowed to used outside tweet data, for example grabbing more tweets to use in the training data. The only outside data we are allowed to use are emoji embeddings. They are using the Macro-F scores instead Micro-F scores to discourage overfitting to the most frequent classes. Plus we train and are evaluated on the 20 most frequent emojis. This task has two subtasks: running against English tweets and running against Spanish tweets.

### Example input and output

| Input                                                            | Output |
|:-----------------------------------------------------------------|:-------|
| A little throwback with my favourite person @ Water Wall         | ❤      |
| Birthday Kisses @ Madison, Wisconsin                             | 😘     |
| Everything about this weekend #hogvibes @ Fayetteville, Arkansas | 💯     |

## Setup
Simply run `./install.sh` to install all dependencies and do all required setup to get up and running on a clean install of **Ubuntu 16.04**. *Note: this script should work on other versions and linux systems but it has not been tested.*

## Running
Simply run `./runit.sh` to run and see the results of the program. `runit.sh` will train a model for both the English and Spanish train data sets and print out the results of each fold in 10 cross folds.

## System code
The code for the **Hopper** system can be found under the `hopper` directory along with all accompanying documentation. All the models are described in detail in the [readme](hopper/README.md) in that directory.

## Contributions
Each file lists the author(s) at the top of the file. In the case of multiple authors it is clearly listed the sections each author contributed (we tried keeping this to a minimum).

**Jonathan:**
- Wrote the framework code. The code for reading the data, running the model and analyzing the results(expect for the scoring which was provided by the task organizers).
- Wrote the code for the Naive Bayes baselines.
- Wrote the code for the Random Model.
- Wrote this README and the README sections on Bernoulli Naive Bayes baselines.

**Dennis:**
- Most Frequent Class Model
- Documentation for Most Frequent Class Model in README and code
- Documentation for NaiveBayes Model in code
- RESULTS-1 and RESULTS-1.md

**Sai:**
- Analyzed the Output and Results which can be found in Results1_1
- Explained 3rd party code in ORIGINS

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
This is a todo list of everything we need for Stage 1 submission that hasn't been done already.
- [x] Write `install.sh` (Jon)
- [x] Analyze results. (Sai) Due: 10/6
- [x] Create baseline based on most frequent class (Dennis) Due: 10/9
- [x] Write readme section about BernoulliNB Model. (Jon) Due: 10/9
- [x] Write readme section about most frequent class baseline model. (Dennis) Due: 10/10
- [x] Clarify Authorship (All)
- [x] Final review, review each others work. (All) Due: 10/12
- [x] Example of actual program input and output and description of problem. (Jon)

## Stage 2 TODO
- [ ] Check out Twitter embeddings at [https://github.com/fvancesco/acmmm2016](https://github.com/fvancesco/acmmm2016)
- [ ] Try normalization.
