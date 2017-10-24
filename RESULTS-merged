
This file has the analysis of results for our stage 1 model. 

Here we also talk about the emoji classes used, definitions and formulas which help in understanding of the results. 

Team: Hopper (Jon, Dennis, Sai)

Task: Multilingual Emoji Prediction (English and Spanish)

Baseline: We implemented two baselines. One was a Bag-of-words model using a Bernoulli Naive Bayes classifier. 
The other was a Most Frequent Class model. 


*****************************Emoji classes*************************

Top 20 Emojis used are collected based on the geographical data and are divided into 20 classes.

English Emoji Classes:												

0	❤	_red_heart_	
1	😍	_smiling_face_with_hearteyes_	
2	😂	_face_with_tears_of_joy_	
3	💕	_two_hearts_	
4	🔥	_fire_	
5	😊	_smiling_face_with_smiling_eyes_	
6	😎	_smiling_face_with_sunglasses_	
7	✨	_sparkles_	
8	💙	_blue_heart_	
9	😘	_face_blowing_a_kiss_	
10	📷	_camera_	
11	🇺🇸	_United_States_	
12	☀	_sun_	
13	💜	_purple_heart_	
14	😉	_winking_face_	
15	💯	_hundred_points_	
16	😁	_beaming_face_with_smiling_eyes_	
17	🎄	_Christmas_tree_	
18	📸	_camera_with_flash_	
19	😜	_winking_face_with_tongue_	

Spanish Emoji Classes:

0	❤	_red_heart_	
1	😍	_smiling_face_with_hearteyes_	
2	😂	_face_with_tears_of_joy_	
3	💕	_two_hearts_	
4	😊	_smiling_face_with_smiling_eyes_	
5	😘	_face_blowing_a_kiss_	
6	💪	_flexed_biceps_	
7	😉	_winking_face_	
8	👌	_OK_hand_	
9	🇪🇸	_Spain_	
10	😎	_smiling_face_with_sunglasses_	
11	💙	_blue_heart_	
12	💜	_purple_heart_	
13	😜	_winking_face_with_tongue_	
14	💞	_revolving_hearts_	
15	✨	_sparkles_	
16	🎶	_musical_notes_	
17	💘	_heart_with_arrow_	
18	😁	_beaming_face_with_smiling_eyes_	
19	🔝	_TOP_arrow_	


Total data provided for the task : 500k english tweets and 100k spanish tweets. Out of this, 10% data is used for testing and 10% is used as trial data. The rest 80% of the data is used to train the model.

The model is trained using the 10 fold cross validation.


The output of the model contains the following

1)Confusion Matrix 
2)Micro F score 
3)Macro F score
4)Precison
5)Recall

******************************Definitions****************************

Confusion Matrix: It is used to evaluate the performance of the classifier. In this confusion matrix, the Gold Emoji class is on the Y-axis and the Predicted Emoji Class is on X-Axis.
True positives for each classes are the along the diagonal of the matrix. The False Negatives for a particular class are along the respective row and False Positives are accross the columns.

For example, if we consider the class 3 from the confusion matrix 
   
3:   26  793  273  751(TP)   78  124   90   81   78  194   36   61   11   10   23   24   12   38   21    2   
                            ********************** FALSE NEGATIVES***************
751 is the True Positive cause the predicted value was 3 and gold emoji value is also 3. All the other elements in the row are False Negatives cause even though their emoji value was 3, they were not predicted as 3.

	247  *
	389  F
	128  A       
(TP)->	751  L
	64   S
	208  E
	78   P  
	120  O
	314  S
	251  I
	15   T
	29   I
	1    V
	258  E
	78   S
	65   *
	55   *
	27   *
	27   *
	86   *

The above values are False Positives because they were predicted as class 3 but they are not from class 3.

Precision: Precision is the ratio of True Positives(TP) to the sum of True Positives(TP) plus False Positives(FP).It measures the quality of predictions, based on the all the positive values .
 Precision = TP/TP+FP
          
In our model, Precision of a particular class would be the ratio of the correct emojis predicted to the attempted emojis.  
 P = Correct predictions / Attempted Predictions (from line 46 of scorer_semeval18.py )

Attempted predictions are all the values that are classified as positive by classifier irrespective of being true or false(TP+FP) 

 Example: In class 3, Value of prediction is
   P = 751(correct predictons)/ 3191(attempted predictions)

Recall: Recall is the ratio of True Positives(TP) to the sum of True Positives(TP) plus False Negatives(FN).It measures how precise the output is.
 Recall = TP/TP+FN

For our model, Recall of a particular would be the ratio to the correct predictions to the gold values in the class.
 R = Correct Predictions/ Gold Occurances (from line 47 of scorer_semeval18.py)

 Gold occurences are the gold emojis for the respective class present in the test data. 
 
 Example: In class 3, Value of recall is 
 R = 751 / 2726 


Note: The above precision and recall examples are limited to a single class, precision and recall of the 20 classes 
      will have correct predictions as True Postives, False Positives and False Negatives from all the 20 classes. (from lines 49-56 in scorer_semeval18.py)
      

Micro F score: Micro F score is the harmonic mean of the precision and recall.

 Microf1 = (2 * (P*R))/P+R    ( Where P is precision and R is recall.)

Macro F score: It is the weighed average of the total f1 score to the number of emoji classes in the data. 

 Macrof1 = f1total/No. of emoji classes  (Where f1total is the sum of all the individual f1 scores)


Precision and Recall are equal when the values of False Positives and False Negatives are equal i.e. when attempted_total is equal to gold_occurrences_total.(See scorer_semeval18.py lines 52 and 53)

********************** Results **********************************
The results of both models are shown below. Detailed analysis is, however, performed only on the Bernoulli model 
since that is what we have chosen for our baseline.


Results:
## Most Frequent Class Model Results

### Results for English Tweets
Macro F-score Range: 1.763 - 1.8 
Micro F-score Range: 21.408 - 21.954
Precision Range: 21.408 - 21.954
Recall: 21.408 - 21.954

### Results for Spanish Tweets
Macro F-score Range: 3.327 - 3.375
Micro F-score Range: 27.59 - 28.107
Precision Range: 27.59 - 28.107
Recall: 27.59 - 28.107


## Bag of Words Bernoulli Model Results

### Results for English Tweets
Macro F-score Range: 28.788 - 29.501
Micro F-score Range: 41.534 - 42.591
Precision Range: 41.534 - 42.591
Recall: 41.534 - 42.591

### Results for Spanish Tweets
Macro F-score Range: 25.48 - 26.304
Micro F-score Range: 45.543 - 46.683
Precision Range: 45.543 - 46.683
Recall: 45.543 - 46.683

******************************** Output Analysis***********************

It is worth noting that our Micro F-score, Precision and Recall have the same values for each set of results. For instance, under Results for English Tweets for Most Frequent Class Model, the values range from 21.408 - 21.954 for all the three scores. This shows that all our models assigned classes (emojis) for all tweets that were in the test set. From the scorer, precision and recall are equal when attempted_total is equal to gold_occurrences_total. (See scorer_semeval18.py lines 52 and 53)

For the English tweets, the difference in the Macro and micro F-scores for our Bernoulli model lies in the fact that about 21% of tweets have label 0 (representing ❤ - the red heart) which is the most frequent class. This can be seen from the results of our Most Frequent Class model which has a precision and recall of about 21%. A similar thing happens with the Spanish tweets where about 28% of the tweets belong to the same red heart class. Note the correlation between the relatively higher percentage of tweets in the most frequent class and the relatively higher Micro F-score for the Spanish tweets. However, once the effect of the most frequent class is offsetted as measured by the Macro F-score, the prediction for the English tweets becomes better than that of the Spanish tweets. 7 classes are missing from both the training data and testing data for Spanish. Another important difference between the Spanish and English is that there are about 100,000 more tweets in the English training data. This, likely, contributes to English's better Macro F-scores.

Analysis of Output for English Data:

Sample Output:

Macro F-Score (official): 29.061
-----
Micro F-Score: 42.242
Precision: 42.242
Recall: 42.242

----- Details -----
Training data len: 445170 Testing data len: 49463
Training data class counts: 0: 96651, 1: 46630, 2: 45710, 3: 24729, 4: 22159, 5: 20921, 6: 19180, 7: 16517, 8: 15273, 9: 14606, 10: 14423, 11: 13823, 12: 12468, 13: 11654, 14: 12132, 15: 12041, 16: 11769, 17: 11527, 18: 11928, 19: 11029
Testing  data class counts: 0: 10824, 1: 5202, 2: 5149, 3: 2726, 4: 2416, 5: 2348, 6: 2093, 7: 1835, 8: 1726, 9: 1524, 10: 1604, 11: 1513, 12: 1388, 13: 1301, 14: 1316, 15: 1343, 16: 1304, 17: 1298, 18: 1303, 19: 1250
--- Matrix ---
 0: 8775  583  248  247   60   76   58   44   44   97   19   65  382    0   17   12   12   72   10    3   ∑ = 10824
 1:   29 2428  800  389  190  162  226  138  103  184   81  179   20    8   38   17   66  101   40    3   ∑ =  5202
 2:    6 1015 2812  128  212  119  211   70   50   64   59  133    3    3   54   85   49   42   31    3   ∑ =  5149
 3:   26  793  273  751   78  124   90   81   78  194   36   61   11   10   23   24   12   38   21    2   ∑ =  2726
 4:    2  476  347   64 1055   43   99   37   36   21   41   48    7    1   15   55    9   25   34    1   ∑ =  2416
 5:   10  667  481  208   99  213  137   42   55  125   42   72    8    6   44   14   55   53   13    4   ∑ =  2348
 6:    5  547  434   78  170   80  311   46   48   41   43  114   15    5   28   44   30   19   29    6   ∑ =  2093
 7:    4  489  249  120  134   52  104  316   20   49   53   81    2    4   26   26    7   62   35    2   ∑ =  1835
 8:   22  487  193  314   69   54   85   60  203   62   18   86    8    9   13   10    7   20    6    0   ∑ =  1726
 9:   10  390  233  251   55   70   68   15   35  254   18   49    2    1   18   14   13   23    4    1   ∑ =  1524
10:    1  257  183   15   89   38   38   26    9    5  694   55    2    0   10    7    6   12  156    1   ∑ =  1604
11:    3  233   96   29   41   35   69   31   20   15   14  865    4    2    8   12    8   17   10    1   ∑ =  1513
12:  577    9    5    1    1    3    5    4    2    1    0    5  771    0    0    0    0    0    4    0   ∑ =  1388
13:   11  365  176  258   44   49   45   34   68   92   12   26    0   72    7    4    8   20    8    2   ∑ =  1301
14:    2  282  438   78   75   63   62   24   42   45   20   43    2    0   60   10   30   20   16    4   ∑ =  1316
15:    7  301  349   65  131   32   92   20   32   24   28   42    0    3   13  166    7   13   17    1   ∑ =  1343
16:    4  362  329   55   76   97   76   21   28   45   21   57    2    4   27    4   58   19   15    4   ∑ =  1304
17:    4  114   62   27   26   24   19   27    6   14   14   23    0    0    6    1    7  918    4    2   ∑ =  1298
18:    2  207  179   27   88   29   34   23   11   12  451   42    0    0    8   13    4    9  163    1   ∑ =  1303
19:    1  315  369   86   67   49   82   25   44   32   24   24    4    2   39   10   33   18   17    9   ∑ =  1250


All the elements in the diagonal of a confusion matrix are correctly matched with their gold values, i.e. they are the true positives.

For the class 0, out of 10824 samples 8775 emojis were predicted accurately and all the other cases are the false negatives. 
Similarly for the class 1, out of 5202 samples, 2428 emojis were accurate predictions. 

Some classes have more False Negatives than the accurate values, this is because the difference in the emojis is minimal and it is hard for the 
model to differentiate between them. 

For Example, Consider class 5, it represents the emoji smiling face with smiling eyes which is very much similar to class 1(smiling face with hearteyes). Mislabelling 5 as 1 is quite understandable since they are both
smiling faces.The model cannot differentiate between these two and predicts most of them as smiling face with hearteyes(Class 1).

2313 tweets were assigned in the gold. We assigned only 251 correctly. 
653 were incorrectly assigned as label 1 (😍 _smiling_face_with_hearteyes_) and 437 were 
incorrectly assigned as label 2 (😂  _face_with_tears_of_joy_). 
 
We have to dig deeper to determine why 5 will be mislabelled 2 since smiling face and face with tears of joy are quite different. 

Similar error can be seen with classes 6 and 1, classes 16 and 1, classes 18 and 10.  

Generally, a lot of tweets get mislabelled as 1(😍 _smiling_face_with_hearteyes_) or 2 (😂  _face_with_tears_of_joy_).

Another poor performing area is label 13 ( 💜  _purple_heart_) where we only correctly identify 67 out of 1229 and mislabel most of them
as either 1 (😍 _smiling_face_with_hearteyes_), 2 (😂  _face_with_tears_of_joy_) or 3 (💕  _two_hearts_). 

Another worth noting is label 18(📸  _camera_with_flash_) where we only correctly label 173 out of 1400. We mislabel 439 as 10(📷  _camera_).

We perform quite well for labels 0(❤ _red_heart_), 1(😍 _smiling_face_with_hearteyes_), 
2(😂  _face_with_tears_of_joy_), 4( 🔥  _fire_), 17 (🎄  _Christmas_tree_).


Analysis of Output for Spanish Data:

------ Spanish Results ------
Macro F-Score (official): 25.924
-----
Micro F-Score: 45.972
Precision: 45.972
Recall: 45.972

----- Details -----
Training data len: 346860 Testing data len: 38539
Training data class counts: 0: 96719, 1: 46684, 2: 45739, 3: 24667, 4: 20955, 5: 14461, 6: 0, 7: 12169, 8: 0, 9: 0, 10: 19181, 11: 15270, 12: 11656, 13: 11055, 14: 0, 15: 16524, 16: 0, 17: 0, 18: 11780, 19: 0
Testing  data class counts: 0: 10756, 1: 5148, 2: 5120, 3: 2788, 4: 2314, 5: 1669, 6: 0, 7: 1279, 8: 0, 9: 0, 10: 2092, 11: 1729, 12: 1299, 13: 1224, 14: 0, 15: 1828, 16: 0, 17: 0, 18: 1293, 19: 0
--- Matrix ---
 0: 9111  695  242  246  103  102    0   18    0    0   82   50    8    2    0   80    0    0   17    0   ∑ = 10756
 1:   41 2725  818  380  216  148    0   65    0    0  302  121   15    2    0  235    0    0   80    0   ∑ =  5148
 2:   18 1145 3023  131  139   81    0   80    0    0  256   69    6   15    0   89    0    0   68    0   ∑ =  5120
 3:   39  888  269  846  116  209    0   34    0    0  129   99   12    1    0  135    0    0   11    0   ∑ =  2788
 4:   20  741  473  213  297  129    0   57    0    0  186   45    6    5    0   81    0    0   61    0   ∑ =  2314
 5:   25  496  235  274  113  303    0   26    0    0   78   42    4    3    0   58    0    0   12    0   ∑ =  1669
 6:    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   ∑ =     0
 7:    3  318  449   72   84   52    0   68    0    0  106   38    5    3    0   48    0    0   33    0   ∑ =  1279
 8:    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   ∑ =     0
 9:    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   ∑ =     0
10:    9  659  457  105  113   49    0   41    0    0  453   62    3    5    0   99    0    0   37    0   ∑ =  2092
11:   15  562  189  320   66   85    0   17    0    0  120  254    6    3    0   84    0    0    8    0   ∑ =  1729
12:   22  432  159  247   54   85    0   21    0    0   57   64   90    2    0   58    0    0    8    0   ∑ =  1299
13:    7  328  395   86   75   31    0   35    0    0  131   40    3   10    0   39    0    0   44    0   ∑ =  1224
14:    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   ∑ =     0
15:    9  567  285  134   78   43    0   43    0    0  148   27    2    4    0  464    0    0   24    0   ∑ =  1828
16:    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   ∑ =     0
17:    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   ∑ =     0
18:    6  441  357   68  105   38    0   30    0    0   98   26    2    1    0   48    0    0   73    0   ∑ =  1293
19:    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   ∑ =     0

For spanish tweets also, the true positives are along the diagonal of the confusion matrix.

The main problem with the spanish emojis is that no separate training data has been provided to train the model. 
Hence, the program cannot predict the emojis which are not present in the english data set. 
This can be seen in classes like 6,8,9,14,16,17 and 19. Because these emojis are absent in the english data set and the model returns 0 for this variables.

 
************************************ References **********************

1) https://www.quora.com/What-is-the-best-way-to-understand-the-terms-precision-and-recall
  