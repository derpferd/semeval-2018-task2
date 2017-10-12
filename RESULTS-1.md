Author: Dennis Asamoah Owusu

We implemented two baselines. One was a Bag-of-words model using a Bernoulli Naive Bayes classifier. 
The other was a Most Frequent Class model. 
The results of both models are shown below. Detailed analysis is, however, performed only on the Bernoulli model 
since that is what we have chosen for our baseline.
We did a 10-fold cross-validation and, hence, we show a range of scores instead of a single score.

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

## Analysis
First, it is worth noting that our Micro F-score, Precision and Recall have the same values for each set of results. 
For instance, under Results for English Tweets for Most Frequent Class Model, 
the values range from 21.408 - 21.954 for all the three scores.
This shows that all our models assigned classes (emojis) for all tweets that were in the test set. From the scorer,
precision and recall are equal when attempted_total is equal to gold_occurrences_total. (See scorer_semeval18.py lines 52 and 53)

For the English tweets, the difference in the Macro and micro F-scores for our Bernoulli model lies in the fact that 
about 21% of tweets have label 0 (representing ❤ - the red heart) which is the most frequent class.
This can be seen from the results of our Most Frequent Class model which has a precision and recall of about 21%.
A similar thing happens with the Spanish tweets where about 28% of the tweets belong to the same red heart class. 
Note the correlation between
the relatively higher percentage of tweets in the most frequent class and the relatively higher Micro F-score 
for the Spanish tweets.
However, once the effect of the most frequent class is offsetted as measured by the Macro F-score, 
the prediction for the English tweets becomes better than that of the Spanish tweets.
7 classes are missing from both the training data and testing data for Spanish. 
Another important difference between the Spanish and English is that there are about 100,000 more
tweets in the English training data. This, likely, contributes to English's better Macro F-scores.   

Below is a confusion matrix for English:
```
 0: 8637  618  215  234   61   88   61   50   35  106   21   79  420    7   15   15   13   62   12    3   ∑ = 10752
 1:   25 2358  832  380  197  168  229  152  102  152   90  188   23    7   52   22   62   68   35    3   ∑ =  5145
 2:   13 1022 2718  129  258  130  185   42   57   69   62  133    5    3   58   61   57   61   40   11   ∑ =  5114
 3:   29  816  293  800   59  107   97   82   88  195   25   72    6   11   22   20   12   29   19    3   ∑ =  2785
 4:    5  472  360   62 1043   34  113   40   24   35   66   60    5    3   14   45   14   24   24    2   ∑ =  2445
 5:   16  653  437  210   95  251  139   48   33  122   47   67    5    2   35   17   53   62   20    1   ∑ =  2313
 6:    7  558  385   98  155   82  332   69   43   43   42  102   16    3   29   42   30   29   24    3   ∑ =  2092
 7:    2  475  242  117  126   66  108  342   20   39   58   79    4    0   24   21   13   54   37    0   ∑ =  1827
 8:   12  510  157  310   48   63   94   54  219   82   22   84    6    2   10   18    8   20    7    1   ∑ =  1727
 9:   12  445  236  263   62  102   51   24   35  285   20   47    4    3   13   10   14   31   10    1   ∑ =  1668
10:    0  209  154   31   98   34   48   30   11   12  668   33    2    2   12   12    4   14  163    0   ∑ =  1537
11:    6  255  116   31   35   38   75   25   19   15   28  928    4    3   25   13    6   10   10    2   ∑ =  1644
12:  586   13    5    3    2    6   13    1    0    0    0    1  712    0    0    0    0    1    2    0   ∑ =  1345
13:   15  385  163  245   36   46   46   37   57   79   12   49    3   67   14   12    7   16   10    0   ∑ =  1299
14:    2  275  416   64   70   69   72   27   25   54   19   40    5    4   45   21   33   23   10    4   ∑ =  1278
15:    2  296  361   59  146   33   96   21   36   30   20   41    1    1   11  171    5   13   19    2   ∑ =  1364
16:    2  393  333   63   74   75   78   32   17   30   19   55    2    1   23    6   56   23   10    0   ∑ =  1292
17:    1  119   56   26   32   20   26   37    8    8    8   23    0    1    7    2    5  887    1    1   ∑ =  1268
18:    1  204  186   31   82   29   45   39   17   15  458   34    3    0   16   13    6   16  148    1   ∑ =  1344
19:    5  286  374   81   64   57   88   25   27   30   23   38    2    2   32   14   38   17   15    6   ∑ =  1224
```
Below is a confusion matrix for Spanish:
```
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
```

### Areas Particularly Performing Poorly
First, some information about the matrix. The first column represents the emoji labels and the last column
(with the summation sign) shows the total number of tweets assigned to that label in the gold standard.
The columns between the first and the last show how many we classified for each label. For instance, for Spanish,
there were 10,756 tweets with label 0 in the gold standard. We labelled 9111 of these tweets correctly as label 0
but labelled 695 of them incorrectly as label 1, 242 of them incorrectly as label 2 and so on. For label 1, 
there were 5148 tweets in the gold standard, we labelled 41 of these incorrectly as label 0 and labeled 2725
correctly as label 1. Thus, you can look diagonally from the top left to the bottom right to see the number
labelled correctly for each class.


Focusing on English, the worst performing areas are as follows:

Label 5 (😊  _smiling_face_with_smiling_eyes_): 
2313 tweets were assigned in the gold. We assigned only 251 correctly. 653 were 
incorrectly assigned label 1 (😍 _smiling_face_with_hearteyes_) and 437 were 
incorrectly assigned label 2 (😂  _face_with_tears_of_joy_). 
Mislabelling 5 as 1 is quite understandable since they are both
smiling faces. We have to dig deeper to determine why 5 will be mislabelled 2 since smiling face and
face with tears of joy are quite different. 

Label 6(😎  _smiling_face_with_sunglasses_):
2092 in the gold standard. We assigned only 332 correctly. We mislabelled 558 as 1(😍 _smiling_face_with_hearteyes_)
and mislabelled 385 as 2 (😂  _face_with_tears_of_joy_). This is almost the same situation with Label 5.

Generally, a lot of tweets get mislabelled as 1(😍 _smiling_face_with_hearteyes_) or 2 (😂  _face_with_tears_of_joy_).

Another poor performing area is label 13 ( 💜  _purple_heart_) where we only correctly identify 67 out of 1229 and mislabel most of them
as either 1 (😍 _smiling_face_with_hearteyes_), 2 (😂  _face_with_tears_of_joy_) or 3 (💕  _two_hearts_). 

Another worth noting is label 18(📸  _camera_with_flash_) where we only correctly label 173 out of 1400. We mislabel 439 as 10(📷  _camera_).

### Areas Performing Well 
We perform quite well for labels 0(❤ _red_heart_), 1(😍 _smiling_face_with_hearteyes_), 
2(😂  _face_with_tears_of_joy_), 4( 🔥  _fire_), 17 (🎄  _Christmas_tree_).
 










































