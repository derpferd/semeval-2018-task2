This file contains our analysis of our Stage 2 Results.

Author: Dennis Asamoah Owusu

Team: Hopper (Jon, Dennis, Sai)

Task: Multilingual Emoji Prediction (English and Spanish)

For stage 2, we tried a number of strategies to improve upon our baseline results. The first approach was to use a character based bidirectional LSTM classifier as was done by Barber et al. [1]. Due to the slowness of testing the models, we were able to try two sets of parameters for the neural network and embedding size. A neural network size and embedding size of 128 seemed to work better than 64 so we used the former in experimenting. We wondered how a uni-directional character based LSTM will fare and so tried that too. We also tried a character based LSTM combined with a deep convolutional neural network (CNN) inspired by dos Santos and Maira Gatti’s work on using CNN for sentiment analysis of short texts [2]. 

For the combined LSTM and CNN, we performed an additional experiment where, for training, we chose an equal number of tweets from each class. As we had a very dominant class (accounting for roughly 20% of tweets), we wondered whether this could be affecting the classifier performance since the performance of naive-Bayes classifiers, for instance, degrade when there is such a dominant class [3]. The change in using the combined CNN and LSTM on a dataset with equally represented classes compared to using it on the entire training set was negligible. We also combined the CNN with a bidirectional LSTM. Not too excited by the results of our neural network models, we tried a Bag of Words model with a Linear Support Vector Machine (LSVM) classifier. This turned out to provide the best results. The table below shows the Macro F1 scores of the various classifiers. Since we performed 10-fold cross-validation, we show the mean and standard deviation for each classifier’s results set. We add the results from our baseline - Bag of Words with Bernoulli naive-Bayes classifier - for comparison. En and Es stand for English and Spanish respectively.

```
			MEAN (En)	SDEV (En)	MEAN (Es)	SDEV (Es)
LSTM			27.83		0.47		N/A		N/A
Bi-LSTM			28.23		2.14		N/A		N/A
CNN + LSTM		29.30		0.36		N/A		N/A
CNN + LSTM (Fair)*	29.35		0.44		N/A		N/A
CNN + Bi-LSTM		29.29		0.36		16.15		0.49
LSVM			32.73		0.24		17.98		0.31
Baseline		29.10		0.20		16.49		0.42

* Fair means each class was equally represented in the training data.
```

Another experiment we performed that is worth noting was done to understand how much semantically similar classes such as camera and camera with flash impact misclassification. From our stage 1 results, we noticed that our model confused label 5 (smiling face with sunglasses) with label 1 (smiling face with heart eyes) significantly. We inferred that the model might have trouble discriminating between classes with similar semantics. That inference is supported by the fact that the model also struggles to discriminate between labels 10 (Camera) and 18 (Camera with flash). 

To measure the impact of these semantically similar classes on our model, we collapsed classes that were semantically similar and rerun the BOW model. The macro F1 score improved by 6 points suggesting that the semantic similarity between classes such as Camera and Camera with Flash does have an effect although the effect was not nearly as significant as we had expected. It is worth noting that after collapsing the semantically similar classes, a lot of tweets were misclassified as the most frequent class. Thus, it may be that the semantic similarity of the classes does really matter but the gain in performance from collapsing the classes was offset by the fact that we now had a class that was really huge and ate up too many classifications.

ANALYSIS OF LSVM RESULTS

Since the LSVM classifier provided the best results, we present an analysis of its performance. LSVM is what?


Below is a confusion matrix for the English results (first fold):

```
 0: 8848  610  248  245   42  100   42   59   27   62   36   34  173    7    5   13    2   57    6    6   ∑ = 10622
 1:   72 2903  665  368  143  165  143  143   71   69   90   74    0    9   13   28   12   67   38    4   ∑ =  5077
 2:   34  828 3171  119  173  125  123   51   26   51   68   57    0    5   23   75   10   61   54   13   ∑ =  5067
 3:   46  872  255  933   30  124   52   89   75   99   43   32    0   11   12   21    4   30   12    3   ∑ =  2743
 4:   13  381  382   71 1168   31   88   72   12   14   60   27    0    5    9   52    1   21   30    2   ∑ =  2439
 5:   29  732  456  202   76  327  100   60   29   69   41   39    0    2   17   17   14   54   16    1   ∑ =  2281
 6:   18  553  489   90  133   98  385   77   26   24   35   49    1    1   19   33    6   22   20    3   ∑ =  2082
 7:   22  510  221  146  107   57   87  422   14   21   57   24    0    3    9   19    5   48   27    6   ∑ =  1805
 8:   33  524  180  346   34   63   70   54  236   44   24   40    0    6   11   10    3   22   10    2   ∑ =  1712
 9:   30  505  232  268   38  113   40   41   26  234   22   19    0    8   11   16    4   29    9    3   ∑ =  1648
10:    4  167  157   29   50   30   35   34   14    4  769   13    0    1    6    6    3   10  198    0   ∑ =  1530
11:   14  285  146   40   28   41   57   27   21    6   21  909    0    2    3   14    0    8   11    0   ∑ =  1633
12:  658   12    6    4    2    4   14    2    3    0    0    0  622    0    0    0    0    0    1    0   ∑ =  1328
13:   29  434  160  284   27   49   24   40   48   48   12   20    0   82    8    6    2   10    7    0   ∑ =  1290
14:    7  325  433   92   35   80   81   31   14   25   16   17    0    5   45   20    4   25   17    4   ∑ =  1276
15:   10  245  387   60  142   33   68   41   29   13   28   16    0    1   10  244    2   11   11    0   ∑ =  1351
16:    8  429  383   50   57   97   64   37   10   16   19   27    0    1   12    9   28   24    8    2   ∑ =  1281
17:   10  111   58   21   24   19   20   27    5    4    4    7    0    1    3    1    0  937    2    1   ∑ =  1255
18:    5  174  172   30   62   22   25   49    9    8  423   12    0    1    8   11    2   10  305    3   ∑ =  1331
19:    9  291  460   85   58   75   69   22   15   15   18   17    0    2   19   12    4   16   17    5   ∑ =  1209
```


Below is a confusion matrix for the Spanish results (first fold):

```
 0: 1362  288   62   63   22   19   17    4    7   20    4    2    1    1    0    2    8    0    0   ∑ = 1882
 1:  299  690  128   44   41   18   21    7   11   50   12    0    2    1    0    3   11    0    0   ∑ = 1338
 2:  109  198  484   13   28   13   18    8    6   15    5    0    1    2    0    0    5    1    2   ∑ =  908
 3:  327  164   30   56   12   12    3    2    7   13    3    0    0    0    1    4    7    0    0   ∑ =  641
 4:  133  225   78   21   97   21   16   10   15   15    9    0    1    1    0    2    3    0    0   ∑ =  647
 5:  152  115   47   18   23   48   10    8    4    6    0    0    1    0    0    1    5    0    0   ∑ =  438
 6:   71   65   64   21   24    6  120    9    8    3    2    1    1    0    0    2    0    0    0   ∑ =  397
 7:   83   99   81    9   20    9   18   26    4   13    5    0    1    0    0    7    9    0    0   ∑ =  384
 8:   98  127   35   18   20    6   13    9   40    8    4    0    1    0    0    4    2    0    0   ∑ =  385
 9:   22   70   10    1    9    0    7    0    2  209    1    0    0    0    0    0    0    0    0   ∑ =  331
10:   66  107   58    6   27    5    7    1    9   20   15    0    1    0    1    5    3    0    1   ∑ =  332
11:  126   71   27   22    9    4    4    2    5    9    4    7    0    1    0    3    3    0    0   ∑ =  297
12:  145   74   19   21   10    2    5    1    2    6    1    0    4    0    0    0    6    0    0   ∑ =  296
13:   41   94   63    7   16    5   10    7    2    5    5    0    0    1    0    1    1    0    0   ∑ =  258
14:  132   74   12   27    6    6    4    0    4    2    0    0    0    0    1    2    6    0    0   ∑ =  276
15:   92   54   20   15   13    3    2    6    2    9    1    0    0    0    0   14   11    0    0   ∑ =  242
16:   72   62   54    9    8    4    2    7    2    2    1    2    0    0    0    0   58    0    0   ∑ =  283
17:  154   59    6   17    3    5    2    0    5    5    1    1    0    0    0    2    2    0    0   ∑ =  262
18:   49  101   53    2   23    9    5    3    1    9    2    0    0    1    0    0    3    0    0   ∑ =  261
```


REFERENCES

[1] Francesco Barbieri, Miguel Ballesteros, and Hora- cio Saggion. 2017. Are emojis predictable? In Proceedings of the 15th Conference of the Eu- ropean Chapter of the Association for Computa- tional Linguistics: Volume 2, Short Papers. Associa- tion for Computational Linguistics, pages 105–111. http://www.aclweb.org/anthology/E17-2017.

[2]Cicero dos Santos and Maira Gatti. 2014. Deep convolutional neural networks for sentiment anal- ysis of short texts. In Proceedings of COLING 2014, the 25th International Conference on Com- putational Linguistics: Technical Papers. Dublin

[3]Jason D. M. Rennie, Lawrence Shih, Jaime Teevan, and David R. Karger. 2003. Tackling the poor assump- tions of naive bayes text classifiers. In In Proceed- ings of the Twentieth International Conference on Machine Learning. pages 616–623.