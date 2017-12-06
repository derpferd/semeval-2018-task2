Author: Dennis Asamoah Owusu
This file contains our analysis of our Stage 2 Results.

Team: Hopper (Jon, Dennis, Sai)
Task: Multilingual Emoji Prediction (English and Spanish)

For stage 2, we tried a number of strategies to improve upon our baseline results. The first approach was to use a character based bidirectional LSTM classifier as was done by Barber et al.[1]. Due to the slowness of testing the models, we were able to try two sets of parameters for the neural network and embedding size. A neural network size and embedding size of 128 seemed to work better than 64 so we used the former in experimenting. We wondered how a uni-directional character based LSTM will fare and so tried that too. We also tried a character based LSTM combined with a deep convolutional neural network (CNN) inspired by dos Santos and Maira Gatti work on using CNN for sentiment analysis of short texts[2]. 

For the combined LSTM and CNN, we performed an additional experiment where, for training, we chose an equal number of tweets from each class. As we had a very dominant class (accounting for roughly 20% of tweets), we wondered whether this could be affecting the classifier performance since the performance of naive-Bayes classifiers, for instance, degrade when there is such a dominant class[3]. Using the combined CNN and LSTM on a dataset with equally represented classes was negligibly better than using it on the entire training set. We also combined the CNN with a bidirectional LSTM. Not too excited by the results of our neural network models, we tried a Bag of Words model with a Linear SVM classifier. This turned out to provide the best results. The table below shows the Macro F1 scores of the various classifiers. Since we performed 10-fold cross-validation, we show the mean and standard deviation for each classifier’s results set. We add the results from our Bag of Words with naive-Bayes classifier for comparison.

```
			MEAN (En)	SDEV (En)	MEAN (Es)	SDEV (Es)
LSTM			27.83		0.47		N/A		N/A
Bi-LSTM			28.23		2.14		N/A		N/A
CNN + LSTM		29.30		0.36		N/A		N/A
CNN + LSTM (Fair)*	29.35		0.44		N/A		N/A
CNN + Bi-LSTM		29.29		0.36		16.15		0.49
LSVM			32.73		0.24		17.98		0.31
Bernoulli		29.10		0.20		16.49		0.42
```


 






REFERENCES
[1] Francesco Barbieri, Miguel Ballesteros, and Hora- cio Saggion. 2017. Are emojis predictable? In Proceedings of the 15th Conference of the Eu- ropean Chapter of the Association for Computa- tional Linguistics: Volume 2, Short Papers. Associa- tion for Computational Linguistics, pages 105–111. http://www.aclweb.org/anthology/E17-2017.

[2]Cicero dos Santos and Maira Gatti. 2014. Deep convolutional neural networks for sentiment anal- ysis of short texts. In Proceedings of COLING 2014, the 25th International Conference on Com- putational Linguistics: Technical Papers. Dublin

[3]Jason D. M. Rennie, Lawrence Shih, Jaime Teevan, and David R. Karger. 2003. Tackling the poor assump- tions of naive bayes text classifiers. In In Proceed- ings of the Twentieth International Conference on Machine Learning. pages 616–623.