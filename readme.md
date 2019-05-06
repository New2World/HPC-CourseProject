# LANL-Earthquake-Prediction

## Introduction

This project is an on-going kaggle competition, aiming at forecasting earthquakes. it is one of the most important problems in Earth science to precise because of the devastating consequences of earthquakes.  
In earthquake prediction, there are three key points: **when**, **where**, **how long**. Our topic focus on **when** the earthquake will happen.

## Preliminary

In pre-proposal, our goals are:

- [ ] prize
- [x] learn some machine learning algorithms
- [x] experience with scikit-learn, TensorFlow and PyTorch

We decided to use _support vector regressor_, _hidden Markov model_, _random forest_ and _recurrent neural network_. Besides, after getting the results of these models, we will choose to apply boosting to get better prediction accuracy.

## State of the Art

> TODO

Continuous chatter of the Cascadia subduction zone revealed by machine learning ([link](https://www.nature.com/articles/s41561-018-0274-6))

Similarity of fast and slow earthquakes illuminated by machine learning ([link](https://www.nature.com/articles/s41561-018-0272-8#data-availability))
- model  
gradient boosted trees25 algorithm

Earthquake prediction model using support vector regressor and hybrid neural networks ([link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0199004))  

The SVR-HNN model
![paper1](https://journals.plos.org/plosone/article/figure/image?id=10.1371/journal.pone.0199004.g003&size=large)

Parameters
- a and b value (from well-known geophysical law known as Gutenberg-Richter law)
- Seismic energy release
- Time of n events
- Mean Magnitude
- Seismic rate changes
- Maximum magnitude in last seven days
- Probability of earthquake occurrence
- Deviation from Gutenberg-Richer law
- Standard deviation of b value
- Magnitude deficit
- Total recurrence time

## Data

The data is stored in a csv format file, and there are two columns: `acoustic data` and `time to failure`, indicating the data collected from experimental earthquake and the remaining time to next earthquake. The size of data is enormous (8.9GB), so to get an overview of data, we down-sample the data with a span of 300 and draw the curve shown below.

Here is what the raw data looks like.

|acoustic_data|time_to_failure|
|:-:|:-:|
|12|1.4690999832|
|6|1.4690999821|
|8|1.469099981|
|5|1.4690999799|
|8|1.4690999788|
|8|1.4690999777|
|9|1.4690999766|
|7|1.4690999755|
|-5|1.4690999744|
|3|1.4690999733|

### Visualization

![visualization](data_visualization.png)

Each peak of red line correspond to a zero value in blue line, indicating an earthquake happens.

As we usually do to anaylize data with temporality, we apply Fourier transform. Here is the figure of one entire earthquake in frequency field. Most of time there is low frequency data, only when an earthquake happens high frequency will appear for few milliseconds.

![fft](fft.png)

### Features

In preliminary we under estimated the importance of feature engineering. As this is a traditional machine learning task, and the raw data is presented in time sequence, we cannot feed the data into machine learning models directly. Shown the raw data visualization, the acoustic data is dense in time steps, but the value disturbance is not significant most of time.
Here are some statistic features extracted from the raw data, including mean, mode, standard deviation etc.

> TODO: 
> - explain `kurtosis`, `skew`, `quantile`  

Kurtosis
- measure of tailedness  
![pic2](https://upload.wikimedia.org/wikipedia/commons/e/e6/Standard_symmetric_pdfs.png)

Skewness
- measure of asymmetry
![pic1](https://cdn-images-1.medium.com/max/800/1*nj-Ch3AUFmkd0JUSOW_bTQ.jpeg)

Quantile
- Cut points that devide data
![pic3](https://www.hr-diagnostics.de/fileadmin/user_upload/Magazin/Artikelbilder/normieren-und-die-normalverteilung.jpg)

|mean| std| max|kurtosis|skew|quantile (.01)|quantile (.05)|quantile (.95)|quantile (.99)|
|:--:|:--:|:--:|:------:|:--:|:-----------:|:-----------:|:-----------:|:-----------:|
|4.79|2.55|13.0000|-0.18|0.26|0.0000|1.0000|9.0000|11.0000|
|4.76|2.52|13.0000|-0.24|0.22|0.0000|1.0000|9.0000|11.0000|
|4.75|2.52|13.0000|-0.24|0.23|0.0000|1.0000|9.0000|11.0000|
|4.75|2.53|13.0000|-0.25|0.24|0.0000|1.0000|9.0000|11.0000|
|4.75|2.52|13.0000|-0.24|0.24|0.0000|1.0000|9.0000|11.0000|
|4.77|2.50|13.0000|-0.22|0.25|0.0000|1.0000|9.0000|11.0000|
|4.78|2.52|13.0000|-0.24|0.26|0.0000|1.0000|9.0000|11.0000|
|4.78|2.52|13.0000|-0.24|0.26|0.0000|1.0000|9.0000|11.0000|
|4.78|2.52|13.0000|-0.24|0.26|0.0000|1.0000|9.0000|11.0000|
|4.78|2.52|13.0000|-0.23|0.26|0.0000|1.0000|9.0000|11.0000|

## Method

### Random Forest (baseline)

Every experiments have baseline, we choose random forest as baseline model. Random forest is an ensemble model, it is a combination of multiple decision trees and **bootstrap aggregation**, also known as **bagging**. The basic idea is to combine multiple decision trees in determining the final output, two heads are better than one.  
Each decision tree in the "forest" is a independent model. So to generate each tree, there must be some criterias to split tree into two subtrees. For classification task, the criterion can be either _entropy_ or _Gini index_; and for regression, we usually use _residual sum of squares_

$$rss = \sum_{left\_tree}(y_i-y^*_L)^2+\sum_{right\_tree}(y_i-y^*_R)^2$$

where $y^*_L$ means y-value for left node, while $y^*_R$ for right node.  
Each tree uses different data generated by bootstrapping, so it helps reduce variance. (Our data has few samples but very large amount of time steps in each earthquake sample)

### Support Vector Machine

Support vector machine is a deterministic classification model, using support vector to get the decision boundary that has a largest margin to nearest data. It can also be used as a regression method, maintaining all the main features that characterize the algorithm.  
Original support vector regression is a linear model. To extend it to fit non-linear functions, kernel functions are introduced. Kernel is first proposed in classification method, like the figure showing below, the data in the figure is not linear separable if we look at it from above. What kernels do is to map the data to higher dimension from which it can be differentiate by a single line or a hyperplane.

We use the support vector machine model in [scikit-learn](https://scikit-learn.org/stable/), a free software machine learning library for the Python programming language. The support vector machine implementation is based on `libsvm`, a high-efficient open source machine learning library, and apart from original support vector regressor, it provide another version of support vector machine which has a upper bound of support vectors.

[![](svm_kernel.gif)](https://towardsdatascience.com/understanding-support-vector-machine-part-2-kernel-trick-mercers-theorem-e1e6848c6c4d)

### Recurrent Neural Network (LSTM, GRU)

Recurrent neural network is a neural network structure aiming at dealing with sequential data. Like reading a book, human will keep knowledge about previous chapters in mind while reading. So as the way how recurrent neural network works, during learning all neural cells share the same parameters and update together.  
However, the original version of recurrent neural network is facing some problems, like losing what has been learned before in the situation that the time sequence is long enough and gradient vanishing. So here comes long-short term memory (LSTM) structure. In the cell of LSTM, there are three "gates" to determine what to **forget**, what to **update** and what to **output**. After each LSTM cell there will be an output to system and a status output to next cell, and the status output will help the neural network model to keep a long-term memory.

[![](lstm_sample.png)](https://colah.github.io/posts/2015-08-Understanding-LSTMs)

Besides, there is another version of recurrent network named gated recurrent unit (GRU). Similar to LSTM, the main idea behind GRU is to let model learn how to keep useful long-term information while learning short-term knowledge. Also GRU has gates: **update gate** and **reset gate**.

[![](gru_sample.png)](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)

### Boosting

Boosting is a machine learning ensemble meta-algorithm for reducing both variance and bias. Different from bagging's parallel mechanism, boosting is a sequential training process. The main idea is to train several weak learners sequentially, and give the misclassified samples higher weight, which means next learner should pay more attention to those samples to correct the mistake from its predecessor.

[![](ensemble.png)](https://medium.com/greyatom/a-quick-guide-to-boosting-in-ml-acf7c1585cb5)

> TODO: 
> - more about boosting

## Result

> have little experience with kernel function for regression

> TODO: 
> for each model:
> - troubles
> - chart for different hyperparameters

## Conclusion

> TODO
