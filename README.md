# Agile User Story Point Estimation

## Table of contents
1. [Introduction](#introduction)
2. [Data](#data)
3. [Approach](#approach)
4. [Evaluation](#evaluation)
5. [Primitive Results](#primitive-results)
6. [How to run](#how-to-run)
6. [Reference](#reference)

## Introduction

Effort estimation is an important part of software project management. Cost and schedule overruns create more risk for software projects. Effort estimations can benefit project manager and clients to make sure that projects can complete in time.

In modern Agile development, software is developed through repeated cycles (iterations). A project has a number of iterations. An iteration is usually a short (usually 2–4 weeks) period in which the development team designs, implements, tests and delivers a distinct product increment, e.g. a working milestone version or a working release. Each iteration requires the completion of a number of user stories, which are a common way for agile teams to express user requirements.

![](pic/agile.png)

Therefore, there is a need to focus on estimating the effort of completing single user story rather than entire project. In fact, it is very common for agile teams to go through each user story estimate the effort required to completing it.

Currently, most agile teams heavily reply on experts' subjective assessment to estimate time. This may be bias and inaccuracy and inconsistent between estimates.
![](pic/effort.png)

## Data

Dataset are from 16 large open source projects in 9 repositories namely
- Apache
- Appcelerator
- DuraSpace
- Atlassian
- Moodle
- Lsstcorp
- Mulesoft
- Spring
- Tatendforge

The authors claim this is the largest dataset in term of number of data points for story point estimation

__Investigate data__

Dataset contain 23,313 issues with story points from 16 different projects.

Features:
- Story points
- title
- description


Among projects, 7 / 16 follow Fibonnaci scale, other 9 did not use any scale. Because of this reason, the authors did not round their estimation to nearest estimation on Fibonacci scale. This makes their approach applicable for wider range of projects.

## Approach

We would propose an end-to-end solution to estimate story point (number) based on user story (text).


- Input: User story (Title + description) of issues
- Output: story-point estimation

End-to-end approach, without worrying about manual features.

In order to do it, we would find a way to represent user story (Text to Vector), there are several ways to do it for example Doc2Vec model, LSTM, CNN or TF-IDF.

After that we can could use a regressor to estimate story point, some ideas could be Neural Network, Recurrent Highway Network, Gradient Boosting Tree.

![](pic/process.png)

__LSTM architecture__

|||
|:--:|:--:|
|![](pic/LSTM.png)|![](pic/LSTM_average.png)|

## Evaluation

- Mean Absolute Error
- Median Absolute Error
- Standardized Accuracy


## Results

1. Word2Vec
![](pic/result_word2vec.png)
2. How embedding size affect Result

||||
|:--:|:--:|:--:|
|![](pic/word2vec_ave.png)|![](pic/doc2vec.png)|![](pic/LSTM_size.png)|

3. Result

|Model|Mean Absolute Error|Median Absolute Error|Mean Square Error|
|:---:|:--:|:--:|:--:|
|Tf-idf + Random Forest|3.96|1.90|82.64|
|Tf-idf + LightGBM|4.41|2.43|84.59|
|tf-idf + Catboost -> cannot deal with sprase data type||||
|Average Word2Vec 100 + Random Forest|4.95|2.7|94.95|
|Average Word2Vec 100 + LightGBM|4.42|2.54|82.25|
|Average Word2Vec 200 + Random Forest|5.13|2.7|98.94|
|Average Word2Vec 200 + LightGBM|4.38|2.48|81.21|
|Average Word2Vec 300 + LightGBM|4.35|2.50|80.21|
|Average Word2Vec 400 + LightGBM|4.31|2.46|78.66|
|Average Word2Vec 400 + Catboost|4.73|3.16|90.31|
|Doc2Vec 100 + Random Forest|5.88|3.4|119.06|
|Doc2Vec 100 + LightGBM|4.84|3.05|97.44|
|LSTM end-to-end 50 unit tf.nn.rnn_cell.LSTMCell|3.97|N/A|92.07|
|LSTM end-to-end 50 unit, 100 embeddingsize tf.nn.rnn_cell.LSTMCell|3.98|N/A|90.51|
|LSTM end-to-end 100 unit tf.nn.rnn_cell.BasicLSTMCell|4.46|N/A|84.78|
|LSTM end-to-end 200 unit tf.nn.rnn_cell.LSTMCell|5.09|N/A|86.27|

## Improvements

## How To Run
### Docker (recommended)
1. Build and start services
```
docker compose up -d --build
```
If build fails on Apple Silicon, set the platform:
```
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up -d --build
```
If you are behind Netskope/SSL inspection, export the Netskope root CA and pass it during build:
```
export NETSKOPE_CA_B64="$(base64 -b 0 /path/to/netskope-ca.pem)"
docker compose up -d --build
```
If you must bypass cert verification (less secure), use trusted hosts for pip:
```
export PIP_TRUSTED_HOSTS="pypi.org pypi.python.org files.pythonhosted.org"
docker compose up -d --build
```

2. Import CSVs into MongoDB (expects ./dataset/*.csv)
```
docker compose exec mongo mongo mydb --eval 'db.createCollection("storypoint")'
docker compose exec mongo bash -lc 'for entry in /dataset/*.csv; do mongoimport -d mydb -c storypoint --type CSV --file "$entry" --headerline; done'
```

3. Run preprocessing and training inside the container
```
docker compose exec app python preprocessing.py
docker compose exec app python Word2VecFeature.py --proc 8
docker compose exec app python Doc2VecFeature.py --proc 8
docker compose exec app python RandomForest.py --size 100 --feature_name word2vec_ave
docker compose exec app python lightGBM.py --size 100 --feature_name word2vec_ave
docker compose exec app python catb.py --size 100 --feature_name word2vec_ave
docker compose exec app python LSTM_regression.py --rnn_layers 2 --rnn_units 100 --embedding_size 100
```

### Local
1. MongoDB
- Create database and collections in MongoDB (no password)
```
use mydb
db.createCollection("storypoint")
```
- From command line
```
bash importCSV.sh
```

2. Install Library
```
bash install_libary.sh
```

3. Preprocessing
```
python preprocessing.py
```

4. Building Word2Vec features

This is a multi-processing program to boost speed
```
(default setting) python Word2VecFeatures.py
(proc setting) python Word2VecFeatures.py --proc 8

Arguments
- proc is number of processor, please set it an even number
```

5. Building Doc2Vec features

This is a multi-processing program to boost speed
```
(default setting) python Word2VecFeatures.py
(proc setting) python Doc2VecFeatures.py --proc 8

Arguments
- proc is number of processor, please set it an even number
```

6. Random Forest, LightGBM and Catboost Models

```
python RandomForest.py --size 100 --feature_name word2vec_Ave
python lightGBM.py --size 100 --feature_name word2vec_Ave
python catb.py --size 100 --feature_name word2vec_Ave

Arguments
- size: embedding size
- feature_name: word2vec_ave or doc2vec
```

7. LSTM models
```
python LSTM_regression.py
python LSTM_regression.py --rnn_layers 2 --rnn_units 100 --embedding_size 100

Arguments
- batch_size: size of training batch
- embedding_size: size of embedding words
- rnn_layers: How many layers you want to stack
- rnn_units: How many nodes in a layer

Erros:
- terminate called after throwing an instance of 'std::bad_alloc'
You need more RAMs
```
__Note:__ To re-train the model please delete folder "models"


# Reference
1. http://deeplearning.net/tutorial/lstm.html
2. [How to Develop Convolutional Neural Network Models for Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/?utm_campaign=Learning%20Posts&utm_content=80012867&utm_medium=social&utm_source=linkedin)
3. [RNN in TF - A practical guide and undocumented features -> very good](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/)
4. https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
5. https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
6. https://www.tensorflow.org/tutorials/sequences/recurrent
