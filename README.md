# Fake News Detection

This project classifies news articles as real or fake using only the text of the article. We built a TF-IDF pipeline, tuned three classic models (SVM, Logistic Regression, Naive Bayes), and compared them on accuracy, stability, training time and interpretability.

The whole work is in one notebook: `beredjem-ouriachi-sellami-boudaba-projet-8.ipynb`.

Authors: Sellami Siham

## Dataset

We used the Fake News Classification dataset from Kaggle (`evaluation.csv`, separator `;`). It has around 8,100 articles with a `text` column and a `label` column (0 = fake, 1 = real). We dropped the title and index columns and kept only the text.

The data was split 80/20 with a stratified split and `random_state=42`, which gives 6,493 articles for training and 1,624 for testing.

## What the notebook does

1. Load the data.
2. Clean the text: lowercase, remove punctuation and numbers, remove English stop words, then apply Porter stemming and WordNet lemmatization.
3. Split into train and test sets.
4. Turn the text into TF-IDF features (up to 5,000 features, unigrams and bigrams).
5. Build one scikit-learn pipeline per model and tune it with GridSearchCV.
6. Check bias and variance for each model.
7. Compare the models with 10-fold cross-validation, ANOVA and paired t-tests.
8. Measure training time.
9. Evaluate on the held-out test set.
10. Look at which words push the SVM toward "fake" or "real".

## Best hyperparameters found

- SVM: linear kernel, C=1, max_df=0.8, min_df=5, ngram_range=(1,2)
- Logistic Regression: C=10, liblinear solver, max_df=0.8, min_df=1, ngram_range=(1,2)
- Naive Bayes: alpha=0.1, max_df=0.8, min_df=10, ngram_range=(1,2)

## Results

### Accuracy of the three models

![Accuracy comparison](https://raw.githubusercontent.com/Siamsell/ML-Fake-News-Detection/main/images/accuracy_comparison.png)

SVM comes first with 96.80% on the test set, followed by Logistic Regression at 96.37% and Naive Bayes at 92.43%. The cross-validation scores are almost the same as the test scores, so the results are consistent.

### Cross-validation spread

![Cross-validation boxplot](https://raw.githubusercontent.com/Siamsell/ML-Fake-News-Detection/main/images/cv_boxplot.png)

SVM and Logistic Regression overlap a lot. Naive Bayes is lower and also less stable from one fold to another.

The ANOVA test shows a significant difference between the three models (p < 0.0001). In the pairwise t-tests, Naive Bayes is significantly different from both of the others. The gap between Logistic Regression and SVM is not significant (p = 0.0775).

### Bias and variance

SVM:

![Bias-variance SVM](https://raw.githubusercontent.com/Siamsell/ML-Fake-News-Detection/main/images/bias_variance_svm.png)

Logistic Regression:

![Bias-variance Logistic Regression](https://raw.githubusercontent.com/Siamsell/ML-Fake-News-Detection/main/images/bias_variance_logistic_regression.png)

Naive Bayes:

![Bias-variance Naive Bayes](https://raw.githubusercontent.com/Siamsell/ML-Fake-News-Detection/main/images/bias_variance_naive_bayes.png)

For all three models the gap between training and validation score stays under 0.10, so none of them overfits badly. Logistic Regression has the biggest gap (0.996 train against 0.965 validation) and Naive Bayes has the lowest scores overall (0.938 against 0.926).

### Training time

![Training time](https://raw.githubusercontent.com/Siamsell/ML-Fake-News-Detection/main/images/training_time.png)

Naive Bayes (4.84 s) and Logistic Regression (4.90 s) train in about the same time. The SVM is around four times slower (20.36 s).

### Which model to pick

SVM is the most accurate and is our final choice. If training speed matters more than the last half percent of accuracy, Logistic Regression is a good alternative because it is statistically close to the SVM and much faster.

## Interpretation

We used the linear SVM for interpretation because its coefficients can be read directly, which is not possible with an RBF kernel. Words with a negative coefficient point to fake news, and words with a positive coefficient point to real news.

![Top words SVM](https://raw.githubusercontent.com/Siamsell/ML-Fake-News-Detection/main/images/top_words_svm.png)

The word clouds below show the most frequent words in each class and agree with what the coefficients say.

![Word clouds](https://raw.githubusercontent.com/Siamsell/ML-Fake-News-Detection/main/images/wordcloud_fake_vs_real.png)

## How to run it

Install the dependencies:

```bash
pip install pandas numpy scikit-learn nltk scipy matplotlib wordcloud jupyter
```

Download the NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

Open the notebook and change the dataset path in the import cell. It currently points to a Kaggle folder (`/kaggle/input/...`), so replace it with the location of your own `evaluation.csv`.

To predict on a new article after running the notebook:

```python
cleaned = preprocess("your article text here")
label = best_svm.predict([cleaned])[0]
print("real" if label == 1 else "fake")
```

## Repository layout

```
.
├── fake-news-detection-model.ipynb
├── images/
└── README.md
```


