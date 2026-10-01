# Online Review Sentiment Analysis (Flipkart / Twitter)

**Author:** Rabea Akter Shoily (Student ID 905253001) · United International University, School of Business and Economics

This project classifies short online product reviews as **positive, neutral or negative** using NLP. Reviews are cleaned (tokenisation, stopword removal, TF-IDF), and three models are compared against a majority-class baseline: **Logistic Regression, Multinomial Naive Bayes and a Linear SVM** (each also tuned with grid search). A key finding is that the class-demo's random split gives a misleading perfect score, because the dataset is built from only 42 repeated sentence templates; the models are therefore also tested on **unseen templates**, which gives realistic scores. The project also tracks how sentiment varies by month, source (Flipkart vs Twitter) and product, and saves the best pipeline to a single file. The written report is in the `report/` folder.

## Repository structure

```
.
├── data/        online_reviews_sentiment.csv           (1,000 synthetic reviews)
├── notebooks/   sentiment_analysis_project.ipynb       (full workflow, run top to bottom)
├── src/         text_utils.py                          (text-cleaning functions used by the notebook and the saved model)
├── models/      sentiment_pipeline.joblib              (saved cleaning + TF-IDF + Linear SVM pipeline)
├── outputs/     charts (*.png) and result tables (*.csv)
├── report/      project report (Word)
├── requirements.txt
└── README.md
```

## How to install and run

1. Install Python 3.10 or newer (developed with Python 3.12).
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Jupyter and open the notebook, then choose **Kernel > Restart & Run All**:
   ```bash
   jupyter notebook notebooks/sentiment_analysis_project.ipynb
   ```

Notes:
- Keep the folder layout above: the notebook reads `../data`, writes charts and tables to `../outputs` and the model to `../models`, and imports `../src/text_utils.py`.
- On first run the code downloads the NLTK stopword and tokenizer data (internet needed once).
- A fixed random seed (42) is used, so results are reproducible. The full run takes a few minutes because of repeated cross-validation.

## Dataset

`data/online_reviews_sentiment.csv`: 1,000 synthetic reviews of 15 products, January to December 2025, no missing values.

| Column | Meaning |
| --- | --- |
| `review_id` | unique id |
| `review_text` | the review (only this column is used as a model input) |
| `product` | one of 15 products |
| `sentiment` | label: positive (400), negative (340), neutral (260) |
| `source` | Flipkart (600) or Twitter (400) |
| `date` | review date |
| `rating` | star rating 1-5; it is unrelated to the sentiment label (correlation -0.005), so it is not used |

The reviews come from only **42 sentence templates** (14 per class) with the product name filled in, which is why a random train/test split leaks and gives F1 = 1.00.

## Key results

Scores on **unseen sentence templates** (weighted F1; the best model is chosen by cross-validated F1):

| Model | Accuracy | F1 (hold-out) | CV F1 (mean ± sd) |
| --- | --- | --- | --- |
| **Linear SVM (saved model)** | 0.806 | **0.807** | **0.804 ± 0.090** |
| Linear SVM (tuned) | 0.801 | 0.802 | 0.800 ± 0.104 |
| Logistic Regression (tuned) | 0.699 | 0.706 | 0.770 ± 0.088 |
| Logistic Regression | 0.755 | 0.749 | 0.748 ± 0.058 |
| Multinomial Naive Bayes | 0.745 | 0.731 | 0.730 ± 0.059 |
| Naive Bayes (tuned) | 0.684 | 0.678 | 0.723 ± 0.102 |
| Majority baseline | 0.378 | 0.207 | 0.229 ± 0.013 |

- With the class-demo random split every model scores F1 = 1.00, so it cannot tell models apart.
- Keeping negation words ("not", "no") raised average cross-validated F1 from 0.724 to 0.761.
- Neutral reviews are the hardest class (recall 0.61).
- Sentiment does not differ significantly by month (p = 0.60), source (p = 0.80) or product (p = 0.40).

All charts and tables are in `outputs/`.

## Using the saved pipeline

Run from the repository root:

```python
import sys, joblib
sys.path.append("src")                      # the pipeline uses the cleaning function in src/text_utils.py
model = joblib.load("models/sentiment_pipeline.joblib")
print(model.predict([
    "Terrible experience, the Monitor does not match the description.",
    "Not good at all, it stopped working after two days.",
]))   # ['negative' 'negative']
```
