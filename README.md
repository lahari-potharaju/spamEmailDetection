# Spam Email Detection

A machine learning project in R that classifies emails and SMS messages as spam or legitimate using multiple classification algorithms — CART decision trees, Naive Bayes, Logistic Regression, and K-Nearest Neighbors.

---

## Project Structure

```
├── email_classification_cart.R     # CART decision tree on Enron email dataset
├── sms_spam_detection.R            # Naive Bayes, Logistic Regression, KNN on SMS spam (main)
├── sms_spam_detection_v1.R         # Earlier draft of multi-model SMS classifier
├── SMSSpamCollection.xlsx          # SMS Spam Collection dataset (5,574 messages)
├── energy_bids.csv                 # Enron email dataset (855 emails, responsive/non-responsive)
├── report_spam_email_detection.docx
├── presentation_final.pptx
└── presentation_progress.pptx
```

---

## Datasets

| Dataset | Source | Size | Task |
|---|---|---|---|
| `SMSSpamCollection.xlsx` | UCI ML Repository | 5,574 SMS messages | Ham vs. Spam classification |
| `energy_bids.csv` | Enron e-discovery corpus | 855 emails | Responsive vs. Non-responsive |

---

## Models & Results

### CART Decision Tree (`email_classification_cart.R`)
Trained on the Enron energy bids email dataset using the `rpart` package.

| Metric | Value |
|---|---|
| Model Accuracy | 85.6% |
| Baseline Accuracy | 83.7% |
| AUC (ROC) | 0.794 |

### Multi-Model SMS Spam Classifier (`sms_spam_detection.R`)
Text preprocessing pipeline: lowercasing → punctuation removal → stopword removal → stemming → Document-Term Matrix.

| Model | Accuracy |
|---|---|
| Naive Bayes | 97.3% |
| Logistic Regression | 91.1% |
| KNN (k = 5) | 90.4% |
| KNN (k = 15) | 87.6% |

---

## Requirements

Install required R packages before running:

```r
install.packages(c("tm", "SnowballC", "rpart", "rpart.plot",
                   "caTools", "ROCR", "readxl", "caret",
                   "e1071", "class", "stringr"))
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/lahari-potharaju/spamEmailDetection.git
   cd spamEmailDetection
   ```

2. Open R or RStudio and run either script:
   ```r
   source("email_classification_cart.R")   # CART on Enron emails
   source("sms_spam_detection.R")           # Multi-model SMS spam classifier
   ```

---

## Techniques Used

- **Text preprocessing**: `tm` package — tokenization, stopword removal, stemming (SnowballC)
- **Feature extraction**: Document-Term Matrix with sparse term removal
- **CART**: `rpart` with ROC curve evaluation via `ROCR`
- **Naive Bayes**: `e1071` with categorical features and Laplace smoothing
- **Logistic Regression**: `glm` with binomial family
- **KNN**: `class::knn` with k = 5 and k = 15
- **Train/test split**: `caTools::sample.split` and `caret::createDataPartition`
