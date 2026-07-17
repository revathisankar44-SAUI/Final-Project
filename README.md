# Final-Project
Amazon Product Reviews – Multi-Class Sentiment Analysis and Topic Modelling

A complete NLP pipeline for the analysis of Amazon US customer reviews through identification of fake reviews, sentiment classification through multiple models, topic modeling, and review quality score into one business insights dashboard.

Introduction

There is an immense number of unstructured product reviews on online marketplaces, but raw review data has two challenges – some of the reviews are not real and even those that are real do not provide insight into why customers feel that way. This project creates a pipeline that helps to filter fake reviews, classify sentiment through three different models, determine what topics the customers are actually talking about, and assign a score to a review based on its information quality, all within the constraints of the 12 GB memory limit of Google Colab’s free tier.

Methodology

Detection of Fake Reviews: An automated rule-based filter, which uses 7 weighted signals—unconfirmed purchase, review length, generic/bot-like language, low number of helpful votes, high volume of posts within a small time window, sentiment/rating discrepancy based on VADER sentiment analysis, and being part of the Vine program—to detect suspect reviews using only raw unlabelled data.
Features Engineering: 18 features: text-related (word count, readability, lexical polarity), timing-related, and interactions.
Sentiment Classification: Three different models were trained on the same 80/20 train/test split based on the MD5 hash of the review ID: Logistic Regression (streaming SGD), Random Forest (SVD-reduced TF-IDF and engineered features), and DistilBERT (fine-tuned transformer).
Topic Modeling: LDA (8 topics predefined) and BERTopic (automatic topic detection) independently, and then cross-checked by sentiment.
Quality Score for Review: Combined score of length, sentiment strength, community helpfulness, and authenticity.

Results

ModelAccuracyWeighted F1Logistic Regression87.4%0.8579Random Forest85.9%0.8185DistilBERT89.0%0.8745


5.8% of reviews were flagged as suspicious by the fake-review filter.
All three classifiers exceeded the 85% accuracy target.
LDA identified 8 distinct topic groups; BERTopic independently confirmed a similar structure.
              
