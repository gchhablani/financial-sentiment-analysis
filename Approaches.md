# Approaches/Models
The same approaches can be used for all the following tasks, we can pick a few of these, or all.
1. ML-Based models:
    - XGBoost(Trees) fine-tuned on TF-IDF Vectors from the sentences.
2. DL-Based models:
    - Only Fine-tuning - Training the given model (pre-trained or otherwise on the given labels:
        - Sequence Models:
            - LSTM
        - Transformer Models:
            - BERT
                - base
                - large (more parameters than base)
            - FinBERT
                - base
            - ELECTRA (Different kind of BERT, proven to perform much better)
                - small
                - large 
            - SpanBERT (Different kind of BERT, pre-trained especially for sentences)
                - base
                - large
    - Pre-training then fine-tuning - The models are pre-trained using self-supervision (no labels) on OUR text data only. Then fine-tuning is done with OUR labels. This is similar to one of the approaches of FinBERT, usually supposed to improve performance.
        - Transformer Models:
            - BERT
                - base
                - large (more parameters than base)
            - FinBERT
                - base
            - ELECTRA (Different kind of BERT, proven to perform much better)
                - small
                - large 
            - SpanBERT (Different kind of BERT, pre-trained especially for sentences)
                - base
                - large

**Note: All of the transformer models can be used with GAN-BERT approach as only the path of the model has to be changed** 


# For Risk Profile Classification
## Single-Class Classification
Description: Classsification with three labels - Risk Averse, Risk Seeker, Risk Neutral. The text can belong to only one of these classes at a time.

# For Sentiment Classification

## Single-Class Classification (Positive, Negative, Neutral)
Description: We can make the problem into a single class classification problem by select either "Positive", "Negative", and marking the sentence as "Neutral" if it has both positive and negative words. Sentences which don't have any word will become "Neutral" by default. This way, each sentence will have ONLY ONE CLASS. 

**Note: We can predict on the data using the pre-trained FinBERT model, without any training, as it already does "positive", "negative", "neutral" sentiment classification. This way, we can also compare the FinBERT model before and after fine-tuning.**


## Multi-Class Classification (Positive, Negative, Litigious, Uncertainty, StrongModal, WeakModal)
Description: We can make the problem into a multi-class classification problem. This way, each sentence will have MULTIPLE CLASSES. Example: "I will invest in gold", 0, 1, 0 1, 1, 0, 0. This example is "Negative", "Uncertainty", and "StrongModal" at the same time.

**Note: Such a model isn't readily available on the Transformers library that we are using and I will have to code each of the following, including the transformer models. Will take some time, but can be done. If we don't want all of them, it would be easier for me.**



## Multi-Label Regression (Positive, Negative, Litigious, Uncertainty, StrongModal, WeakModal)
Description: We can make the problem into a multi-label regression problem. This way, each sentence will have MULTIPLE LABELS. For each category, we store the count of the words. Example: "I will invest in gold", 0, 2, 0 1, 3, 0, 0. This will be normalized, say using min-max scaling, across all examples:  [0., 0.66666667, 0., 0.33333333, 1.,0., 0.]. This example has 0.667 negative, 0.33 uncertain, and 1.0 strong modal rating. 

**Note: Such a model isn't readily available on the Transformers library that we are using and I will have to code each of the models, including the transformer models. Will take some time, but can be done. If we don't want all of them, it would be easier for me.**


# Handwriting-Correlation

We can use the Pos, Neg, Neutral labels, same as that in **Single-Class Classification** and see if there a correlation between the kind of handwriting and the class. Example, Positive class is highly correlated with "Cursive Slanted" handwriting.