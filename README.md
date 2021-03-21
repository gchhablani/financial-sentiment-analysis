# Analyzing Handwritten Text to Understand Financial Sentiments/Behaviour

## Potential Ideas
### Data Statistics
- [x] WordClouds after removing Stopwords
    - [x] Parts-of-Speech, especially Nouns
    - [x] Financial Terms - buy, sell, debt, funds, invest, stocks, share, gold, property, real estate
- [x] Top-K words in each PoS category using Stanford PoS
    - [x] Bar-chart of the word-frequencies
    
- Noun-verb combinations, possible, need to check how to select one noun/verb from the sentence.

- Demographics vs Risk-Capacity 
    - Gender (Bar Chart)
    - Age (Scatter Plot) or Groups with Bar Char
    - Marital (Bar Chart)
    - Religion (Bar Chart)
    - Income (Scatter Plot) or Groups with Bar Chart
    - Handwriting Type

- Radar Chart for Investment Percentages in the MCQ.
    - Age Groups
    - Gender
    - Income Groups
    - Class Labels
    - Handwriting Type

- Categorize from Answers into - Misc, Tour/Travel, Education, MF, B, GCJ, AO, Debt, Ins, Pro, PPF. Plot a radar chart for this against labels, age groups, etc.


### Embedding-based Analysis
We will use text based embeddings:

- Using BERT/FinBERT (Before and After Fine-tuning)
    - Represent embeddings from these models
    - Make tSNE/PCA/UMAP based plots for words and see how they match
    - Check if we can segregate these based on the labels, age, income, gender - Embedding Representation 

- Using FinBERT only
    - Give sentiment labels based on FinBERT
    - Analyze stuff with respect to labels, in a similar fashion as EDA.

### Classification Task

- Fine-tune BERT/FinBERT on the labels with just test.
- Understand importance of the words based on either of the networks in predicting the category using IG.
- Incorporate embeddings+this information for other features provided and learn from them using a simple neural network. [768->768+x]
- Incorporate embedding with only the handwriting information and test again.
- Handwriting vs labels correlation.
- Repeat this with IG to get importance of these features.
- Standalone XGBoost, Simple Neural Network, without text to learn.
- Maybe add multiple labels (or span level information) and repeat the process.
