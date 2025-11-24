# DS_Training_2025_CS50
Scripts and examples from CS50 Training completed 2025.

# CS50 AI + Dimensions: Applied Notebooks

This folder contains six Jupyter notebooks that adapt **CS50 AI** concepts to **Dimensions-style research data** (grants, publications, institutions, topics). Each notebook is self-contained but designed to work together as a mini-course.


---

## 1. `CS50_ML.ipynb`  
**Title:** Core Machine Learning with Dimensions Grant Data  

**CS50 Topic:** Supervised & Unsupervised Learning  
**Focus:**

- Supervised learning
  - Classification: k-NN, Perceptron, SVM on `is_ai_ml`
  - Regression: predict `citations_5yr` from topic scores & funding
- Loss functions
  - 0–1 loss (classification error)
  - L₁ (MAE) and L₂ (MSE) for regression
- Overfitting & regularization
  - Logistic regression with different `C` values
- Validation
  - Holdout (train/test split)
  - k-fold cross-validation
- Unsupervised learning
  - k-means clustering on grant features

**Key data used:**

- `grants` DataFrame with columns like:
  - `grant_id`, `topic_ai_score`, `topic_bioinfo_score`, `topic_data_repo_score`
  - `total_funding`, `citations_5yr`, `is_ai_ml`

---

## 2. `CS50_NeuralNetworks.ipynb`  
**Title:** Artificial Neural Networks with Dimensions Grant Data  

**CS50 Topic:** Neural Networks & Deep Learning  
**Focus:**

- Feed-forward networks (ANNs)
  - Binary classification: AI vs non-AI grants using Dense layers (ReLU + sigmoid)
  - Regression: predict `citations_5yr` with a DNN
- Logical functions
  - Learn AND/OR-like behavior from binary features (e.g., `has_ai`, `has_repo`)
- Deep neural networks
  - Multi-layer models with hidden layers
- Overfitting & dropout
  - Compare models with and without `Dropout`
- Sequence-style examples
  - 1D CNN on yearly citation sequences (`citations_FY19`–`citations_FY25`)
  - LSTM/RNN classification based on citation time series

**Key data used:**

- `grants` with:
  - Topic scores, `total_funding`, `citations_5yr`
  - Per-year citation columns: `citations_FY19` … `citations_FY25`
  - `is_ai_ml` labels

---

## 3. `CS50_NLP.ipynb`  
**Title:** Natural Language Processing with Dimensions Grant Data  

**CS50 Topic:** NLP & Language Models  
**Focus:**

- Core NLP operations
  - Tokenization, n-grams, Bag-of-Words
  - Markov (n-gram) text generation using grant abstracts
- Text classification
  - Naive Bayes (MultinomialNB) to classify AI vs non-AI from abstracts
- Word representations
  - Word2Vec embeddings trained on grant text
- Syntax & semantics
  - Simple CFG parsing using NLTK
- Neural NLP
  - LSTM text classifier on abstracts
  - Transformer-based sentence embeddings (via TF Hub)
- Named entity recognition (NER) on abstracts (institutions, locations, etc.)

**Key data used:**

- `grants` or `papers` with:
  - `abstract`, `title`, `is_ai_ml`

---

## 4. `CS50_Search.ipynb`  
**Title:** Optimization & Search with Dimensions Grant Data  

**CS50 Topic:** Search, Local Search, Optimization, CSPs  
**Focus:**

- Local search (hill climbing)
  - Assign grants to initiatives to maximize abstract–initiative similarity
  - Reassign grants to programs to minimize mismatch
- Linear programming (LP) with `pulp`
  - Maximize total citations under a budget
  - Add equity constraints (e.g., LMIC share, initiative share)
- Constraint satisfaction / integer programming
  - Reviewer assignment:
    - Min/max reviewers per grant
    - Reviewer load
    - Conflicts of interest
    - Optional topic similarity threshold

**Key data used:**

- `grants`: initiatives, programs, funding, citations, country/income, etc.
- `reviewers`: expertise, max load
- `conflicts`: grant–reviewer pairs to exclude

---

## 5. `CS50_Uncertainty.ipynb`  
**Title:** Modeling Uncertainty & Probabilistic Reasoning with Dimensions Data  

**CS50 Topic:** Probability, Bayesian Reasoning, Markov Models, Risk  

**Focus:**

1. **Modeling topic trends**
   - Beta-Binomial model for “probability of increase in topic publications next year”

2. **Bayesian inference for funding**
   - Naive Bayes–style `P(Funding | Proposal Features)` using binned topic scores & institution tiers

3. **Markov models**
   - Topic transition matrix and simulated trajectories for topic evolution

4. **Missing data**
   - EM-like imputation of missing `citations_5yr` using Gaussian mixtures

5. **Risk assessment for collaborations**
   - Posterior probabilities of low/medium/high impact for collaborative projects (Dirichlet smoothing)

**Key data used:**

- `topic_trends`: topic, year, `n_pubs`
- `grants`: topic scores, institution tier, `funding_awarded`, `citations_5yr`
- `collabs`: collaborative projects with `citations_5yr`, topic, institutions

---

## 6. `CS50_Search.ipynb`  
**Title:** Knowledge Representation & Logic with Dimensions Research Data  

**CS50 Topic:** Knowledge Representation & Inference  

**Focus:**

- Propositional logic modeling
  - Symbols like `P`, `Q`, `R` representing paper/journal/institution/topic facts
  - Simple implications like `P ∧ Q → R`

- Knowledge base for trends
  - Facts: `increase(T)` from publication counts
  - Rules: `increase(T) → gaining_interest(T)` with forward chaining

- Inference & entailment
  - Journal peer-review rules:
    - `published_in(P,J) ∧ peer_reviewed_journal(J) → peer_reviewed(P)`

- Model checking for hypotheses
  - Test statements like:
    - “Collaboration between institutions A and B → high citations”

- Logic-programming style queries
  - Simple rule-based queries over collaboration facts to find high-impact / AI-related collaborations

**Key data used:**

- `papers`: paper–journal–topic–institution info
- `institutions`: topic specialty
- `collabs`: collaboration edges + citations
- `topic_trends`: for trend rules

---
