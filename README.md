# Comparative Analysis of Classical and Anomaly Detection Methods for Imbalanced Mixed-Data Classification

This project systematically compares classical machine learning and anomaly detection models for handling highly imbalanced mixed-data classification in e-commerce. The experimental pipeline, built on AWS SageMaker, evaluates eight models, including logistic regression, ensemble methods, and deep learning approaches, on Amazon product listings (source: Asaniczka (September, 2023) Amazon Products Dataset 2023).

Key methodologies include Bayesian hyperparameter optimization using SageMaker’s `HyperparameterTuner` for model tuning and statistical testing to assess performance differences. Data preprocessing and feature engineering of text and numerical attributes are handled via PySpark in AWS Glue Interactive Sessions. Results will document model trade-offs in performance and computational efficiency.  

**Note:** This document serves as a proposal and will be updated as the project progresses.  


## Objectives 
The challenge of imbalanced data classification has been extensively studied in machine learning literature, both empirically and theoretically (Japkowicz & Stephen, 2002; Krawczyk, 2016; Liu et al., 2019; Sun et al., 2009). Research has established three primary approaches for handling class imbalance: data-level modifications, algorithm-level adaptations, and specialized evaluation metrics. While these approaches have shown promise, the optimal strategy often depends on specific data characteristics and application contexts.

Recent studies have explored the adaptation of classical and deep learning-based classification methods for anomaly detection tasks (Bergman & Hoshen, 2020; Steinwart et al., 2005), as well as the inverse - applying anomaly detection methods to classification problems (Gerych et al., 2019; Kong et al., 2020). However, given the fundamental differences in their underlying assumptions and optimization objectives (He, 2020), the effectiveness of this cross-application requires systematic evaluation.

The key research questions this project aims to address empirically are:
1. How do anomaly detection models compare to classical ML approaches in identifying bestselling products, given the inherent rarity of bestseller status and the mixed nature of product data (pricing, and other attributes)?
2. What is the comparative performance of traditional classifiers versus anomaly detection methods in:
- Precision of bestseller identification (reducing false positives)
- Recall of bestseller detection (minimizing missed opportunities)
- Computational efficiency (inference latency, resources to train e.g., GPU mem utilization etc.)

## Dataset Description

This project uses the Amazon Best Sellers dataset (Asaniczka, 2023) sourced from Kaggle, comprising 1,426,337 product listings across multiple categories. The dataset exhibits a significant natural class imbalance, with bestseller products representing <1% of total listings. 

### Data Characteristics

- Sample Size: 1,426,337 observations
- Features: 6 variables spanning multiple data types
- Class Distribution: <1% bestsellers vs 99.40% non-bestsellers

### Feature Description

1. Product Identifiers
   - asin: Unique Amazon product identifier (`string`)
   - productURL: Reference URL (`string`, unused in current analysis)
   - imgUrl: Product image URL (`string`, unused in current analysis)

2. Text Features
   - title: Product title (unstructured `text`)
   - Text features were excluded from model comparison due to storage limitations with embedding vectors
   - Initial tests with logit and XGBoost model do show a slight increase in precision when text features are included

3. Numerical Features
   - price: Current price in USD (`float`)
   - listPrice: Original price in USD (`float`)
   - reviews: Review count (`integer`)
   - stars: Average rating (`float`)

4. Categorical Features
   - category_id: Product category identifier (`integer`)

5. Target Variable
   - isBestSeller: Binary indicator (`boolean`)

### Dataset Selection Rationale

1. **Natural Imbalance**: Reflects real-world e-commerce class distribution
2. **Scale**: Sufficient size for robust model evaluation
3. **Multimodal Potential**: Image URLs and text features enable future extension to multimodal analysis

## Experiment Set-Up

Based on the challenges and mitigation strategies discussed, this section outlines the experimental framework that will be used  to compare classical machine learning and anomaly detection models for classifying Amazon bestsellers in a highly imbalanced, mixed-data. The experiments will leverage AWS Glue and SageMaker for scalable data processing and model training.


```mermaid
flowchart TD
    subgraph Data["Data Processing (AWS Glue)"]
        A[Raw Data] --> B[PySpark Preprocessing]
        B --> C[Feature Engineering]
        C --> D[Processed Data]
    end

    subgraph Split["Data Split"]
        D --> E[Train Data]
        D --> F[Validation Data]
        D --> G[Test Data]
    end

    subgraph Training["Model Training (SageMaker)"]
        E --> H[Classical Models]
        E --> I[Anomaly Detection]
        F --> |Hyperparameter Tuning| Training
        
        subgraph CM[Classical Models]
            H --> J[Logistic Regression]
            H --> K[Random Forest]
            H --> L[XGBoost]
            H --> M[SVM]
        end
        
        subgraph AD[Anomaly Detection]
            I --> N[Isolation Forest]
            I --> O[One-Class SVM]
            I --> P[k-NN]
            I --> Q[Autoencoder]
        end
    end

    subgraph Evaluation["Model Evaluation"]
        J & K & L & M & N & O & P & Q --> R[Performance Metrics]
        R --> T[Interpretability Analysis]
        F --> |Validation Performance| R
    end

    G --> |Final Evaluation| Evaluation
```


## Anticipated Pitfalls and Proposed Solutions

Applying anomaly detection models to classification problems, even imbalanced classification problems, is not a novel approach. While several studies have demonstrated the potential of using anomaly detection for imbalanced classification tasks, the effectiveness of this approach depends heavily on data characteristics and problem context. Before detailing the experimental setup, it is important to understand the potential pitfalls and challenges in applying these methods to e-commerce bestseller classification, particularly with mixed data types.

#### 1. Fundamental paradigm mistamtch
The primary pitfall in applying anomaly detection to bestseller classification stems from the fundamental difference in assumptions. While classification approaches aim to learn decision boundaries between defined classes, anomaly detection is inherently asymmetric, focusing on characterizing a single "normal" class and identifying deviations from it. Speifically, in context of our dataset:
- Non-bestsellers don't constitute a homogeneous "normal" class
- Areas where bestsellers and non-bestsellers share similar features may be mishandled
- Unusual non-bestsellers might be falsely flagged as bestsellers merely for being atypical

Recent studies have proposed sophisticated solutions like outlier scores (Kong et al., 2020) and generative models (Buitrago et al., 2018) to bridge this paradigm gap. However, systematic empirical evidence comparing basic implementations of both approaches remains limited. This project focuses on:
- *Direct Comparison*: Evaluating standard implementations of both paradigms, specifically selecting anomaly detection models (Isolation Forests, k-NN, One-Class SVM, Autoencoders) whose underlying assumptions *broadly* align with heterogeneous e-commerce data characteristics (some customizations might still be needed), rather than methods requiring strict distributional assumptions like Gaussian Mixture Models
- *Consistent Evaluation*: Using the same metrics and validation strategy across all models
- *Practical Viability*: Assessing computational requirements and implementation complexity

#### 2. Mixed-Data complexities in Anomaly Detection

The curse of dimensionality from text vectors can make normal points appear as outliers, while different scales and distributions across feature types can distort anomaly detection. Furthermore, algorithm-specific issues arise: Isolation Forests may over-split on text features, k-NN distances become unreliable in mixed spaces, and One-Class SVM kernel selection becomes complex. A critical additional challenge is feature overlap between bestseller and non-bestseller products in this mixed space - a bestselling product might share many text descriptions, similar price points, and category characteristics with non-bestsellers, making it difficult for anomaly detection algorithms to identify meaningful deviations.

**Key Risks:**
- Loss of important numerical signals (like price) in combined feature space
- Unreliable distance metrics in mixed high-dimensional space
- Feature overlap between classes making "anomalous" patterns unclear
- Normal bestseller characteristics might be indistinguishable from non-bestsellers in many dimensions.
- Text vectors dominating anomaly detection due to high dimensionality (node: current implementation does not include text vectors)

#### 4. Big Data Processing
Processing a large-scale e-commerce dataset (n=1.4M) with mixed data types presents significant computational challenges. Traditional in-memory processing becomes unfeasible with high-dimensional text vectors and multiple preprocessing steps. Additionally, the iterative nature of model tuning and validation requires efficient data handling strategies, particularly when running multiple models for comparison.

The project will implement a two-phase computational approach. Initial data preprocessing and feature engineering is handled through PySpark in AWS Glue Interactive Sessions, enabling distributed processing of the large dataset. The processed data will then be passed to SageMaker for text vectorization, model training, and validation, leveraging its optimized infrastructure for ML workflows. This separation allows efficient large-scale data processing while maintaining robust model training and validation capabilities within SageMaker's controlled environment.

#### 5. Model Interpretability and Business Value
Perhaps one of the most important questions for this analysis is to answer: why is a product a bestseller? Anomaly detection models, particularly when applied to mixed data, often operate as "black boxes" with scores that are difficult to interpret in business terms. For practical application, stakeholders need clear insights into model decisions.

This project will implement three key approaches to enhance model interpretability:

1. **Prediction Uncertainty Quantification**:
Both modeling paradigms will require different approaches for uncertainty estimation. Classical models will utilize probabilistic outputs and bootstrap sampling, while for anomaly detection models, I will explore ensemble-based uncertainty estimates through bootstrap aggregation of anomaly scores to establish confidence bounds in predictions.

2. **Local Feature Attribution**:
SHAP (SHapley Additive exPlanations) values will be implemented to understand individual predictions, providing insights into which features contribute most to bestseller classification. 

We may limit interpretation to the best-performing model to save computational resources.

## Limitations and Future Work

1. This project **does not incorporate time series data**, which may limit the insights gained on how best-selling products vary over time. Seasonal trends and evolving consumer preferences can significantly influence product performance and best-seller status.

2. The project relies solely on one dataset, which may not represent the broader e-commerce landscape. Findings derived from this dataset **may not generalize well to other applications**. Future work should include multiple datasets from various sources to validate the models and ensure their applicability across different contexts.

3. This project also **ignores image features**, which are likely to be key predictors in consumer decisions and preferences in e-commerce settings. Images can provide valuable insights into product attributes and consumer preferences. The next phase can focus on multimodal dataset classification.

4. Optimization, hyperparameter tuning, and interpretation will be **constrained by available computational resources**.

5. While the project emphasizes empirical results, the **use of anomaly detection methods for imbalanced classification is controversial**. As mentioned in the pitfalls section, these approaches represent a fundamental paradigm shift and may not be suitable for production environments without rigorous testing and validation.

6. The dataset had **insufficient features for predicting best sellers**. Future work could benefit from incorporating user behavior data, such as browsing history and click-through rates. This additional information may help refine the models and improve predictions for best-selling products.
