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
