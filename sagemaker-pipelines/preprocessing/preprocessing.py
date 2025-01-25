import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


## Text feature integration - for later

# from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer, DistilBertModel
# import torch
# from sklearn.base import BaseEstimator, TransformerMixin
# from datasets import Dataset

# MODEL_NAME = "distilbert-base-uncased"
# text_feature = "title"

# # 1. Function to tokenize text and return tokenized dataset
# def convert_to_tokenized_tensors(df: pd.DataFrame, column_list: list) -> Dataset:
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     transformers_dataset = Dataset.from_pandas(df)
    
#     def tokenize(model_inputs_batch: Dataset) -> Dataset:
#         return tokenizer(
#             model_inputs_batch[text_feature],
#             padding=True,
#             max_length=200,
#             truncation=True,
#         )
    
#     tokenized_dataset = transformers_dataset.map(
#         tokenize, batched=True, batch_size=128
#     )
#     tokenized_dataset.set_format("torch", columns=column_list)
#     columns_to_remove = set(tokenized_dataset.column_names) - set(column_list)
#     tokenized_dataset = tokenized_dataset.remove_columns(list(columns_to_remove))
#     return tokenized_dataset

# # 2. Function to extract hidden states
# def hidden_state_from_text_inputs(df: Dataset) -> pd.DataFrame:
#     def extract_hidden_states(batch):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#         model = AutoModel.from_pretrained(MODEL_NAME)
#         inputs = {
#             k: v.to(device)
#             for k, v in batch.items()
#             if k in tokenizer.model_input_names
#         }
#         with torch.no_grad():
#             last_hidden_state = model(**inputs).last_hidden_state
#             return {"cls_hidden_state": last_hidden_state[:, 0].cpu().numpy()}
    
#     cls_dataset = df.map(extract_hidden_states, batched=True, batch_size=128)
#     cls_dataset.set_format(type="pandas")
#     return pd.DataFrame(
#         cls_dataset["cls_hidden_state"].to_list(),
#         columns=[f"feature_{n}" for n in range(1, 769)],
#     )

## Categorical and numerical features integration

num_features = ['stars', 'reviews', 'price', 'listPrice','boughtInLastMonth', 'rating_weighted_reviews', 'percent_rank', 'price_log', 'title_length']
cat_features = ['has_listPrice', 'category_name', 'isPopular']
text_features = ['title']
target =['isBestSeller']

np.random.seed(13)

if __name__ == "__main__":
    base_dir = "/opt/ml/processed"

    df = pd.read_parquet(
        f"{base_dir}/input/transformed-data.paraquet")


    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )

    X = df[num_features + cat_features]
    y = df[target]

    X_prior = preprocessor.fit_transform(X)
    y_prior = np.array(y).reshape(-1, 1)

    print("X_prior shape:", X_prior.shape)
    print("y_prior shape:", y_prior.shape)



    combined = np.concatenate((y_prior, X_prior), axis=1) # Concatenate target with features for shuffling
    print(f"Shape of combined data: {combined.shape}")

    np.random.shuffle(combined) # Shuffle the data

    # Split the data into train, validation, and test sets in 70:15:15 ratio
    train, validation, test = np.split(combined, [int(0.7 * len(X)), int(0.85 * len(X))])

    print("Train shape:", train.shape)
    print("Validation shape:", validation.shape)
    print("Test shape:", test.shape)

    # Save the data to csv files in the processed directory

    pd.DataFrame(train).to_parquet(f"{base_dir}/train/train.parquet", header=False, index=False)
    pd.DataFrame(validation).to_parquet(f"{base_dir}/validation/validation.parquet", header=False, index=False)
    pd.DataFrame(test).to_parquet(f"{base_dir}/test/test.parquet", header=False, index=False)
