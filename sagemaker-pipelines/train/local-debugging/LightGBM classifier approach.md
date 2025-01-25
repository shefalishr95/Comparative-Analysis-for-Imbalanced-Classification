LightGBM classifier approach
Preprocessing
Preprocessing numerical and categorical features
In a first step we will preprocess the numerical and categorical features using SKLearn’s ColumnTransformer.  For the numerical features (price) we decided to fill missing values using the median wine price and scale them using a StandardScaler. 

We preprocessed the categorical features by filling missing values with “other” and OneHot encoded them. 

We saved the output of the ColumnTransformer as a pandas DataFrame to concatenate it later with the vector representation of the text.

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
def preprocess_number():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
def preprocess_categories():
    return make_pipeline(
       SimpleImputer(strategy="constant", fill_value="other", missing_values=np.nan),
       OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
def create_preprocessor():
    transformers = [
        ("num_preprocessor", preprocess_number(), [NUMERICAL_FEATURE]),
        ("cat_preprocessor", preprocess_categories(), [CATEGORICAL_FEATURE]),
    ]
    return ColumnTransformer(transformers=transformers, remainder="drop")
column_transformer = create_preprocessor()
column_transformer.set_output(transform="pandas")
preprocessed_num_cat_features_df = column_transformer.fit_transform(
    train_df[[NUMERICAL_FEATURE, CATEGORICAL_FEATURE]]
)
Extracting text vector representation with a transformer model
We then moved on to the preprocessing of the text features. We decided to use distilbert-base-uncased as the base transformer model to extract the vector representation of the wine description. 

BERT-type models use stacks of transformer encoder layers. Each of these blocks processes the tokenized text through a multi-headed self-attention step and a feed-forward neural network, before passing outputs to the next layer. The hidden state is the output of a given layer. The [CLS] token (short for classification) is a special token that represents an entire text sequence. We choose the hidden state of the [CLS] token in the final layer as the vector representation of our wine descriptions.

In order to extract the [CLS] representations, we first transform the text inputs into a Dataset of tokenized PyTorch tensors.  

For this step we tokenized batches of 128 wine descriptions padded to a fixed length of 120 suitable for our wine descriptions of ~40 words. The max_length is one of the parameters which should be adjusted depending on the length of the text feature to prevent truncating longer inputs. The padding is necessary to ensure we will process fixed-shape inputs with the transformer model. The tokenizer returns a dictionary of input_ids, attention_mask and token_type_ids. Only the input_ids (the indices of each token in the wine description) and the attention mask (binary tensor indicating the position of the padded indice) are required inputs to the model. 

The code for the tokenization is shown below:

from datasets import Dataset
from transformers import AutoTokenizer
MODEL_NAME = "distilbert-base-uncased"
def tokenized_pytorch_tensors(
        df: pd.DataFrame,
        column_list: list
    ) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    transformers_dataset = Dataset.from_pandas(df)
    def tokenize(model_inputs_batch: Dataset) -> Dataset:
        return tokenizer(
            model_inputs_batch[TEXT_FEATURE],
            padding=True,
            max_length=120,
            truncation=True,
        )
    tokenized_dataset = transformers_dataset.map(
        tokenize,
        batched=True,
        batch_size=128
    )
    tokenized_dataset.set_format(
        "torch",
        columns=column_list
    )
    
    columns_to_remove = set(tokenized_dataset.column_names) - set(column_list)
    tokenized_dataset = tokenized_dataset.remove_columns(list(columns_to_remove))
    return tokenized_dataset
print("Tokenize text in Dataset of Pytorch tensors")
train_df[TEXT_FEATURE] = train_df[TEXT_FEATURE].fillna("")
tokenized_df = tokenized_pytorch_tensors(
    train_df[[TEXT_FEATURE]],
    column_list=["input_ids", "attention_mask"]
)
Now we are in a position to extract the vector representation of the text using our pre-trained model. The code below shows how we returned the last hidden state of our tokenized text inputs and saved the transformer Dataset into a pandas DataFrame. 

import torch
from transformers import AutoModel
def hidden_state_from_text_inputs(df) -> pd.DataFrame:
    def extract_hidden_states(batch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        inputs = {
            k: v.to(device)
            for k, v in batch.items()
            if k in tokenizer.model_input_names
        }
        with torch.no_grad():
            last_hidden_state = model(**inputs).last_hidden_state
            # get the CLS token, which is the first one
            # [:, 0] gives us a row for each batch with the first column of 768 for each
            return {"cls_hidden_state": last_hidden_state[:, 0].cpu().numpy()}
    cls_dataset = df.map(extract_hidden_states, batched=True, batch_size=128)
    cls_dataset.set_format(type="pandas")
    return pd.DataFrame(
        cls_dataset["cls_hidden_state"].to_list(),
        columns=[f"feature_{n}" for n in range(1, 769)],
    )
print("Extract text feature hidden state")
hidden_states_df = hidden_state_from_text_inputs(tokenized_df)
print(f"Data with hidden state shape: {hidden_states_df.shape}") 
All that remains to be done before we can train our classifier is to concatenate the preprocessed features together.

print("Saving preprocessed features and targets")
preprocessed_data = pd.concat(
    [
        preprocessed_num_cat_features_df,
        hidden_states_df,
        train_df[TARGET]
    ],
    axis=1
)
Train a LightGBM model
Encode target and format input names
To train our classifier we first needed to encode our categorical target variable into an integer. To prevent issues when training the LightGBM classifier, we renamed the feature columns and removed non alpha-numeric characters from the names. 

We then trained our classifier as shown below:

import lightgbm as lgbm
features = [col for col in list(preprocessed_data.columns) if col not in [TARGET, "encoded_target"]]
# create the model
lgbm_clf = lgbm.LGBMClassifier(
    n_estimators=100,
    max_depth=10,
    num_leaves=10,
    objective="multiclass",
)
lgbm_clf.fit(preprocessed_data[features], preprocessed_data["encoded_target"])
Evaluate the results
We preprocessed our evaluation data in a consistent way and generated predictions. The predicted category in this case is the category with the highest model output score. We use the SKLearn accuracy_score method to compare the actual and predicted categories for this approach, achieving an accuracy of 0.81.

