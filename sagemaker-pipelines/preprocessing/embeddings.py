import torch
from transformers import DistilBertTokenizer, DistilBertModel

from transformers import pipeline
import random
from text_cleaning import process_text

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertModel.from_pretrained("bert-base-uncased")

RandomSeed = 52
random.seed(RandomSeed)

torch.manual_seed(RandomSeed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RandomSeed)


# text = process_text("I love Columbia University so much~! 😍")
text = "I love learning about Transformers"


def tokenize_titles(text):
    return tokenizer(text=text, padding=True, truncation=False,
                     return_tensors='pt',
                     return_attention_mask=True, return_length=False)

encoded_sample = tokenize_titles(text)


with torch.no_grad():
    output_sample = model(**encoded_sample, return_dict=True, output_hidden_states=True, output_attentions=True)
    word_embeddings = output_sample.last_hidden_state

print("Word embeddings shape:", word_embeddings.shape)

####### Decode?
token_ids = encoded_sample['input_ids']
decodedText = tokenizer.decode(token_ids[0], skip_special_tokens=True)
tokenizedtxt = tokenizer.tokenize(decodedText)

print(f"Decoded Text: {decodedText}")
print(f"Tokenized Text: {tokenizedtxt}")
print(f"Token IDs: {token_ids}")