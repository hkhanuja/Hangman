import torch
import random
import torch.nn as nn
import math
import numpy as np
import string
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_scheduler
import torch.nn.functional as F
import os

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


MAX_NUM_INPUTS = 29
print('Max word length: {}'.format(MAX_NUM_INPUTS))

val_words = []
with open('./val_words.txt','r') as file:
    for line in file:
        val_words.append(line.strip())

max_len = MAX_NUM_INPUTS+2

class BertHangman(nn.Module):
  def __init__(self, base_model, vocab_size = 26, hidden_size = 328):
    super(BertHangman,self).__init__()
    self.model = base_model.bert

    for param in self.model.parameters():
        param.requires_grad = False

    for layer in self.model.encoder.layer[10:]:
        for param in layer.parameters():
            param.requires_grad = True


    self.dropout = nn.Dropout(0.2)
    self.fc = nn.Linear(768+vocab_size, hidden_size)
    self.relu = nn.ReLU()
    self.output = nn.Linear(hidden_size, vocab_size)


  def forward(self, input_ids=None, attention_mask=None, guessed_letters=None):

    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    sequence_output = outputs.last_hidden_state
    sequence_output = self.dropout(sequence_output)

    
    sequence_output = torch.cat((sequence_output, guessed_letters.unsqueeze(1).repeat(1, sequence_output.shape[1], 1)), dim=2)

    logits = self.fc(sequence_output)
    logits = self.relu(logits)
    logits = self.output(logits)


    logits[guessed_letters.unsqueeze(1).repeat(1, sequence_output.shape[1], 1) == 1] = -float("inf")

    mask_token_indices = (input_ids == bert_tokenizer.mask_token_id).unsqueeze(-1)

    masked_logits = logits.masked_fill(~mask_token_indices, float('-inf'))

    max_masked_logits = masked_logits.max(dim=1).values 

    return max_masked_logits
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hangman_model = BertHangman(bert_model, vocab_size = 26).to(device)

checkpoint = torch.load('bert_rl/bert_rl_model_epoch_490000.pth', map_location=device)  

# Restore model state
new_state_dict = {key.replace("bert.", "model."): value for key, value in checkpoint['model_state_dict'].items()}
hangman_model.load_state_dict(new_state_dict)
hangman_model.eval()
episode = checkpoint['episode']

print(f"Model restored from episode {episode}")


correct_words = []
incorrect_words = []
char_to_index = {chr(i + ord('a')): i for i in range(26)}

with torch.no_grad():
  for word in tqdm(val_words):
    guessed = set()
    guessed_list = []
    word_list = list(word)
    word_set = set(word)
    lives = 6
    label_letters = [char_to_index[char] for char in word_list]
    # print(word)
    letters_to_be_masked = word_set - guessed

    while lives>=0 and len(letters_to_be_masked)>0:
        tokenized_word = bert_tokenizer(
          " ".join(list(word)),
          return_tensors="pt",
          truncation=True,
        )

        input_ids = tokenized_word.input_ids.clone().squeeze(0)
        attention_mask = tokenized_word.attention_mask.squeeze(0)
        guessed_letters = torch.zeros(26, dtype=torch.int64)
        for i in range(len(word_list)):
            if word[i] in letters_to_be_masked:
                input_ids[i+1] = bert_tokenizer.mask_token_id
        
        for i in range(len(guessed_list)):
            guessed_letters[ord(guessed_list[i]) - ord('a')] = 1
        
        logits = hangman_model(torch.unsqueeze(input_ids,0).to(device), torch.unsqueeze(attention_mask,0).to(device), torch.unsqueeze(guessed_letters,0).to(device))
        logits = F.softmax(logits, dim=-1)
        predicted_letter_index = torch.argmax(logits, dim=-1)

        guessed_letter = chr(ord('a')+predicted_letter_index)
        guessed_list.append(guessed_letter)
        guessed.add(guessed_letter)

        # print(input_ids)
        # print(guessed_letter)


        if guessed_letter not in letters_to_be_masked:
            lives -=1
        letters_to_be_masked  = word_set - guessed
        # print(letters_to_be_masked)
        # print(lives)
    if lives<0:
       incorrect_words.append(word)
    elif len(letters_to_be_masked)==0:
      correct_words.append(word)

with open("bert_rl/bert_rl_correct_words.txt", "w") as f:
    for item in correct_words:
        f.write(f"{item}\n")

print('Number of words correcrtly predicted', len(correct_words))
print('Number of words inorrecrtly predicted', len(incorrect_words))

print('Fraction of words correcrtly predicted', len(correct_words)/len(val_words))
print('Fraction of words incorrecrtly predicted', len(incorrect_words)/len(val_words))