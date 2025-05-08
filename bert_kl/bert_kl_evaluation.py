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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

words = []
with open('./words_250000_train.txt','r') as file:
    for line in file:
        words.append(line.strip())
words = list(np.random.permutation(words))

train_val_split_idx = int(len(words) * 0.8)
print('Training with {} words'.format(train_val_split_idx))

MAX_NUM_INPUTS = max([len(i) for i in words])
print('Max word length: {}'.format(MAX_NUM_INPUTS))

train_words = words[:train_val_split_idx]
val_words = words[train_val_split_idx:]


class BERTHangman(nn.Module):
  def __init__(self, vocab_size = 26, hidden_size = 512):
    super(BERTHangman,self).__init__()
    self.model = bert_model.bert

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

      logits[guessed_letters.unsqueeze(1).repeat(1, sequence_output.shape[1], 1) == 1] = -1e9

      mask_token_indices = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)  
      batch_size = input_ids.shape[0]
      label_size = logits.shape[-1]

      mean_masked_logits = torch.zeros((batch_size, label_size), device=logits.device)

      for i in range(batch_size):
          mask_positions = mask_token_indices[mask_token_indices[:, 0] == i][:, 1]

          if len(mask_positions) > 0:
              mean_masked_logits[i] = logits[i, mask_positions].mean(dim=0)

      return mean_masked_logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hangman_model = BERTHangman(vocab_size = 26).to(device)
# load_dir = '/home/hkhanuja3/trexquant/ce'
epoch = 63
# model_load_path = os.path.join(load_dir, f"model_epoch_{epoch}_ce.pth")
checkpoint = torch.load('model_epoch_63_kl.pth', map_location=device)  

# Restore model state
hangman_model.load_state_dict(checkpoint['model_state_dict'])
hangman_model.eval()
epoch = checkpoint['epoch']
train_losses = checkpoint['train_losses']

print(f"Model restored from epoch {epoch}")
print(train_losses)

correct_words = []
incorrect_words = []
char_to_index = {chr(i + ord('a')): i for i in range(26)}

val_words = random.sample(val_words, 2000)

with torch.no_grad():
  for word in tqdm(val_words):
    guessed = set()
    guessed_list = []
    word_list = list(word)
    word_set = set(word)
    lives = 6
    # print(word)
    letters_to_be_masked = word_set - guessed
    while lives>=0 and len(letters_to_be_masked)>0:
      tokenized_word = tokenizer(
          " ".join(list(word)),
          return_tensors="pt",
          truncation=True,
    )

      input_ids = tokenized_word.input_ids.clone().squeeze(0)
      attention_mask = tokenized_word.attention_mask.clone().squeeze(0)
      guessed_letters = torch.zeros(26, dtype=torch.int64)

      for i in range(len(word_list)):
        if word[i] in letters_to_be_masked:
            input_ids[i+1] = 103

      for i in range(len(guessed_list)):
        guessed_letters[ord(guessed_list[i]) - ord('a')] = 1

      remaining_letters = word_set - guessed
      remaining_letters = list(remaining_letters)


      logits = hangman_model(torch.unsqueeze(input_ids,0).to(device), torch.unsqueeze(attention_mask,0).to(device), torch.unsqueeze(guessed_letters,0).to(device))
      logits = torch.softmax(logits, dim=-1)
      max_prob_index = torch.argmax(logits, dim=-1)

      guessed_letter = chr(ord('a')+max_prob_index)
      guessed_list.append(guessed_letter)
      guessed.add(guessed_letter)
    #   print(input_ids)
    #   print(guessed_letter)


      if guessed_letter not in remaining_letters:
        lives -=1
      letters_to_be_masked  = word_set - guessed
    #   print(letters_to_be_masked)
    #   print(lives)
    if lives<0:
       incorrect_words.append(word)
    elif len(letters_to_be_masked)==0:
      correct_words.append(word)

with open("kl_correct_words_63_epochs.txt", "w") as f:
    for item in correct_words:
        f.write(f"{item}\n")

print('Number of words correcrtly predicted', len(correct_words))
print('Number of words inorrecrtly predicted', len(incorrect_words))

print('Fraction of words correcrtly predicted', len(correct_words)/len(val_words))
print('Fraction of words incorrecrtly predicted', len(incorrect_words)/len(val_words))