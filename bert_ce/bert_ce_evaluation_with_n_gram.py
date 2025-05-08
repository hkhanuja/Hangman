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

train_words = []
with open('./train_words.txt','r') as file:
    for line in file:
        train_words.append(line.strip())


from collections import defaultdict

def generate_ngram_counts(word_list):
    """
    Generates trigram counts, left bigram counts, and right bigram counts from a list of words.
    Also normalizes these counts into 26-length vectors.

    Parameters:
    - word_list (list): List of words.

    Returns:
    - ngram_counts (dict): Trigram counts in the form {context: {letter: count}}.
    - left_bigram_counts (dict): Bigram counts for left context {left: {letter: count}}.
    - right_bigram_counts (dict): Bigram counts for right context {right: {letter: count}}.
    """
    ngram_counts = defaultdict(lambda: defaultdict(int))
    left_bigram_counts = defaultdict(lambda: defaultdict(int))
    right_bigram_counts = defaultdict(lambda: defaultdict(int))

    for word in word_list:
        length = len(word)

        for i in range(length - 2):  
            left, middle, right = word[i], word[i+1], word[i+2]

            context = left + right
            ngram_counts[context][middle] += 1

        for i in range(length - 1):
            left, next_letter = word[i], word[i + 1]
            left_bigram_counts[left][next_letter] += 1

        for i in range(1, length): 
            right, prev_letter = word[i], word[i - 1]
            right_bigram_counts[right][prev_letter] += 1

    def normalize_to_vector(counts):
        normalized_vectors = {}
        for context, letter_counts in counts.items():
            total_count = sum(letter_counts.values())  
            vector = [0] * 26 

            for letter, count in letter_counts.items():
                index = ord(letter) - ord('a') 
                if 0 <= index < 26:
                    vector[index] = count / total_count
            normalized_vectors[context] = vector

        return normalized_vectors

    normalized_ngram_counts = normalize_to_vector(ngram_counts)
    normalized_left_bigram_counts = normalize_to_vector(left_bigram_counts)
    normalized_right_bigram_counts = normalize_to_vector(right_bigram_counts)

    return normalized_ngram_counts, normalized_left_bigram_counts, normalized_right_bigram_counts

ngram_counts, left_bigram_counts, right_bigram_counts = generate_ngram_counts(train_words)

print("Normalized Trigram Counts:", ngram_counts)
print("Normalized Left Bigram Counts:", left_bigram_counts)
print("Normalized Right Bigram Counts:", right_bigram_counts)



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

    return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hangman_model = BertHangman(bert_model, vocab_size = 26).to(device)

checkpoint = torch.load('bert_ce/bert_ce_model_epoch_100.pth', map_location=device)  

hangman_model.load_state_dict(checkpoint['model_state_dict'])
hangman_model.eval()
epoch = checkpoint['epoch']
train_losses = checkpoint['train_losses']

print(f"Model restored from epoch {epoch}")
print(train_losses)
print(checkpoint['val_losses'])

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
            else:
                label_letters[i] = -100
        
        for i in range(len(guessed_list)):
            guessed_letters[ord(guessed_list[i]) - ord('a')] = 1
        
        logits = hangman_model(torch.unsqueeze(input_ids,0).to(device), torch.unsqueeze(attention_mask,0).to(device), torch.unsqueeze(guessed_letters,0).to(device))
        logits = F.softmax(logits, dim=-1)
        logits = logits.squeeze()[1:-1,:]
        ngram_probabilities = torch.zeros_like(logits)

        label_mask = [x != -100 for x in label_letters]

        missing_count = sum(label_mask)

        if len(word) > 3 and missing_count <= 2:
           for i in range(len(word)):
                left = None
                right = None
                if i!=0 and i!=len(word)-1 and label_letters[i] != -100:
                    if label_letters[i-1] == -100:
                        left = word[i-1]
                    if label_letters[i+1] == -100:
                        right = word[i+1]
                    if left is not None and right is not None:
                        context = left+right
                        ngram_probabilities[i] = torch.tensor(ngram_counts[context])
                    elif left is None and right is None:
                        pass
                    elif left is None:
                        ngram_probabilities[i] = torch.tensor(right_bigram_counts[right])
                    elif right is None:
                        ngram_probabilities[i] = torch.tensor(left_bigram_counts[left])
                elif i==0 and label_letters[i] != -100:
                    if label_letters[i+1] == -100:
                        right = word[i+1]
                    if right is not None:
                        ngram_probabilities[i] = torch.tensor(right_bigram_counts[right])
                elif i==len(word)-1 and label_letters[i] != -100:
                    if label_letters[i-1] == -100:
                        left = word[i-1]
                    if left is not None:
                        ngram_probabilities[i] = torch.tensor(left_bigram_counts[left])
        alpha = 0.8
        logits = alpha *logits +  (1- alpha) * ngram_probabilities
        contributing_logits = logits[label_mask]

        max_prob_for_each_sequence = torch.max(contributing_logits, dim=-1).values
        max_prob_char_for_each_sequence = torch.argmax(contributing_logits, dim=-1)
        max_prob_index_out_of_max_prob_from_each_sequence = torch.argmax(max_prob_for_each_sequence)
        max_prob_char = max_prob_char_for_each_sequence[max_prob_index_out_of_max_prob_from_each_sequence]
    
        guessed_letter = chr(ord('a')+max_prob_char)
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

with open("bert_ce/bert_ce_with_n_gram_correct_words.txt", "w") as f:
    for item in correct_words:
        f.write(f"{item}\n")

print('Number of words correcrtly predicted', len(correct_words))
print('Number of words inorrecrtly predicted', len(incorrect_words))

print('Fraction of words correcrtly predicted', len(correct_words)/len(val_words))
print('Fraction of words incorrecrtly predicted', len(incorrect_words)/len(val_words))