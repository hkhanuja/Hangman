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

batch_size = 32
max_len = MAX_NUM_INPUTS+2

class HangmanTrainDataset(Dataset):
    def __init__(self, words, tokenizer, max_length):
        self.words = words
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.all_letters = set(string.ascii_lowercase)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        letters_in_word = set(word)

        word_list = list(word)
        mask_ratio = random.randint(4, 9) / 10
        num_masks = int(len(word) * mask_ratio)
        if num_masks == 0:
          num_masks=1
        mask_indices = random.sample(range(len(word)), num_masks)
        letters_to_be_masked = set([word_list[i] for i in mask_indices])

        tokenized_word = self.tokenizer(
            " ".join(list(word)),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length = self.max_length
        )

        input_ids = tokenized_word.input_ids.clone().squeeze(0)
        attention_mask = tokenized_word.attention_mask.squeeze(0)
        guessed_letters = torch.zeros(26, dtype=torch.int64)
        for i in range(len(word_list)):
          if word[i] in letters_to_be_masked:
            input_ids[i+1] = 103
          else:
            guessed_letters[ord(word[i]) - ord('a')] = 1


        random_guesses = random.randint(2, 4)
        letters_not_in_word = list(self.all_letters - letters_in_word)
        random_letters_guessed = random.sample(letters_not_in_word, random_guesses)

        for i in range(len(random_letters_guessed)):
          guessed_letters[ord(random_letters_guessed[i]) - ord('a')] = 1

        remaining_letters = letters_to_be_masked
        letter_frequency = Counter(word)
        rem_vector = np.zeros(26, dtype=np.float32)
        for letter in remaining_letters:
            rem_vector[ord(letter) - ord('a')] = letter_frequency[letter]

        rem_vector = rem_vector / np.sum(rem_vector)
        rem_tensor = torch.tensor(rem_vector, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": rem_tensor,
            "guessed_letters": guessed_letters
        }


train_dataset = HangmanTrainDataset(train_words, tokenizer, max_len)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for batch in train_dataloader:
    print(batch['input_ids'])
    print(batch['labels'])
    print(batch['guessed_letters'])
    break

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

print("Trainable parameters:")
for name, param in hangman_model.named_parameters():
    if param.requires_grad:
        print(name)

lr = 5e-5
optimizer = AdamW(hangman_model.parameters(), lr=lr)
loss_fct = nn.KLDivLoss(reduction="batchmean")

num_epochs = 70
num_training_steps = int( ( num_epochs * len(train_dataset) )/ batch_size)
warmup_steps = int(num_training_steps * 0.1)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps,
)

train_losses = []
save_frequency = 7
save_dir = 'kl'
for epoch in range(num_epochs):
    hangman_model.train()
    total_train_loss = 0
    epsilon = 1e-8

    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = hangman_model( batch['input_ids'], batch['attention_mask'], batch['guessed_letters'] )
        batch['labels'] += epsilon
        predicted_probs = F.log_softmax(outputs, dim=-1)
        loss = loss_fct(predicted_probs, batch['labels'])
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_train_loss = total_train_loss + (loss.item()*batch_size)

    avg_train_loss = total_train_loss / len(train_dataset)
    train_losses.append(avg_train_loss)
    if (epoch) % save_frequency == 0:
        model_save_path = os.path.join(save_dir, f"model_epoch_{epoch}_kl.pth")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': hangman_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
        }
        torch.save(checkpoint, model_save_path)
        print(f"Model saved to {model_save_path}")
