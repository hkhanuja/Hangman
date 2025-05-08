import torch
import random
import torch.nn as nn
import math
import numpy as np
import string
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
from transformers import CanineTokenizer, CanineModel, AdamW, get_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import os

canine_tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
canine_model = CanineModel.from_pretrained("google/canine-s")

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


train_words = []
with open('./train_words.txt','r') as file:
    for line in file:
        train_words.append(line.strip())
print('Training with {} words'.format(len(train_words)))

MAX_NUM_INPUTS = 29
print('Max word length: {}'.format(MAX_NUM_INPUTS))

val_words = []
with open('./val_words.txt','r') as file:
    for line in file:
        val_words.append(line.strip())

all_characters = ''.join(train_words)
char_freq = Counter(all_characters)
total_chars = sum(char_freq.values())
char_freq_normalized = {char: count / total_chars for char, count in char_freq.items()}
char_freq_inverse = {char: 1 / count for char, count in char_freq_normalized.items()}
char_freq_inverse_tensor = torch.tensor([torch.tensor(char_freq_inverse.get(chr(i + ord('a')), 1.0)) for i in range(26)])

min_val = char_freq_inverse_tensor.min()
max_val = char_freq_inverse_tensor.max()

char_freq_inverse_tensor = (char_freq_inverse_tensor - min_val) / (max_val - min_val)

batch_size = 64
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
            str(word),
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
            input_ids[i+1] = self.tokenizer.mask_token_id
          else:
            guessed_letters[ord(word[i]) - ord('a')] = 1


        random_guesses = random.randint(0, 4)
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


train_dataset = HangmanTrainDataset(train_words, canine_tokenizer, max_len)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_batch_size = batch_size
val_dataset = HangmanTrainDataset(val_words, canine_tokenizer, max_len)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

for batch in val_dataloader:
    print(batch['input_ids'].shape)
    print(batch['labels'].shape)
    print(batch['guessed_letters'].shape)
    break

class CanineHangman(nn.Module):
  def __init__(self, base_model, vocab_size = 26, hidden_size = 328):
    super(CanineHangman,self).__init__()
    self.model = base_model

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

    mask_token_indices = (input_ids == canine_tokenizer.mask_token_id).unsqueeze(-1)

    masked_logits = logits.masked_fill(~mask_token_indices, float('-inf'))

    max_masked_logits = masked_logits.max(dim=1).values 

    return max_masked_logits
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hangman_model = CanineHangman(canine_model, vocab_size = 26).to(device)

print("Trainable parameters:")
for name, param in hangman_model.named_parameters():
    if param.requires_grad:
        print(name)

lr = 3e-5 
optimizer = AdamW(hangman_model.parameters(), lr=lr, weight_decay=0.01)

loss_ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

class FrequencyWeightedFocalLoss(nn.Module):
    def __init__(self, gamma=1.5, frequency_weights=None):
        super(FrequencyWeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = frequency_weights

    def forward(self, logits, targets):

        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        pt = log_pt.exp()

        focal_weight = (1 - pt) ** self.gamma
    
        alpha_factor = self.alpha.to(logits.device).gather(dim=0, index=targets)
        focal_loss = alpha_factor * focal_weight * (-log_pt)

        return focal_loss.mean()

char_freq_inverse_tensor = char_freq_inverse_tensor.to(device)
loss_fct = FrequencyWeightedFocalLoss(frequency_weights = char_freq_inverse_tensor)
num_epochs = 70
num_training_steps = int((num_epochs * len(train_dataset)) // batch_size)
warmup_steps = int(num_training_steps * 0.1)

lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps,
)

train_losses = []
save_frequency = 7
prev_val_accuracy = -1
val_losses = []
val_accuracies = []
save_dir = 'canine/canine_focal'
for epoch in range(num_epochs):
    hangman_model.train()
    total_train_loss = 0
    epsilon = 1e-8
    
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = hangman_model( batch['input_ids'], batch['attention_mask'], batch['guessed_letters'] )

        labels = torch.multinomial(batch['labels'], 1).squeeze()

        logits = outputs.view(-1, 26)  
        loss = loss_fct(logits, labels)
        
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_train_loss = total_train_loss + (loss.item()*batch_size)

    avg_train_loss = total_train_loss / len(train_dataset)
    train_losses.append(avg_train_loss)
    
    hangman_model.eval()
    total_val_loss = 0
    val_correct = 0

    for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = hangman_model( batch['input_ids'], batch['attention_mask'], batch['guessed_letters'] )

        prediction = torch.argmax(outputs, dim=-1)
        logits = outputs.view(-1, 26)  
        loss = loss_fct(logits, prediction)
        
        total_val_loss = total_val_loss + (loss.item()*val_batch_size)

        labels = batch['labels'].clone().squeeze(0)
        correct = labels.gather(dim=-1, index=prediction.unsqueeze(-1)).squeeze(-1) > 0
        val_correct = val_correct + correct.sum()        

    avg_val_loss = total_val_loss / len(val_dataset)
    val_accuracy = val_correct / len(val_dataset)
    val_accuracies.append(val_accuracy)
    val_losses.append(avg_val_loss)

    if (epoch+1) % save_frequency == 0 or val_accuracy>prev_val_accuracy:
        model_save_path = os.path.join(save_dir, f"canine_focal_model_epoch_{epoch+1}.pth")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': hangman_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracy': val_accuracies
        }
        torch.save(checkpoint, model_save_path)
        prev_val_accuracy = val_accuracy
        print(f"Model saved to {model_save_path}")