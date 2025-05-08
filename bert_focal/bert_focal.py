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

char_freq_inverse_tensor = torch.tensor(
    [char_freq_inverse.get(chr(i + ord('a')), 1.0) for i in range(26)]
)

# Convert to logarithmic scale with epsilon for stability
epsilon = 1e-9
char_freq_log_tensor = torch.log(char_freq_inverse_tensor + epsilon)
char_freq_log_tensor = char_freq_log_tensor/min(char_freq_log_tensor)


batch_size = 64
max_len = MAX_NUM_INPUTS+2

class HangmanTrainDataset(Dataset):
    def __init__(self, words, tokenizer, max_length):
        self.words = words
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.all_letters = set(string.ascii_lowercase)
        self.char_to_index = {ch: i for i, ch in enumerate(string.ascii_lowercase)}

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        letters_in_word = set(word)

        word_list = list(word)
        mask_ratio = random.randint(3, 8) / 10
        num_masks = int(len(letters_in_word) * mask_ratio)
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
            input_ids[i+1] = self.tokenizer.mask_token_id
          else:
            guessed_letters[ord(word[i]) - ord('a')] = 1


        random_guesses = random.randint(0, 4)
        letters_not_in_word = list(self.all_letters - letters_in_word)
        random_letters_guessed = random.sample(letters_not_in_word, random_guesses)

        for i in range(len(random_letters_guessed)):
          guessed_letters[ord(random_letters_guessed[i]) - ord('a')] = 1

        letter = random.sample(letters_to_be_masked, 1)[0]
        label = torch.tensor(self.char_to_index[letter], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
            "guessed_letters": guessed_letters
        }


train_dataset = HangmanTrainDataset(train_words, bert_tokenizer, max_len)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_batch_size = batch_size
val_dataset = HangmanTrainDataset(val_words, bert_tokenizer, max_len)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

for batch in val_dataloader:
    print(batch['input_ids'].shape)
    print(batch['labels'].shape)
    print(batch['guessed_letters'].shape)
    break

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

print("Trainable parameters:")
for name, param in hangman_model.named_parameters():
    if param.requires_grad:
        print(name)

lr = 3e-5 
optimizer = AdamW(hangman_model.parameters(), lr=lr, weight_decay=0.01)

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

char_freq_log_tensor = char_freq_log_tensor.to(device)
loss_fct = FrequencyWeightedFocalLoss(frequency_weights = char_freq_log_tensor)

num_epochs = 100
num_training_steps = int((num_epochs * len(train_dataset)) // batch_size)
warmup_steps = int(num_training_steps * 0.1)

lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps,
)

train_losses = []
save_frequency = 10
prev_val_loss = 1e9
val_losses = []
val_accuracies = []
save_dir = 'bert_focal'
for epoch in range(num_epochs):
    hangman_model.train()
    total_train_loss = 0
    
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = hangman_model( batch['input_ids'], batch['attention_mask'], batch['guessed_letters'] )

        logits = outputs.view(-1, 26)
        labels = batch['labels'].view(-1)
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

        labels = batch['labels'].view(-1) 
        logits = outputs.view(-1, 26)  
        loss = loss_fct(logits, labels)

        prediction = torch.argmax(logits, dim=-1)

        val_correct = val_correct + (prediction == labels).sum().item()
        
        total_val_loss = total_val_loss + (loss.item()*val_batch_size)


    avg_val_loss = total_val_loss / len(val_dataset)
    val_accuracy = val_correct / len(val_dataset)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    if (epoch+1) % save_frequency == 0 or avg_val_loss<prev_val_loss:
        model_save_path = os.path.join(save_dir, f"bert_focal_model_epoch_{epoch+1}.pth")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': hangman_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracy': val_accuracies,
        }
        torch.save(checkpoint, model_save_path)
        prev_val_loss = avg_val_loss
        print(f"Model saved to {model_save_path}")