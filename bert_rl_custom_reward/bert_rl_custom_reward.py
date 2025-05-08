import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os
from collections import Counter
from collections import deque
from transformers import BertTokenizer, BertForMaskedLM

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
with open('val_words.txt','r') as file:
    for line in file:
        val_words.append(line.strip())
max_length = MAX_NUM_INPUTS+2
save_dir = 'bert_rl_custom_reward'

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

class HangmanEnv:
    def __init__(self, word, hidden_word, guessed_letters_set):
        self.word = word
        self.hidden_word = hidden_word
        self.guessed_letters = guessed_letters_set
        self.remaining_attempts = 6
        self.done = False
    
    def step(self, letter):
        if letter in self.guessed_letters:
            return self.hidden_word, -10, self.done  # Penalize repeated guesses
        
        self.guessed_letters.add(letter)
        if letter in self.word:
            for i, ch in enumerate(self.word):
                if ch == letter:
                    self.hidden_word[i+1] = bert_tokenizer(letter)['input_ids'][1]
            reward = round(char_freq_log_tensor[ord(letter) - ord('a')].item(),2)  # Correct guess
        else:
            self.remaining_attempts -= 1
            reward = -2  # Incorrect guess
        
        if bert_tokenizer.mask_token_id not in self.hidden_word or self.remaining_attempts == 0:
            self.done = True
        
        return self.hidden_word, reward, self.done
    
    def reset(self, word):
        self.__init__(word)
        return self.hidden_word
    

class HangmanPolicy(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        vocab_size = 26
        hidden_size = 328
        self.bert = base_model.bert
        for param in self.bert.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[10:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(768+vocab_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, attention_mask, guessed_letters):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        sequence_output = torch.cat((outputs, guessed_letters.unsqueeze(1).repeat(1, outputs.shape[1], 1)), dim=2)
        logits = self.fc(sequence_output)
        logits = self.relu(logits)
        logits = self.output(logits)
        logits[guessed_letters.unsqueeze(1).repeat(1, sequence_output.shape[1], 1) == 1] = -float("inf")

        mask_token_indices = (input_ids == bert_tokenizer.mask_token_id).unsqueeze(-1)

        masked_logits = logits.masked_fill(~mask_token_indices, float('-inf'))

        max_masked_logits = masked_logits.max(dim=1).values 

        return F.softmax(max_masked_logits, dim=-1).squeeze(0)
    

def train_policy_gradient(train_words, model, optimizer, patience=5000):
    reward_history = deque(maxlen=100)
    best_avg_reward = -float("inf")
    no_improvement_count = 0
    random.shuffle(train_words)

    for episode in range(500000):
        word = random.choice(train_words)
        if episode<len(train_words):
            word = train_words[episode]            
        guessed_letters_set = set()
        word_list = list(word)
        mask_ratio = random.randint(3, 4) / 10
        num_masks = int(len(word_list) * mask_ratio)
        if num_masks == 0:
          num_masks=1
        mask_indices = random.sample(range(len(word)), num_masks)
        letters_to_be_masked = set([word_list[i] for i in mask_indices])
        tokenized_word = bert_tokenizer(
            " ".join(list(word)),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length = max_length
            )

        input_ids = tokenized_word.input_ids.clone().squeeze(0)
        attention_mask = tokenized_word.attention_mask.squeeze(0)
        guessed_letters = torch.zeros(26, dtype=torch.float32).to(device)

        for i in range(len(word_list)):
            if word[i] in letters_to_be_masked:
                input_ids[i+1] = bert_tokenizer.mask_token_id
            else:
                guessed_letters[ord(word[i]) - ord('a')] = 1
                guessed_letters_set.add(word[i])

        env = HangmanEnv(word, input_ids.clone(), guessed_letters_set)
        state = env.hidden_word
        log_probs = []
        rewards = []
        
        while not env.done:

            probs = model(state.unsqueeze(0).to(device), attention_mask.unsqueeze(0).to(device), guessed_letters.unsqueeze(0).to(device))
            distribution = torch.distributions.Categorical(probs)
            action = distribution.sample()
            letter = chr(action.item() + ord('a'))
            state, reward, done = env.step(letter)
            
            guessed_letters[action.item()] = 1
            log_probs.append(distribution.log_prob(action))
            rewards.append(reward)
        
        # Compute policy loss
        discounted_rewards = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.tensor(discounted_rewards).to(device)
        policy_loss = -torch.stack(log_probs) * discounted_rewards
        policy_loss = policy_loss.mean()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        reward_history.append(sum(rewards))
        avg_reward = np.mean(reward_history)
        
        # if avg_reward > best_avg_reward:
        #     best_avg_reward = avg_reward
        #     no_improvement_count = 0
        # else:
        #     no_improvement_count += 1
        
        # if no_improvement_count >= patience:
        #     print(f"Early stopping at episode {episode} with avg reward {avg_reward}")
        #     break
        
        if episode % 1000 == 0:
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
        
        if episode % 10000 == 0:
            model_save_path = os.path.join(save_dir, f"bert_rl_custom_reward_model_epoch_{episode}.pth")
            checkpoint = {
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),

            }
            torch.save(checkpoint, model_save_path)
            print(f"Model saved to {model_save_path}")


hangman_policy = HangmanPolicy(bert_model).to(device)
optimizer = optim.Adam(hangman_policy.parameters(), lr=3e-5)
train_policy_gradient(train_words, hangman_policy, optimizer)
