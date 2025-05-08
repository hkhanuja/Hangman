# Hangman

## What is Hangman?

**Hangman** is a classic word-guessing game where the player attempts to uncover a hidden word by suggesting letters one at a time.  
Each incorrect guess results in a part of a stick figure being drawn on a gallows.  
The game concludes when the player either successfully guesses the word or the stick figure is fully drawn, indicating too many incorrect guesses.


```
 _______
 |     |
 |     O
 |    /|\
 |    / \
 |
_|_
```

*Example Hangman figure after several incorrect guesses.*

## Approaches

To solve the Hangman problem, I approached the problem with a few approaches, the 
classical n-gram matching approach, neural networks, LSTM, Bi-LSTM, and transformers. In 
order to test the efficacy of all the approaches I divided my dataset (words_250000.txt) into 2 
parts (90% training, 10% validation). These approaches could not achieve the desired 
performance level, however, while using transformers I realised that the Hangman game 
seems like the perfect use-case for BERT, since BERT was also trained to predict masked 
words in a sentence. 

I tried several approaches with the BERT and Canine-S architecture. Canine-S model is 
similar to BERT but instead of word embeddings it was trained on character level 
embeddings. I did 16-20 experiments with different losses, architectures and training setups: 
● Cross Entropy loss over all masked characters 
● Cross Entropy loss over the mean of the output logits of all masked characters (mean 
pooling) 
● Cross Entropy loss over the maximum of the output logits of all masked characters 
(max pooling) 
● KL Divergence loss 
● Focal Loss - this basically assigns a higher loss to infrequent characters such as ‘q’, 
‘x’ and lower loss to frequent characters like ‘a’ and ‘e’. The frequency was calculated 
using the training set. 
● **Cross Entropy loss combined with n-gram prediction when the word is almost 
predicted (3 characters or less remaining to be predicted) **
● Reinforcement Learning with BERT model as the policy 
● Reinforcement Learning with BERT model as the policy and custom rewards 
depending on the character frequency (higher reward for infrequent characters) 

The highlighted approach gives the best results.
