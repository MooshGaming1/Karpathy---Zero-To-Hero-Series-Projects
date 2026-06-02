import os
import torch
import torch.nn.functional as F

# Open the dataset of names and store each name as a separate string in a list.
# .splitlines() splits the giant string by newline characters so each name is its own entry.
# __file__ is the absolute path of this script.
# os.path.dirname(__file__) gives the folder it lives in.
# This means names.txt is always found next to trigram.py,
# regardless of which directory you run the script from.
names_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'names.txt')
words = open(names_path, 'r').read().splitlines()


# Get the unique characters in the dataset.
# We join all words into one long string, turn it into a set (which removes duplicates),
# then sort alphabetically so every run of the program uses the same consistent ordering.
chars = sorted(list(set(''.join(words))))

# Map each character to a unique integer index.
# We start at 1 (i+1) so that index 0 is reserved for the special '.' token (start/end of word).
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0  # '.' acts as both the start-of-word and end-of-word marker

# Store the reverse map: integer index -> character.
# This lets us convert predicted indices back into readable characters during sampling.
itos = {i:s for s,i in stoi.items()}

#BUILD TRIGRAM TRAINING DATA 
# A trigram is a sliding window of 3 characters.
# Given the first two characters (context), the model must predict the third.

xs, ys = [], []  # xs = inputs (context pairs), ys = labels (next character to predict)

# GOAL: Go through each word and generate all its trigrams.
# We prepend TWO '.' tokens so the model sees (., .) before the very first character,
# which lets it learn what characters typically start a name.
for w in words:
    # The two '.'s allow a sliding 3-character window to isolate the first character cleanly.
    # e.g. "ab" becomes ['.', '.', 'a', 'b', '.']
    chs = ['.'] + ['.'] + list(w) + ['.']

    # zip(chs, chs[1:], chs[2:]) slides a window of size 3 across the character list.
    # Each iteration gives us three consecutive characters: ch1, ch2, ch3.
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]

        # We encode the two-character context as a single integer.
        # Since there are 27 possible characters (26 letters + '.'), we treat (ix1, ix2)
        # like a 2-digit base-27 number: ix1*27 + ix2.
        # This maps each unique pair to one of 729 (27*27) possible indices.
        xs.append(ix1 * 27 + ix2)

        # The label (what we want to predict) is simply the third character's index.
        ys.append(ix3)

xs = torch.tensor(xs)  # Shape: (N,)  — N training examples, each is a single integer 0-728
ys = torch.tensor(ys)  # Shape: (N,)  — N labels, each is an integer 0-26
num = xs.nelement()
print('number of examples: ', num)

#MODEL INITIALIZATION 

# Set a manual seed for reproducibility — every time you run the script you get
# the same random numbers, making experiments comparable.
g = torch.Generator().manual_seed(2147483647)

# W is the weight matrix
# Shape: (729 rows, 27 columns)
#   - 729 rows: one row per unique two-character context (27 * 27 = 729 combinations)
#   - 27 columns: one column per possible next character
# requires_grad=True tells PyTorch to track all operations on W so it can
# compute gradients automatically during the backward pass.
W = torch.randn((729, 27), requires_grad=True)

# ONE-HOT ENCODING: Convert each integer context index into a 729-dimensional vector
# where every element is 0 except for a single 1 at position xs[i].
# This is needed for the one-hot × W matrix multiplication style of forward pass.
# However, note that in the training loop below we use W[xs[ix]] (direct row lookup)
# which is mathematically equivalent but much faster — xenc is kept here for clarity.
xenc = F.one_hot(xs, num_classes=729).float()  # Shape: (N, 729)



# BATCH SIZE = 64:
# Instead of computing the loss on all 228k examples at once (which is slow and
# uses a lot of memory), we randomly pick 64 examples each iteration.
batch_size = 64

# LEARNING RATE = 1.0:
# The learning rate controls how big a step we take each update: W -= lr * W.grad
learning_rate = 1.0

#TRAINING LOOP
# We run 100,000 iterations. Each iteration processes one mini-batch of 64 examples.
# Total data seen = 100,000 × 64 = 6.4 million character predictions.
for k in range(100000):

    num_examples = xs.shape[0]  # Total number of training examples

    # MINI-BATCH SAMPLING
    # Instead of using the whole dataset every step, we randomly sample 64 indices.
    # This is called Stochastic Gradient Descent (SGD) with mini-batches.
    

    # torch.randint picks 'batch_size' random integers in the range [0, num_examples).
    ix = torch.randint(0, xs.shape[0], (batch_size,), generator=g)

    #  FORWARD PASS 
    # xs[ix] selects the 64 context indices for this mini-batch (values 0-728).
    # W[xs[ix]] looks up the corresponding row of W for each context.
    # Result shape: (64, 27) -> 64 examples, each with 27 raw scores (one per character).
    logits = W[xs[ix]]

    
    # COUNTS: We exponentiate the logits to make them all positive.
    # exp() maps (-infinity, infinity) to (0, infinity), so every count is strictly positive.
    counts = logits.exp()

    # PROBS: Normalise the counts so they sum to 1 across the 27 characters (dim=1).
    # Now each row is a proper probability distribution: all values in [0,1] and sum = 1.
    # This is the full Softmax: probs = exp(logits) / sum(exp(logits))
    # keepdims=True keeps the shape (64,1) so broadcasting divides each row correctly.
    probs = counts / counts.sum(1, keepdims=True)

    # LOSS: NEGATIVE LOG-LIKELIHOOD + L2 REGULARIZATION 
    # probs[torch.arange(batch_size), ys[ix]] picks out, for each of the 64 examples,
    # the probability the model assigned to the *correct* next character.
    # A perfect model would assign probability 1.0, giving log(1) = 0 loss.
    # A bad model assigns low probability, giving a large negative log → high loss.
    # We negate and average over the batch to get a single scalar loss value.
    #
    # L2 REGULARIZATION (0.01 * (W**2).mean()):
    # This adds a small penalty for large weights, which discourages the model from
    # becoming overconfident. It's a common trick to improve generalisation.
    # The coefficient 0.01 controls how strong the penalty is — small enough not to
    # overwhelm the main loss, large enough to keep weights from exploding.
    loss = -probs[torch.arange(batch_size), ys[ix]].log().mean() + 0.01 * (W**2).mean()

    # BACKWARD PASS 
    # Reset gradients to zero before backprop.
    # PyTorch *accumulates* gradients by default, so without this step the gradients
    # from the previous iteration would add on top of the new ones — causing wrong updates.
    W.grad = None

    # Compute the gradient of the loss with respect to every element in W.
    # PyTorch traces the math from the loss all the way back to W automatically.
    loss.backward()

    # PARAMETER UPDATE (Gradient Descent) 
    # Move W in the direction that reduces the loss.
    # W.grad tells us the slope — subtracting it moves us downhill.
    # We use .data so PyTorch doesn't try to track this update operation itself.
    W.data += -learning_rate * W.grad

    # Print progress every 20,000 iterations so we can watch the loss decrease.
    if k % 20000 == 0 or k == 99999:
        print(f"Iteration {k:6d} | Batch Loss: {loss.item():.4f}")

# SAMPLING LOOP 
# After training, we use the learned W to generate brand-new name-like strings.
# The process mimics how the model was trained: given two characters, predict the next,
# then slide the window forward and repeat until we hit the end-of-word token '.'.

print("\n--- Sampling from Trigram Model ---")
for i in range(15):  # Generate 15 sample names
    out = []

    # Start with the same context we used at the beginning of training: ('.', '.')
    # ix1 and ix2 both equal 0 because stoi['.'] = 0.
    ix1, ix2 = 0, 0

    while True:
        # Compute the single integer that encodes the current two-character context.
        # This is the same formula we used when building xs: ix1*27 + ix2.
        # e.g. context ('.', '.') → 0*27+0 = 0, context ('.', 'a') → 0*27+1 = 1
        ch_ix = ix1 * 27 + ix2

        # Look up the row of W for this context gives us 27 raw logit scores.
        logits = W[ch_ix]

        # Convert logits -> positive counts -> probability distribution (same as training).
        counts = logits.exp()
        p = counts / counts.sum()  # Shape: (27,) — one probability per character

        # MULTINOMIAL SAMPLING: Instead of always picking the most likely character
        # (which would be deterministic and repetitive), we sample from the distribution.
        # Characters with higher probability are more likely to be picked, but lower
        # probability characters can still be chosen, this creates variety in the output.
        # torch.multinomial returns the index of the sampled character.
        ix3 = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

        # If we sampled the '.' token (index 0), the name is finished -> stop the loop.
        if ix3 == 0:
            break

        # Convert the predicted index back to a character and add it to our output.
        out.append(itos[ix3])

        # SLIDE THE WINDOW FORWARD:
        # To predict the next character, we need to update our 2-character history.
        # We discard the oldest character (ix1), move the second character (ix2) to the front,
        # and append the character we just predicted (ix3) as the new history.
        # e.g., if history was ('a', 'b') and we predicted 'c', the new history becomes ('b', 'c').
        ix1 = ix2
        ix2 = ix3

    print(''.join(out))