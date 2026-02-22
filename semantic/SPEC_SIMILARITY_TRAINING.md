# Technical Specification: Pairwise Cosine Similarity Training for Language Models

## Purpose

This system trains language models using pairwise cosine similarity loss in embedding space rather than cross-entropy next-token loss. The embedding matrix is frozen as a static random codebook — the transformer learns to map input contexts to fixed points in this space, preserving the geometric relationships between token embeddings.

The core thesis: training on geometric relationship preservation in embedding space — rather than discrete token identity — yields models with stronger analogical reasoning and compositional generalization. Cross-entropy treats the vocabulary as an unstructured set where "almost right" and "completely wrong" are penalized identically. Cosine similarity operates on the same manifold where semantic arithmetic works (king - man + woman = queen), directly rewarding the model for maintaining geometric relationships between meanings.

The loss is not "how close is the model's output to the right answer." It is "does the model's output preserve the same geometric relationships as the known-good embeddings." If the embedding matrix says tokens A and B have cosine similarity 0.78, and the model produces hidden states A' and B' with cosine similarity 0.52, the loss is the squared gap (0.78 - 0.52)². The model learns to satisfy distance constraints — like reconstructing a triangle from its side lengths.

## Key Design Decisions

- **Frozen embedding matrix**: The token embedding matrix is initialized randomly and frozen. It serves as a static codebook — a fixed coordinate system that the transformer must learn to target. This eliminates embedding collapse risk and provides a stable, non-moving target for training.
- **Geometric relationship preservation, not absolute loss**: The system compares the cosine similarity between pairs of model outputs against the cosine similarity between the corresponding frozen token embeddings. The loss is the squared discrepancy between these two similarities.
- **Two pair types per batch**: Each batch includes geometric pairs (hidden state vs. hidden state) and anchor pairs (hidden state vs. raw embedding). Geometric pairs teach relationships. Anchor pairs keep hidden states aligned to the embedding space for inference.
- **Only the last hidden vector**: The training loop uses only the final hidden state at the last target position. Intermediate hidden states are not compared — we don't care how the model gets there, only where it lands.
- **Cosine similarity as distance metric**: The loss operates in the same space where semantic arithmetic works. The model is directly optimized for semantic fidelity, not token identity.
- **No separate reward model**: Quality signal comes directly from cosine similarity to known-good text, eliminating the reward model training stage required by RLHF.

## Functional Requirements

### Loss Function

The loss measures how well the model's output vectors preserve the geometric relationships defined by the frozen embedding matrix. For any two vectors X and Y, the pairwise loss is:

    loss(X, Y) = (sim(X', Y') - sim(X, Y))²

where X and Y are the known-good token embeddings from the frozen codebook and X' and Y' are the corresponding model outputs or embeddings being compared.

Each mini-batch MUST include two types of pairs:

#### Geometric Pairs (ratio: 0.7)

- The system MUST compute `(sim(A', B') - sim(A, B))²` where A' and B' are the model's last hidden states for two observations and A and B are the corresponding frozen token embeddings
- These pairs teach the model to preserve inter-token geometric relationships in its output space

#### Anchor Pairs (ratio: 0.3)

- The system MUST compute `(sim(A', E_k) - sim(A, E_k))²` where A' is the model's last hidden state, A is the frozen token embedding, and E_k is a randomly sampled token embedding from the frozen vocabulary matrix
- These pairs anchor the model's hidden states to the embedding space, preventing the output geometry from rotating away from the embedding matrix
- Without anchor pairs, the model could learn a valid but rotated geometry that makes nearest-neighbor token selection at inference time impossible

#### General Requirements

- The system MUST NOT apply any activation function (sigmoid, softmax) to the model's hidden state before computing cosine similarity — the comparison operates on raw embedding geometry
- The system SHOULD reuse forward pass activations across pairs within a mini-batch rather than recomputing — each observation's hidden state is independent of its pair partner, so N observations require N forward passes regardless of pairing count
- Embedding pairs are unnecessary because the embedding matrix is frozen — collapse is impossible when embeddings cannot move

### Training

- The system MUST train on (prompt, single next token) pairs
- The system MUST use only the last hidden vector (the final position of the target token after running through the transformer) for loss computation
- The system MUST NOT aggregate intermediate hidden states — only the final vector matters

### Inference-Time Token Selection

- At inference time, the system MUST compute cosine similarity between the model's output hidden state and every token embedding in the frozen vocabulary matrix to produce a score distribution over tokens
- This nearest-neighbor selection works because anchor pairs during training keep the model's hidden states aligned to the embedding space
- The system MUST support sampling (softmax over similarity scores with temperature) for token selection
- The system SHOULD apply a temperature parameter to the similarity scores before softmax to control generation diversity
- The system MUST use `torch.inference_mode()` during generation

### Frozen Embedding Rationale

In standard transformers, the embedding matrix is learned during training. Similar tokens like "cat" and "dog" move closer together, so the attention/feed-forward mechanism doesn't have to do that work — the structure is "priced in" to the embeddings.

With frozen random embeddings, the transformer does all the work. There is no pre-existing semantic structure in the codebook. The model must learn to map contexts to the right points in a random coordinate system. This is harder but has advantages:

- **Stable target**: The loss landscape doesn't shift as embeddings move
- **No collapse risk**: Frozen embeddings cannot converge toward each other
- **Simpler training**: No need for embedding pairs, stop-gradient tricks, or collapse monitoring
- **Clean separation**: The codebook defines the coordinate system; the transformer defines the mapping

## Relationship to DPO

Direct Preference Optimization (DPO, Rafailov et al. 2023) is the closest existing technique. Both use pairwise comparison of responses. The key differences:

| Aspect | DPO | This System |
|--------|-----|-------------|
| Comparison metric | Log-probability ratios | Cosine similarity in embedding space |
| Base model dependency | Requires cross-entropy-pretrained base | Trains from scratch |
| Quality signal source | Human preference labels | Geometric fidelity to frozen codebook |
| Embedding matrix | Learned (co-evolves with model) | Frozen (static random codebook) |
| What it optimizes | Token probability distributions | Preservation of semantic geometry |
| Pair semantics | Good vs. bad response | Two valid observations, geometric comparison |

### The DPO Objection

DPO works because it stands on a cross-entropy-trained base model that already understands language. The log-probability comparison is meaningful because the base model has learned a calibrated distribution over tokens. This system removes that foundation — without cross-entropy pretraining, the embedding space may lack the geometric structure that makes cosine similarity meaningful. The risk: cosine similarity training never converges to the point where its geometric advantages matter, because the geometry itself is malformed without the bootstrap phase.

**Counter-argument**: With frozen random embeddings, the system does not rely on pre-existing embedding structure. The geometric structure emerges entirely from the transformer learning to map contexts to the right codebook entries. The two pair types (geometric, anchor) provide sufficient gradient signal. The structure may take longer to emerge but is not dependent on cross-entropy or learned embeddings.

This is an empirical question. The validation experiment tests it directly.

## Additional Counter-Arguments and Mitigations

### Discrete Token Validity

Language is discrete. Embeddings close in cosine distance may correspond to invalid tokens. Cross-entropy's harshness forces commitment to valid tokens — cosine similarity's softness may allow the model to settle in "meaning-adjacent" regions that produce invalid output.

- Anchor pairs partially mitigate this: by forcing hidden states to maintain correct similarities to raw token embeddings, the model learns to produce vectors that resolve unambiguously to specific tokens

### Loss of Calibrated Confidence

Pairwise similarity discards absolute confidence. Cross-entropy produces calibrated probability distributions that drive hedging and uncertainty expression.

- The system MAY apply post-hoc calibration to map embedding distances to calibrated confidence scores

## Validation Plan: GPT-2 on Shakespeare

### Rationale

- GPT-2 is small enough to train from scratch in reasonable time
- Shakespeare provides a constrained but rich vocabulary with clear stylistic patterns
- Analogical structure is testable: character relationships, thematic parallels, meter patterns
- Standard cross-entropy GPT-2 on Shakespeare is a well-characterized baseline

### Experiment Design

- **Baseline**: Train GPT-2 on Shakespeare with standard cross-entropy loss
- **Cosine-only**: Train GPT-2 on Shakespeare with pairwise cosine similarity loss and frozen embeddings

### Evaluation

- The experiment MUST measure generation quality: coherence, grammatical validity, and stylistic fidelity to Shakespeare
- The experiment SHOULD probe the embedding space for semantic arithmetic (character relationships, thematic vectors) to test the analogical reasoning hypothesis
- The experiment SHOULD measure convergence speed (training steps to comparable generation quality)
- The experiment SHOULD evaluate generation coherence on out-of-distribution prompts (modern English, non-Shakespearean literary styles) to test whether cosine-trained models degrade more gracefully than cross-entropy baselines

### Expected Outcomes

- The cosine-trained model converges slower initially
- The cosine-trained model produces generation that preserves semantic relationships
- Generation quality at response level is competitive with or exceeds cross-entropy baseline
- The frozen embedding approach eliminates collapse failure modes

## Open Questions

- What is the optimal batch size for pairwise comparison? Sampling strategies need exploration for large batches.
- Does the approach benefit from a temperature parameter on cosine similarity (analogous to the beta parameter in DPO)?
- What is the optimal ratio of geometric pairs to anchor pairs per batch?
- Does the system need a repulsive loss term (negative examples or contrastive pairs) to establish global embedding structure, or do the two pair types provide sufficient implicit contrast?
- How many anchor pairs per observation are needed to prevent rotation drift? Is one random embedding sufficient, or does the system need k anchors per hidden state?
- How does the frozen random codebook compare to a frozen pre-trained codebook (e.g., frozen GPT-2 embeddings)?

## Error Handling

- **Degenerate embeddings**: If cosine similarity produces NaN (zero-norm vectors), the affected observation MUST be excluded from the batch rather than producing undefined gradients
- **Pair imbalance**: If a mini-batch contains observations with very similar cosine similarity gaps, the pairwise signal becomes weak. The system SHOULD log batches where the maximum pairwise difference falls below a configurable threshold
