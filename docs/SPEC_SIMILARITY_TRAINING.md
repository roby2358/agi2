# Technical Specification: Pairwise Cosine Similarity Training for Language Models

## Purpose

This system trains language models using pairwise cosine similarity loss in embedding space rather than cross-entropy next-token loss. It draws from pairwise similarity loss used in financial prediction (see SPEC_TRAINING.md) and extends it to language generation through a curriculum that progresses from single-token to full-response training.

The core thesis: training on geometric relationship preservation in embedding space — rather than discrete token identity — yields models with stronger analogical reasoning and compositional generalization. Cross-entropy treats the vocabulary as an unstructured set where "almost right" and "completely wrong" are penalized identically. Cosine similarity operates on the same manifold where semantic arithmetic works (king - man + woman = queen), directly rewarding the model for maintaining geometric relationships between meanings.

The loss is not "how close is the model's output to the right answer." It is "does the model's output preserve the same geometric relationships as the known-good embeddings." If the embedding matrix says tokens A and B have cosine similarity 0.78, and the model produces hidden states A' and B' with cosine similarity 0.52, the loss is the gap |0.78 - 0.52|. The model learns to satisfy distance constraints — like reconstructing a triangle from its side lengths.

## Key Design Decisions

- **Geometric relationship preservation, not absolute loss**: The system compares the cosine similarity between pairs of model outputs against the cosine similarity between the corresponding known-good token embeddings. The loss is the discrepancy between these two similarities. This is a geometric constraint satisfaction problem — the model learns to produce hidden state vectors that preserve the distance relationships defined by the embedding matrix.
- **Three pair types per batch**: Each batch includes geometric pairs (hidden state vs. hidden state), anchor pairs (hidden state vs. raw embedding), and embedding pairs (raw embedding vs. raw embedding). Geometric pairs teach relationships. Anchor pairs keep hidden states aligned to the embedding space for inference. Embedding pairs prevent embedding matrix collapse.
- **Cosine similarity as distance metric**: The loss operates in the same space where semantic arithmetic works. The model is directly optimized for semantic fidelity, not token identity.
- **Curriculum progression**: Training begins with single-token comparison and gradually extends to multi-token and full-response comparison. The loss function remains unchanged across stages — only the observation window widens.
- **No separate reward model**: Quality signal comes directly from cosine similarity to known-good text, eliminating the reward model training stage required by RLHF.
- **Single-phase training**: The curriculum eliminates the two-phase pretrain-then-align pipeline. Token grounding, sequence coherence, and stylistic fidelity emerge from the same loss function applied at increasing observation scale. The corpus provides both knowledge and preference signal — there is no separate alignment stage.

## Functional Requirements

### Loss Function

The loss measures how well the model's output vectors preserve the geometric relationships defined by the embedding matrix. For any two vectors X and Y (which may be model hidden states or raw token embeddings), the pairwise loss is:

    loss(X, Y) = |sim(X', Y') - sim(X, Y)|

where X and Y are the known-good token embeddings and X' and Y' are the corresponding model outputs or embeddings being compared.

Each mini-batch MUST include three types of pairs:

#### Geometric Pairs

- The system MUST compute |sim(A', B') - sim(A, B)| where A' and B' are the model's output hidden states for two observations and A and B are the corresponding known-good token embeddings
- These pairs teach the model to preserve inter-token geometric relationships in its output space

#### Anchor Pairs

- The system MUST compute |sim(A', E_j) - sim(A, E_j)| where A' is the model's output hidden state, A is the known-good token embedding, and E_j is a sampled token embedding from the vocabulary matrix
- These pairs anchor the model's hidden states to the embedding space, preventing the output geometry from rotating away from the embedding matrix
- Without anchor pairs, the model could learn a valid but rotated geometry that makes nearest-neighbor token selection at inference time impossible

#### Embedding Pairs

- The system MUST include pairs of raw token embeddings in the loss to maintain embedding matrix structure during training
- These pairs prevent embedding collapse (all embeddings converging toward each other)

#### General Requirements

- The system MUST NOT apply any activation function (sigmoid, softmax) to the model's hidden state before computing cosine similarity — the comparison operates on raw embedding geometry
- The system SHOULD reuse forward pass activations across pairs within a mini-batch rather than recomputing — each observation's hidden state is independent of its pair partner, so N observations require N forward passes regardless of pairing count
- The embedding matrix and the model co-evolve during training; no stop-gradient is applied by default, but the system SHOULD monitor for collapse (see Embedding Matrix Stability)

### Curriculum Training

The system MUST train in three progressive stages. The loss function does not change between stages — only the number of tokens compared per observation.

#### Stage 1: Single Token

- Each observation MUST consist of a prompt with a single known next token
- The system MUST compute one output embedding at the final prompt position
- For multi-token observations, only the final hidden state is used — intermediate hidden states are not compared
- This stage teaches basic token prediction through geometric relationship preservation

#### Stage 2: Short Sequences

- Each observation MUST consist of a prompt with a known continuation of 2–5 tokens
- The system MUST compute output embeddings at each position in the continuation window and aggregate them into a single observation vector
- The system SHOULD apply exponentially decaying position weights (earlier positions weighted higher) when aggregating into the single observation vector
- Rationale: errors at earlier positions cascade, making later-position hidden states less meaningful
- The system SHOULD compare front-weighted vs. uniform aggregation during validation to confirm the benefit
- This stage introduces sequence-level coherence across multiple generation steps

#### Stage 3: Full Responses

- Each observation MUST consist of a complete (prompt, response) pair
- The system MUST compute output embeddings across the full response length and aggregate into a single observation vector via arithmetic mean
- This stage addresses exposure bias — the model evaluates its own multi-token trajectory against known-good responses, learning from its own compounding errors

#### Stage Transitions

- The system SHOULD advance to the next stage when pairwise accuracy on a held-out set plateaus
- Each stage MUST begin with the weights from the previous stage
- Learning rate MAY be reset or reduced at each transition

### Multi-Token Aggregation

- When comparing sequences longer than one token, the system MUST aggregate per-position hidden states into a single observation vector before computing pairwise similarities
- For stage 2, the system SHOULD use exponentially decaying position weights as the default aggregation, with earlier positions weighted higher
- For stage 3, the default aggregation MUST be arithmetic mean across all positions (uniform weighting is defensible at response scale where the goal is holistic quality)
- Alternative aggregations (minimum similarity, other weighting schemes) MAY be explored

### Observation Pairing

- The system MUST pair observations within each mini-batch for pairwise loss computation
- Both observations in a pair MUST be valid corpus continuations
- No external quality labels are required; the signal is purely geometric — how well the model's output vectors preserve the similarity relationships defined by the embedding matrix
- The system MUST sample pairs rather than computing all combinations (N observations yield N(N-1)/2 possible pairs; exhaustive pairing is prohibitive)
- Each batch MUST include a mix of geometric pairs, anchor pairs, and embedding pairs (see Loss Function)

### Inference-Time Token Selection

- At inference time, the system MUST compute cosine similarity between the model's output hidden state and every token embedding in the vocabulary matrix to produce a score distribution over tokens
- This nearest-neighbor selection works because anchor pairs during training keep the model's hidden states aligned to the embedding space
- The system MUST support greedy selection (argmax over similarities) and sampling (softmax over similarity scores with temperature) for token selection
- The system SHOULD apply a temperature parameter to the similarity scores before softmax to control generation diversity

## Relationship to DPO

Direct Preference Optimization (DPO, Rafailov et al. 2023) is the closest existing technique. Both use pairwise comparison of responses. The key differences:

| Aspect | DPO | This System |
|--------|-----|-------------|
| Comparison metric | Log-probability ratios | Cosine similarity in embedding space |
| Base model dependency | Requires cross-entropy-pretrained base | Trains from scratch or fine-tunes |
| Quality signal source | Human preference labels | Geometric fidelity to known-good embeddings |
| Training scope | Response-level only | Single token through full response (curriculum) |
| What it optimizes | Token probability distributions | Preservation of semantic geometry |
| Pair semantics | Good vs. bad response | Two valid observations, geometric comparison |

### The DPO Objection

DPO works because it stands on a cross-entropy-trained base model that already understands language. The log-probability comparison is meaningful because the base model has learned a calibrated distribution over tokens. This system removes that foundation — without cross-entropy pretraining, the embedding space may lack the geometric structure that makes cosine similarity meaningful. The risk: cosine similarity training never converges to the point where its geometric advantages matter, because the geometry itself is malformed without the bootstrap phase.

**Counter-argument**: Embedding spaces develop geometric structure through co-occurrence patterns in training data, not specifically from cross-entropy loss. The three-pair-type design (geometric, anchor, embedding) provides multiple gradient pathways for establishing structure. Embedding pairs maintain global structure. Anchor pairs ground hidden states. Geometric pairs teach relationships. The structure may take longer to emerge but is not dependent on cross-entropy.

This is an empirical question. The validation experiment tests it directly.

## Additional Counter-Arguments and Mitigations

### Embedding Space Anisotropy

Embedding spaces are high-dimensional and potentially anisotropic — not all dimensions carry meaning equally. Cosine similarity treats all dimensions as equal weight. If the space has degenerate dimensions, cosine similarity optimizes for a distorted geometry.

- The system SHOULD monitor embedding space isotropy during training
- The system MAY apply learned dimension weighting or whitening transforms if anisotropy develops

### Embedding Matrix Stability

The embedding matrix co-evolves with the model — it serves as both the comparison target (defining the geometry the model must preserve) and a trainable parameter. Without safeguards, there is a collapse mode where embeddings converge toward each other to trivially satisfy similarity constraints.

- Embedding pairs in each batch provide direct gradient signal to maintain embedding matrix structure
- The system SHOULD monitor pairwise distances within the embedding matrix during training to detect convergence/collapse
- If collapse is detected, the system SHOULD apply stop-gradient on the target embeddings (treating them as fixed for that step) or freeze the embedding matrix for a recovery period
- If using weight tying between input embeddings and output projection, the system MUST verify that cosine loss gradients do not destabilize the shared weights

### Discrete Token Validity

Language is discrete. Embeddings close in cosine distance may correspond to invalid tokens. Cross-entropy's harshness forces commitment to valid tokens — cosine similarity's softness may allow the model to settle in "meaning-adjacent" regions that produce invalid output.

- Anchor pairs partially mitigate this: by forcing hidden states to maintain correct similarities to raw token embeddings, the model learns to produce vectors that resolve unambiguously to specific tokens
- The curriculum's single-token stage provides extensive practice on token-level validity before extending to sequences

### Loss of Calibrated Confidence

Pairwise similarity discards absolute confidence. Cross-entropy produces calibrated probability distributions that drive hedging and uncertainty expression.

- The system MAY apply post-hoc calibration (analogous to the financial system's quartile_logi calibration) to map embedding distances to calibrated confidence scores

## Validation Plan: GPT-2 on Shakespeare

### Rationale

- GPT-2 is small enough to train from scratch in reasonable time
- Shakespeare provides a constrained but rich vocabulary with clear stylistic patterns
- Analogical structure is testable: character relationships, thematic parallels, meter patterns
- Standard cross-entropy GPT-2 on Shakespeare is a well-characterized baseline

### Experiment Design

- **Baseline**: Train GPT-2 on Shakespeare with standard cross-entropy loss
- **Cosine-only**: Train GPT-2 on Shakespeare with pairwise cosine similarity loss using the three-stage curriculum
- **Hybrid**: Cross-entropy for stage 1, pairwise cosine similarity for stages 2–3. Tests whether cosine similarity needs the bootstrap phase — if hybrid significantly outperforms cosine-only, that's evidence the coordinate system matters. If cosine-only converges anyway (just slower), that validates the stronger thesis.

| Condition | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|
| Baseline | Cross-entropy | Cross-entropy | Cross-entropy |
| Cosine-only | Pairwise cosine | Pairwise cosine | Pairwise cosine |
| Hybrid | Cross-entropy | Pairwise cosine | Pairwise cosine |

### Evaluation

- The experiment MUST measure perplexity on a held-out set (via cross-entropy, even for the cosine-trained model) to enable direct comparison
- The experiment MUST evaluate generation quality: coherence, grammatical validity, and stylistic fidelity to Shakespeare
- The experiment SHOULD probe the embedding space for semantic arithmetic (character relationships, thematic vectors) to test the analogical reasoning hypothesis
- The experiment SHOULD measure convergence speed (training steps to comparable generation quality)
- The experiment SHOULD measure embedding geometry: isotropy, clustering quality, semantic neighborhood coherence
- The experiment SHOULD evaluate generation coherence on out-of-distribution prompts (modern English, non-Shakespearean literary styles) to test whether cosine-trained models degrade more gracefully than cross-entropy baselines
- The experiment SHOULD monitor pairwise accuracy during training — if it saturates near 1.0 (the model predicts all pairs correctly), the difficulty gradient has collapsed and training signal is exhausted
- The system SHOULD track the distribution of pairwise cosine similarity differences within each batch; narrowing variance indicates signal decay
- If Shakespeare proves too small for stable signal, the experiment SHOULD fall back to a larger corpus (e.g., Project Gutenberg prose) before concluding the approach doesn't work

### Expected Outcomes

- The cosine-trained model converges slower on perplexity
- The cosine-trained model produces richer embedding geometry with more meaningful semantic neighborhoods
- Generation quality at the response level (stage 3) is competitive with or exceeds cross-entropy baseline
- The hybrid approach MAY offer both fast convergence and semantic richness

## Open Questions

- What is the optimal batch size for pairwise comparison? Sampling strategies need exploration for large batches.
- Should the embedding comparison use the final hidden state, an intermediate layer, or a learned projection?
- How should curriculum transition thresholds be set? Fixed epoch counts vs. validation-based advancement?
- Does the approach benefit from a temperature parameter on cosine similarity (analogous to the beta parameter in DPO)?
- What is the optimal ratio of geometric pairs to anchor pairs to embedding pairs per batch?
- Does the system need a repulsive loss term (negative examples or contrastive pairs) to establish global embedding structure, or do the three pair types provide sufficient implicit contrast?
- What is the minimum corpus size at which the difficulty gradient across examples produces stable pairwise signal? (Shakespeare may be small enough to expose this.)
- How many anchor pairs per observation are needed to prevent rotation drift? Is one random embedding sufficient, or does the system need k anchors per hidden state?

## Error Handling

- **Degenerate embeddings**: If cosine similarity produces NaN (zero-norm vectors), the affected observation MUST be excluded from the batch rather than producing undefined gradients
- **Curriculum regression**: If pairwise accuracy drops after stage transition, the system SHOULD revert to the previous stage's weights and retry with a reduced learning rate
- **Pair imbalance**: If a mini-batch contains observations with very similar cosine similarity gaps, the pairwise signal becomes weak. The system SHOULD log batches where the maximum pairwise difference falls below a configurable threshold
- **Embedding collapse**: If monitoring detects embedding pairwise distances falling below a threshold, the system SHOULD increase the weight of embedding pairs or apply stop-gradient on target embeddings
