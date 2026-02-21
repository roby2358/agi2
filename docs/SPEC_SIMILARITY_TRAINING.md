# Technical Specification: Pairwise Cosine Similarity Training for Language Models

## Purpose

This system trains language models using pairwise cosine similarity loss in embedding space rather than cross-entropy next-token loss. It draws from pairwise similarity loss used in financial prediction (see SPEC_TRAINING.md) and extends it to language generation through a curriculum that progresses from single-token to full-response training.

The core thesis: training on semantic distance in embedding space — rather than discrete token identity — yields models with stronger analogical reasoning and compositional generalization. Cross-entropy treats the vocabulary as an unstructured set where "almost right" and "completely wrong" are penalized identically. Cosine similarity operates on the same manifold where semantic arithmetic works (king - man + woman = queen), directly rewarding the model for maintaining geometric relationships between meanings.

## Key Design Decisions

- **Pairwise comparison, not absolute loss**: The system compares two (prompt, response) observations and bases loss on the difference in cosine similarity to known-good continuations. Both observations are valid corpus continuations; the gradient signal comes from the model being more or less accurate at reproducing known geometric relationships across different examples. This is a geometric constraint satisfaction problem — the model learns to place its hidden state vectors where the embedding geometry says they should be.
- **Cosine similarity as distance metric**: The loss operates in the same space where semantic arithmetic works. The model is directly optimized for semantic fidelity, not token identity.
- **Curriculum progression**: Training begins with single-token comparison and gradually extends to multi-token and full-response comparison. The loss function remains unchanged across stages — only the observation window widens.
- **No separate reward model**: Quality signal comes directly from cosine similarity to known-good text, eliminating the reward model training stage required by RLHF.
- **Single-phase training**: The curriculum eliminates the two-phase pretrain-then-align pipeline. Token grounding, sequence coherence, and stylistic fidelity emerge from the same loss function applied at increasing observation scale. The corpus provides both knowledge and preference signal — there is no separate alignment stage.

## Functional Requirements

### Loss Function

- The system MUST compute loss by comparing pairs of observations, where each observation consists of a prompt and a known-good continuation
- For each observation, the system MUST compute the cosine similarity between the model's output embedding (the hidden state before the vocabulary projection) and the known-correct token's embedding from the embedding matrix
- The system MUST apply pairwise cosine loss to the similarity scores: the loss is the gap between the model's output similarity and the known-good token's position in embedding space, compared across observations to provide relative gradient signal
- The pairwise loss MUST follow the same structure as the financial prediction system's pairwise loss (see SPEC_TRAINING.md), with cosine similarity replacing raw prediction scores as the comparison metric
- The system MUST NOT apply any activation function (sigmoid, softmax) to the model's hidden state before computing cosine similarity — the comparison operates on raw embedding geometry
- The system SHOULD reuse forward pass activations across pairs within a mini-batch rather than recomputing — each observation's cosine similarity score is independent of its pair partner, so N observations require N forward passes regardless of pairing count

### Curriculum Training

The system MUST train in three progressive stages. The loss function does not change between stages — only the number of tokens compared per observation.

#### Stage 1: Single Token

- Each observation MUST consist of a prompt with a single known next token
- The system MUST compute one output embedding at the final prompt position and compare it against the known token's embedding
- This stage teaches basic token prediction through semantic proximity

#### Stage 2: Short Sequences

- Each observation MUST consist of a prompt with a known continuation of 2–5 tokens
- The system MUST compute output embeddings at each position in the continuation window
- The system SHOULD apply exponentially decaying position weights (earlier positions weighted higher) as the default aggregation into a single observation score
- Rationale: errors at earlier positions cascade, making later-position similarity scores less meaningful
- The system SHOULD compare front-weighted vs. uniform aggregation during validation to confirm the benefit
- This stage introduces sequence-level coherence across multiple generation steps

#### Stage 3: Full Responses

- Each observation MUST consist of a complete (prompt, response) pair
- The system MUST compute output embeddings across the full response length and aggregate via arithmetic mean
- This stage addresses exposure bias — the model evaluates its own multi-token trajectory against known-good responses, learning from its own compounding errors

#### Stage Transitions

- The system SHOULD advance to the next stage when pairwise accuracy on a held-out set plateaus
- Each stage MUST begin with the weights from the previous stage
- Learning rate MAY be reset or reduced at each transition

### Multi-Token Aggregation

- When comparing sequences longer than one token, the system MUST aggregate per-position cosine similarities into a single score
- For stage 2, the system SHOULD use exponentially decaying position weights as the default aggregation, with earlier positions weighted higher
- For stage 3, the default aggregation MUST be arithmetic mean across all positions (uniform weighting is defensible at response scale where the goal is holistic quality)
- Alternative aggregations (minimum similarity, other weighting schemes) MAY be explored

### Observation Pairing

- The system MUST pair observations within each mini-batch for pairwise loss computation
- Both observations in a pair MUST be valid corpus continuations
- The loss for each observation is the gap between the model's output cosine similarity and the known-good token's position in embedding space — the model is learning to place its hidden state vectors where the embedding geometry says they should be
- Pairwise comparison provides relative gradient signal: the model receives stronger gradients from observations where its geometric reconstruction is less accurate
- No external quality labels are required; the signal is the model's fidelity to the embedding space's existing geometric relationships
- The system SHOULD sample pairs rather than computing all combinations when batch sizes make exhaustive pairing prohibitive

### Inference-Time Token Selection

- At inference time, the system MUST compute cosine similarity between the model's output hidden state and every token embedding in the vocabulary matrix to produce a score distribution over tokens
- The system MUST support greedy selection (argmax over similarities) and sampling (softmax over similarity scores with temperature) for token selection
- The system SHOULD apply a temperature parameter to the similarity scores before softmax to control generation diversity

## Relationship to DPO

Direct Preference Optimization (DPO, Rafailov et al. 2023) is the closest existing technique. Both use pairwise comparison of responses. The key differences:

| Aspect | DPO | This System |
|--------|-----|-------------|
| Comparison metric | Log-probability ratios | Cosine similarity in embedding space |
| Base model dependency | Requires cross-entropy-pretrained base | Trains from scratch or fine-tunes |
| Quality signal source | Human preference labels | Cosine distance to known-good text |
| Training scope | Response-level only | Single token through full response (curriculum) |
| What it optimizes | Token probability distributions | Semantic geometry of representations |

### The DPO Objection

DPO works because it stands on a cross-entropy-trained base model that already understands language. The log-probability comparison is meaningful because the base model has learned a calibrated distribution over tokens. This system removes that foundation — without cross-entropy pretraining, the embedding space may lack the geometric structure that makes cosine similarity meaningful. The risk: cosine similarity training never converges to the point where its geometric advantages matter, because the geometry itself is malformed without the bootstrap phase.

**Counter-argument**: Embedding spaces develop geometric structure through co-occurrence patterns in training data, not specifically from cross-entropy loss. Cosine similarity loss still rewards clustering semantically related tokens and separating unrelated ones — through a different gradient pathway. The structure may take longer to emerge but is not dependent on cross-entropy.

This is an empirical question. The validation experiment tests it directly.

## Additional Counter-Arguments and Mitigations

### Embedding Space Anisotropy

Embedding spaces are high-dimensional and potentially anisotropic — not all dimensions carry meaning equally. Cosine similarity treats all dimensions as equal weight. If the space has degenerate dimensions, cosine similarity optimizes for a distorted geometry.

- The system SHOULD monitor embedding space isotropy during training
- The system MAY apply learned dimension weighting or whitening transforms if anisotropy develops

### Embedding Matrix Stability

The cosine loss compares hidden states against embedding vectors, but those embedding vectors are themselves being trained. If the embedding matrix is optimized simultaneously as both the comparison target and a trainable parameter, there is a collapse mode where embeddings converge toward each other to trivially maximize similarity.

- The system SHOULD monitor pairwise distances within the embedding matrix during training to detect convergence/collapse
- The system SHOULD consider freezing the embedding matrix periodically or using a stop-gradient on the target embedding when computing cosine similarity (analogous to target networks in RL)
- If using weight tying between input embeddings and output projection, the system MUST verify that cosine loss gradients do not destabilize the shared weights

### Discrete Token Validity

Language is discrete. Embeddings close in cosine distance may correspond to invalid tokens. Cross-entropy's harshness forces commitment to valid tokens — cosine similarity's softness may allow the model to settle in "meaning-adjacent" regions that produce invalid output.

- The pairwise comparison partially mitigates this: valid tokens consistently score higher similarity than invalid ones, so the model learns to prefer them through relative similarity
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

- What is the optimal batch size for pairwise comparison? Exhaustive pairing within large batches may be prohibitive — sampling strategies need exploration.
- Should the embedding comparison use the final hidden state, an intermediate layer, or a learned projection?
- How should curriculum transition thresholds be set? Fixed epoch counts vs. validation-based advancement?
- Does the approach benefit from a temperature parameter on cosine similarity (analogous to the beta parameter in DPO)?
- For multi-token comparison, should later positions be weighted higher since they depend on earlier predictions?
- Does the system need a repulsive loss term (negative examples or contrastive pairs) to establish global embedding structure, or does the pairwise comparison provide sufficient implicit contrast?
- Should the embedding comparison target use a stop-gradient to prevent co-adaptation between the model's hidden state and the target embeddings?
- What is the minimum corpus size at which the difficulty gradient across examples produces stable pairwise signal? (Shakespeare may be small enough to expose this.)

## Error Handling

- **Degenerate embeddings**: If cosine similarity produces NaN (zero-norm vectors), the affected observation MUST be excluded from the batch rather than producing undefined gradients
- **Curriculum regression**: If pairwise accuracy drops after stage transition, the system SHOULD revert to the previous stage's weights and retry with a reduced learning rate
- **Pair imbalance**: If a mini-batch contains observations of highly uneven quality (all very similar cosine scores), the pairwise signal becomes weak. The system SHOULD log batches where the maximum pairwise difference falls below a configurable threshold
