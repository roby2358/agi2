"""
Text Generation

This module provides text generation functions for the AGI2 model.

Generation uses cosine similarity between the model's hidden state and the
embedding matrix to produce token scores, rather than raw logits. This matches
the cosine similarity training objective.
"""

from typing import List

import torch
import torch.nn.functional as F

from .basic_tokenizer import BasicTokenizer


def _hidden_to_scores(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute token scores from the last hidden state via cosine similarity.

    Returns a (vocab_size,) tensor of similarity scores scaled by temperature.
    """
    hidden_states = model._run_transformer(input_ids)
    last_hidden = hidden_states[0, -1, :]  # (n_embd,)

    emb_weight = model.token_embeddings.embedding.weight  # (vocab_size, n_embd)

    # Cosine similarity between hidden state and every token embedding
    scores = F.cosine_similarity(
        last_hidden.unsqueeze(0), emb_weight, dim=-1
    )  # (vocab_size,)

    return scores / temperature


def _apply_top_k(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero out all but the top-k scoring tokens."""
    top_k_scores, top_k_indices = torch.topk(scores, top_k)
    filtered = torch.full_like(scores, -float("inf"))
    filtered[top_k_indices] = top_k_scores
    return filtered


def _apply_top_p(scores: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out tokens outside the nucleus (top-p cumulative probability)."""
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_scores, dim=-1), dim=-1)

    sorted_mask = cumulative_probs > top_p
    sorted_mask[1:] = sorted_mask[:-1].clone()
    sorted_mask[0] = False

    indices_to_remove = sorted_indices[sorted_mask]
    scores[indices_to_remove] = -float("inf")
    return scores


def generate_text(
    model: torch.nn.Module,
    prompt: str,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    tokenizer: BasicTokenizer,
    device: str,
) -> str:
    """
    Generate text from a prompt using cosine similarity scoring.

    Args:
        model: The trained AGI2 model
        prompt: Input text prompt
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature applied to similarity scores
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        tokenizer: Tokenizer for encoding/decoding
        device: Device to run generation on

    Returns:
        Generated text string
    """
    model = model.to(device)
    model.eval()

    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated_ids = input_ids.clone()

    eos_id = tokenizer.vocab.get("<EOS>", -1)

    with torch.inference_mode():
        for _ in range(max_length):
            scores = _hidden_to_scores(model, generated_ids, temperature)

            scores = _apply_top_k(scores, top_k)
            scores = _apply_top_p(scores, top_p)

            probs = F.softmax(scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == eos_id:
                break

    return tokenizer.decode(generated_ids[0].tolist())


def generate_with_beam_search(
    model: torch.nn.Module,
    prompt: str,
    max_length: int,
    beam_width: int,
    temperature: float,
    tokenizer: BasicTokenizer,
    device: str,
) -> List[str]:
    """
    Generate text using beam search with cosine similarity scoring.

    Args:
        model: The trained AGI2 model
        prompt: Input text prompt
        max_length: Maximum number of tokens to generate
        beam_width: Number of beams to maintain
        temperature: Sampling temperature applied to similarity scores
        tokenizer: Tokenizer for encoding/decoding
        device: Device to run generation on

    Returns:
        List of generated text strings, one per beam
    """
    model = model.to(device)
    model.eval()

    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    beams: list[tuple[torch.Tensor, float]] = [(input_ids.clone(), 0.0)]
    eos_id = tokenizer.vocab.get("<EOS>", -1)

    with torch.inference_mode():
        for _ in range(max_length):
            new_beams: list[tuple[torch.Tensor, float]] = []

            for beam_seq, beam_score in beams:
                scores = _hidden_to_scores(model, beam_seq, temperature)
                top_scores, top_indices = torch.topk(scores, beam_width)

                for score_val, token_id in zip(top_scores, top_indices):
                    new_seq = torch.cat(
                        [beam_seq, token_id.unsqueeze(0).unsqueeze(0)], dim=1
                    )
                    new_score = beam_score + score_val.item()
                    new_beams.append((new_seq, new_score))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            if all(b[0][0, -1].item() == eos_id for b in beams):
                break

    return [tokenizer.decode(seq[0].tolist()) for seq, _ in beams]


def generate_interactive(
    model: torch.nn.Module,
    tokenizer: BasicTokenizer,
    max_length: int,
    temperature: float,
    device: str,
) -> None:
    """
    Interactive text generation loop.

    Args:
        model: The trained AGI2 model
        tokenizer: Tokenizer to use for encoding/decoding
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        device: Device to run generation on
    """
    print("Interactive text generation (type 'quit' to exit)")
    print("=" * 50)

    while True:
        try:
            prompt = input("\nEnter your prompt: ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not prompt:
                continue

            print("\nGenerating...")
            generated_text = generate_text(
                model,
                prompt,
                max_length,
                temperature,
                50,
                0.9,
                tokenizer,
                device,
            )

            print(f"\nGenerated text:\n{generated_text}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue
