import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        # raise NotImplementedError()
        self.d_latent = d_latent
        self.n_tokens = n_tokens

        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=d_latent, nhead=8, dropout=0.1)
        self.proj = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # raise NotImplementedError()

        B, h, w = x.shape
        L = h * w  

        x_seq = x.view(B, L)
        emb = self.token_embedding(x_seq)
        start_token = torch.zeros(B, 1, self.d_latent, device=emb.device, dtype=emb.dtype)
        emb_shifted = torch.cat([start_token, emb[:, :-1, :]], dim=1)
        emb_shifted = emb_shifted.transpose(0, 1)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(L).to(emb_shifted.device)

        transformer_out = self.transformer(emb_shifted, mask=causal_mask)
        transformer_out = transformer_out.transpose(0, 1)  

        logits = self.proj(transformer_out)
        logits = logits.view(B, h, w, self.n_tokens)

        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        # raise NotImplementedError()
        device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        L = h * w  
        generated = torch.zeros(B, L, dtype=torch.long, device=device)

        for t in range(L):
            cur_tokens = generated[:, :t+1]
            emb = self.token_embedding(cur_tokens)
            start_token = torch.zeros(B, 1, self.d_latent, device=device, dtype=emb.dtype)
            emb_shifted = torch.cat([start_token, emb[:, :-1, :]], dim=1)
            emb_shifted = emb_shifted.transpose(0, 1)
            cur_len = emb_shifted.size(0)
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(cur_len).to(device)
            transformer_out = self.transformer(emb_shifted, mask=causal_mask)
            transformer_out = transformer_out.transpose(0, 1)  
            logits = self.proj(transformer_out[:, -1, :])  
            next_token = logits.argmax(dim=-1)
            generated[:, t] = next_token

        generated = generated.view(B, h, w)
        return generated
