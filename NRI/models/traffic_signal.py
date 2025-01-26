import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from NRI.models.utils import SinusoidalPositionalEncoding

from SinD.dataset.type import EncodedTrafficSignal

class TrafficSignalSequenceModel(nn.Module):
    '''
    abstract interface for forecasting traffic signals
    '''
    def __init__(self):
        super().__init__()

    def forward(
        self, 
    ) -> torch.FloatTensor:
        raise NotImplementedError()

    def training_loss(
        self, 
        signals: torch.IntTensor, 
        obs_len: int, 
        pred_len: int
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
        Args:
        ---
        - signals: [B, t]
        - obs_len: observation length
        - pred_len: prediction length

        Returns:
        ---
        - logits: [B, pred_len, vocab_size]
        - loss
        '''
        raise NotImplementedError()
    
    def most_probable_outcome(
        self, 
        signals: torch.IntTensor, 
        n_steps: int
    ) -> torch.LongTensor:
        '''
        forecast most probable signals in next n_steps given past signals

        Args:
        ---
        - signals: past signals
        - n_steps: no. of steps to be predicted

        Returns:
        ---
        - predicted signals: [B, n_steps]
        '''
        raise NotImplementedError()
    
class TrafficSignalSeq2SeqModel(TrafficSignalSequenceModel):
    '''
    forecast future traffic signals with seq2seq
    '''
    
    def __init__(
        self,
        hid_dim: int,
        num_layers: int = 1,
        do_prob: float = 0.0,
        autoregression: bool = False,
        vocab_size: Optional[int] = None
    ):
        '''
        Args:
        ---
        - hid_dim: hidden state dim
        - num_layers: no. of layers in encoder and decoder
        - do_prob: dropout prob.
        - autoregression: feed in most likely outcomes back to the decoder if set to `True`. Otherwise feed the `[bos]` token during every step.
        - vocab_size: sequence vocabulary size
        '''
        super().__init__()

        if vocab_size is None:
            vocab_size = len(EncodedTrafficSignal)

        self.vocab_size = vocab_size
        self.autoregression = autoregression

        # add [bos] token
        self.embedding = nn.Embedding(vocab_size + 1, hid_dim, max_norm=1.0)

        self.encoder = nn.LSTM(
            hid_dim, 
            hid_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=do_prob
        )
        
        self.decoder = nn.LSTM(
            hid_dim, 
            hid_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=do_prob
        )

        self.logit_layer = nn.Linear(hid_dim, vocab_size)

    def forward(
        self, 
        signals: torch.IntTensor, 
        n_steps: int,
    ):
        '''
        Args:
        ---
        - signals: [B, t]
        - n_steps: no. of steps into future to be predicted

        Returns:
        ---
        - logits: [B, n_steps, vocab_size]
        '''
        batch_shape = signals.shape[:-1]
        signal_embedding = self.embedding(signals)
        
        _, hidden_state = self.encoder(signal_embedding)

        if not self.autoregression:
            mask_tokens = torch.zeros((*batch_shape, n_steps), device=signals.device, dtype=torch.int32).fill_(self.vocab_size)
            mask_tokens = self.embedding.forward(mask_tokens)

            signal_embedding, _ = self.decoder.forward(mask_tokens, hidden_state)
            logits = self.logit_layer.forward(signal_embedding)

        else:
            prev_tokens = torch.zeros((*batch_shape, 1), device=signals.device, dtype=torch.int32).fill_(self.vocab_size)
            prev_tokens = self.embedding.forward(prev_tokens)

            logits = torch.zeros((*batch_shape, n_steps, self.vocab_size), device=signals.device)

            for step in range(n_steps):
                signal_embedding, hidden_state = self.decoder.forward(prev_tokens, hidden_state)
                logits[..., step, :] = self.logit_layer.forward(signal_embedding[..., 0, :])
                
                prev_tokens = logits.argmax(dim=-1).detach()
                prev_tokens = self.embedding.forward(prev_tokens)

        return logits

    def training_loss(
        self, 
        signals: torch.IntTensor, 
        obs_len: int, 
        pred_len: int
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
        Args:
        ---
        - signals: [B, t]
        - obs_len: observation length
        - pred_len: prediction length

        Returns:
        ---
        - logits: [B, pred_len, vocab_size]
        - loss
        '''
        logits = self.forward(signals[..., :obs_len], pred_len)
            
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            signals[:, obs_len:].reshape(-1).to(torch.long)
        )

        assert not loss.isnan().any()
        return logits, loss
    
    def most_probable_outcome(
        self, 
        signals: torch.IntTensor, 
        n_steps: int
    ) -> torch.LongTensor:
        logits = self.forward(signals, n_steps)
        return logits.argmax(dim=-1)

class TrafficSignalTransformerModel(TrafficSignalSequenceModel):
    '''
    forecast future traffic signals with decoder-only transformer
    '''
    
    def __init__(
        self,
        hid_dim: int,
        ffn_dim: int,
        n_head: int,
        max_seq_len: int,
        num_layers: int = 1,
        do_prob: float = 0.0,
        vocab_size: Optional[int] = None
    ):
        '''
        Args:
        ---
        - hid_dim: hidden state dim
        - ffn_dim: transformer ffn dim
        - n_head: no. of head in multihead attention
        - max_seq_len: max sequence length
        - num_layers: no. of transformer blocks
        - do_prob: dropout prob.
        - vocab_size: vocab size
        '''
        super().__init__()

        if vocab_size is None:
            vocab_size = len(EncodedTrafficSignal)

        self.vocab_size = vocab_size

        # add [bos] token
        self.embedding = nn.Embedding(vocab_size + 1, hid_dim)

        self.pe = SinusoidalPositionalEncoding(dim=hid_dim, max_len=max_seq_len + 1, do_prob=do_prob)

        self.logit_layer = nn.Linear(hid_dim, vocab_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=n_head,
            dim_feedforward=ffn_dim,
            batch_first=True,
            dropout=do_prob
        )

        self.seq_model = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

    def forward(
        self, 
        signals: torch.IntTensor, 
    ) -> torch.FloatTensor:
        '''
        Args:
        ---
        - signals: [B, t]

        Returns:
        ---
        - logits: [B, t + 1, vocab_size]
        '''
        batch_size, seq_len = signals.shape[:2]

        # prepend [bos] token
        signals = torch.cat([
            torch.zeros((batch_size, 1), dtype=torch.int32, device=signals.device).fill_(self.vocab_size), 
            signals
        ], dim=1)

        signals = self.embedding.forward(signals)
        signals = self.pe.forward(signals)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len + 1, device=signals.device)

        # transformer blocks
        output = self.seq_model.forward(signals, causal_mask, is_causal=True)

        return self.logit_layer.forward(output)

    def training_loss(
        self, 
        signals: torch.IntTensor, 
        obs_len: int, 
        pred_len: Optional[int] = None,
        include_obs_loss: bool = False
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
        Args:
        ---
        - signals: [B, t]
        - obs_len: observation length
        - pred_len: not used. just put here for consistency.
        - include_obs_loss: compute loss in observed segment if `True`. default `False`.

        Returns:
        ---
        - logits: [B, pred_len, vocab_size]
        - loss
        '''
        logits = self.forward(signals[:, :-1])

        if include_obs_loss:
            loss = F.cross_entropy(
                logits[:, 1:].reshape(-1, self.vocab_size), 
                signals[:, 1:].reshape(-1).to(torch.long)
            )
        else:
            loss = F.cross_entropy(
                logits[:, obs_len:].reshape(-1, self.vocab_size), 
                signals[:, obs_len:].reshape(-1).to(torch.long)
            )

        return logits[:, obs_len:], loss
    
    def most_probable_outcome(
        self, 
        signals: torch.IntTensor, 
        n_steps: int
    ) -> torch.LongTensor:
        batch_size = signals.size(0)

        outcomes = torch.zeros((batch_size, n_steps), dtype=torch.long, device=signals.device)

        for step in range(n_steps):
            next_tokens = self.forward(signals)[:, -1].argmax(dim=-1)

            outcomes[:, step] = next_tokens
            signals = torch.concat([signals, next_tokens[..., None]], dim=1)
            
        return outcomes