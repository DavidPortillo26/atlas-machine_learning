#!/usr/bin/env python3
"""
Module for creating padding and look-ahead masks for Transformer models.
"""
import tensorflow as tf
def create_padding_mask(seq):
    """
    Create a padding mask for a given sequence.

    Marks all padding tokens (0) in the input sequence with 1, and non-padding
    tokens with 0. This mask is suitable for use in Transformer attention.

    Parameters
    ----------
    seq : tf.Tensor
        Tensor of shape (batch_size, seq_len) with padding tokens as 0.

    Returns
    -------
    tf.Tensor
        Padding mask of shape (batch_size, 1, 1, seq_len), with 1s at
        padding positions and 0s elsewhere.
    """
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)  # 1 for PAD tokens
    return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    Create a look-ahead mask to prevent attention to future tokens.

    The look-ahead mask sets all positions in the upper triangle of a
    (size, size) matrix to 1, so the model cannot "see" future tokens.

    Parameters
    ----------
    size : int
        Sequence length of the target input.

    Returns
    -------
    tf.Tensor
        Look-ahead mask of shape (size, size), where masked positions are 1
        and unmasked positions are 0.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (size, size)


def create_masks(inputs, target, fixed_len=36):
    """
    Create encoder, combined, and decoder masks for Transformer training.

    Generates the three masks used in the Transformer:
    1. Encoder padding mask for masking input padding tokens.
    2. Combined mask for masking future tokens and padding in the decoder.
    3. Decoder mask for the 2nd attention block, which masks encoder padding.

    Parameters
    ----------
    inputs : tf.Tensor
        Tensor of shape (batch_size, seq_len_in) representing encoder input.
    target : tf.Tensor
        Tensor of shape (batch_size, seq_len_out) representing decoder input.
    fixed_len : int, optional
        Fixed sequence length to truncate inputs and target, default is 36.

    Returns
    -------
    tuple of tf.Tensor
        - encoder_mask : tf.Tensor of shape (batch_size, 1, 1, fixed_len)
        - combined_mask : tf.Tensor of shape (batch_size, 1, fixed_len,
          fixed_len)
        - decoder_mask : tf.Tensor of shape (batch_size, 1, 1, fixed_len)
    """
    # Force inputs and target to fixed_len
    inputs = inputs[:, :fixed_len]
    target = target[:, :fixed_len]

    # Encoder padding mask
    encoder_mask = create_padding_mask(inputs)

    # Decoder target padding mask
    decoder_padding_mask = create_padding_mask(target)

    # Look-ahead mask (to mask future tokens)
    seq_len = tf.shape(target)[1]
    look_ahead_mask = create_look_ahead_mask(seq_len)

    # Combine look-ahead and target padding mask
    combined_mask = tf.maximum(
        decoder_padding_mask[:, :, :, :seq_len],
        look_ahead_mask[tf.newaxis, tf.newaxis, :, :]
    )

    # Decoder's 2nd attention block uses encoder padding mask
    decoder_mask = encoder_mask

    return encoder_mask, combined_mask, decoder_mask
