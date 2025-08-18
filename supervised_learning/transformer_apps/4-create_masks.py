#!/usr/bin/env python3
import tensorflow as tf


def create_padding_mask(seq):
    """
    Creates a padding mask for a given sequence.
    seq: tf.Tensor of shape (batch_size, seq_len)
    Returns: mask of shape (batch_size, 1, 1, seq_len)
    """
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)  # 1 for PAD tokens
    return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    Creates a look-ahead mask to mask future tokens.
    size: int (seq_len_out)
    Returns: mask of shape (seq_len_out, seq_len_out)
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len_out, seq_len_out)


def create_masks(inputs, target, fixed_len=36):
    """
    Creates encoder, combined, and decoder masks for Transformer training.
    Ensures masks are aligned to the expected sequence length.
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
