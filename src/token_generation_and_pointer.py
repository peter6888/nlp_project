'''
token_generation_and_pointer.py
'''
import tensorflow as tf
import numpy as np
import argparse

# below function will take parameters (self, decoder_outputs, hps, vsize,
# *extra_args):


def tokenization(encoder_attn_score, decoder_state, encoder_context,
                 decoder_context, attn_score_size, vocab_size,
                 use_pointer=False):
    '''
    Token generation and pointer (2.3, p.3). u_t = 1 if we
    want to pay attention to or copy the inputs and u_t = 0 if we do not. The
    tokenization mechanism allows our model to learn the representation of
    words it had not seen in training by copying the representation of an
    unknown word from its input (p.5 of article). p(u_t) = p_gen from Abi.

    Args: Dictionary params
        encoder_attn_score: tensor of attenion scores from Equation 4,
            size = [batch_size x max_enc_steps]
        decoder_state: decoder state tensor at timestep t,
            size = [batch_size x hidden_dim]
        encoder_context: input context vector at timestep t,
            size = [batch_size x hidden_dim]
        decoder_context: decoder context vector at timestep t,
            size = [batch_size x hidden_dim]
        attn_score_size: the attn_score_size from hyper-parameter
        vocab_size: vocabulary size,
            size = [vocab_size]
        use_pointer: boolean, True = pointer mechanism, False = no pointer

    Returns:
        final_dists: the final distribution of words for each timestep y_t
        vocab_scores: unnormalized scores for each word for each timestep t
    '''
    # Variables
    attentions = tf.concat(
        values=[decoder_state, encoder_context, decoder_context], axis=1)

    # Hyperparameters
    attn_conc_size = attentions.get_shape().as_list()[1]

    # Initializations
    xavier_init = tf.contrib.layers.xavier_initializer()
    zeros_init = tf.zeros_initializer()

    # Tokenization with the pointer mechanism
    # TODO: pointer has not yet been implemented
    if use_pointer:
        # TODO: Calculate number of out-of-vocabulary words
        oov_size = 0
        # Expanded vocabulary size
        new_vocab_size = vocab_size + oov_size

        with tf.variable_scope("Tokenization"):
            W_out = tf.get_variable('W_out',
                                    shape=[attn_conc_size, new_vocab_size],
                                    initializer=xavier_init)
            b_out = tf.get_variable("b_out",
                                    shape=[new_vocab_size],
                                    initializer=zeros_init)

        # Reuse variables across timesteps
        tf.get_variable_scope().reuse_variables()

        # Equation 9
        vocab_scores = tf.nn.xw_plus_b(attentions, W_out, b_out)
        vocab_dists = tf.nn.softmax(vocab_scores)

        # Equation 10; encoder_attn_size is the output of Equation 4
        copy_distn = encoder_attn_score

        # TODO: turn copy_distn into a new_vocab_size-vector

        # TODO: Being implemented by Apurva
        with tf.variable_scope("Copy_mechanism", reuse=tf.AUTO_REUSE):
            W_u = tf.get_variable('W_u',
                                  shape=[attn_conc_size, 1],
                                  initializer=xavier_init)
            b_u = tf.get_variable("b_u",
                                  shape=[1],
                                  initializer=zeros_init)

        # Probability of using copy mechanism for decoding step t
        # Equation 11; pointer size = [batch_size x 1]
        z_u = tf.nn.xw_plus_b(attentions, W_u, b_u)
        pointer = tf.nn.sigmoid(z_u)

        # Final probability distribution for output token y_t (Equation 12)
        final_dists = tf.add(pointer * copy_distn, (1 - pointer) * vocab_dists)

    # Tokenization with the token-generation softmax layer
    else:
        with tf.variable_scope("Tokenization"):
            W_out = tf.get_variable('W_out',
                                    shape=[attn_conc_size, vocab_size],
                                    initializer=xavier_init)
            b_out = tf.get_variable("b_out",
                                    shape=[vocab_size],
                                    initializer=zeros_init)

        # Reuse variables across timesteps
        tf.get_variable_scope().reuse_variables()

        # Equation 9
        vocab_scores = tf.nn.xw_plus_b(attentions, W_out, b_out)
        final_dists = tf.nn.softmax(vocab_scores)

    return final_dists, vocab_scores


def test_tokenization(args):
    ''' test tokenization function
    python token_generation_and_pointer.py test1
    :param args:
    :return:
    '''
    batch_size = 16
    decoder_hidden_size = 512
    encoder_hidden_size = 256
    decoder_t = 6
    encoder_t = 5
    vsize = 50000

    attn_score = np.random.randn(batch_size, decoder_t)
    attn_score = tf.convert_to_tensor(attn_score, np.float32)

    dec_hidden_state = np.random.randn(batch_size, decoder_hidden_size)
    dec_hidden_state = tf.convert_to_tensor(dec_hidden_state, np.float32)

    enc_context = np.random.randn(batch_size, encoder_hidden_size)
    enc_context = tf.convert_to_tensor(enc_context, np.float32)

    dec_context = np.random.randn(batch_size, decoder_hidden_size)
    dec_context = tf.convert_to_tensor(dec_context, np.float32)

    generated = tokenization(attn_score, dec_hidden_state,
                             enc_context, dec_context, encoder_t, vsize)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ret = sess.run(generated)
        print(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test tokenization for matching parameter dimensions')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser(
        'test1', help='Test tokenization function')
    command_parser.set_defaults(func=test_tokenization)

    ARGS = parser.parse_args()
    if not hasattr(ARGS, 'func'):
        parser.print_help()
    else:
        ARGS.func(ARGS)
