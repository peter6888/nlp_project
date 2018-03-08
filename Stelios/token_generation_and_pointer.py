'''
token_generation_and_pointer.py
'''
import tensorflow as tf
import numpy as np
import argparse

batch_size = 16
decoder_hidden_size = 512
encoder_hidden_size = 256
decoder_t = 6
vsize = 50000


def tokenization(alpha_e_ti, h_d_t, c_e_t, c_d_t, vsize, use_pointer=False):
    '''
    Implementation of token generation and pointer (2.3, p.3). u_t = 1 if we
    want to pay attention to or copy the inputs and u_t = 0 if we do not. The
    tokenization mechanism allows our model to learn the representation of
    words it had not seen in training by copying the representation of an
    unknown word from its input (p.5 of article). p(u_t) = p_gen from Abi.

    Args:
        u_t:
        h_d_t: decoder state tensor at timestep t
        c_e_t: input context vector at timestep t
        c_d_t: decoder context vector at timestep t
        vsize: vocabulary size scalar
        alpha_e_ti: tensor of attenion scores from equation (4)
        use_pointer: boolean, True = pointer mechanism, False = no pointer

    Returns:
        y_t_distn: token probability distribution y_t
    '''
    # Variables
    attn_conc = tf.concat(values=[h_d_t, c_e_t, c_d_t], axis=1)

    # Hyperparameters
    # TODO: I am not sure whether row dim is vize or alpha_e_ti size
    attn_conc_size = attn_conc.get_shape()[1].value
    attn_score_size = alpha_e_ti.get_shape()[1].value

    # Initializations
    xavier_init = tf.contrib.layers.xavier_initializer()
    zeros_init = tf.zeros_initializer()

    # Tokenization with the pointer mechanism
    # Equation 10; alpha_e_ti is taken from Equation 4
    y_t_u_zero = alpha_e_ti

    # Tokenization with the token-generation softmax layer
    # TODO: I need to decide between vsize vs attn_score size
    with tf.variable_scope("Tokenization"):
        W_out = tf.get_variable('W_out',
                                shape=[attn_conc_size, vsize],
                                initializer=xavier_init)
        b_out = tf.get_variable("b_out",
                                shape=[vsize],
                                initializer=zeros_init)

    # Reuse variables across timesteps
    tf.get_variable_scope().reuse_variables()

    # Equation 9
    z_out = tf.nn.xw_plus_b(attn_conc, W_out, b_out)
    y_t_u_one = tf.nn.softmax(z_out)

    # Probability of using copy mechanism for decoding step t
    # TODO: I need to decide between vsize vs attn_score size
    with tf.variable_scope("Copy_mechanism", reuse=tf.AUTO_REUSE):
        W_u = tf.get_variable('W_u',
                              shape=[attn_conc_size, attn_score_size],
                              initializer=xavier_init)
        b_u = tf.get_variable("b_u",
                              shape=[attn_score_size],
                              initializer=zeros_init)

    # Equation 11
    z_u = tf.nn.xw_plus_b(attn_conc, W_u, b_u)
    u_t_one = tf.nn.sigmoid(z_u)

    # Toggle pointer mechanism
    # TODO: This can be simplified into 1 step when we decide row dims
    if use_pointer:
        # Final probability distribution for output token y_t (Equation 12)
        # TODO: Test whether I should be doing this in the TensorFlow API
        y_t = tf.add(u_t_one * y_t_u_one, (1 - u_t_one) * y_t_u_zero)
    else:
        y_t = y_t_u_zero

    return y_t


def test_tokenization(args):
    ''' test tokenization function
    python token_generation_and_pointer.py test1
    :param args:
    :return:
    '''
    attn_score = np.random.randn(batch_size, decoder_t)
    attn_score = tf.convert_to_tensor(attn_score, np.float32)

    dec_hidden_state = np.random.randn(batch_size, decoder_hidden_size)
    dec_hidden_state = tf.convert_to_tensor(dec_hidden_state, np.float32)

    enc_context = np.random.randn(batch_size, encoder_hidden_size)
    enc_context = tf.convert_to_tensor(enc_context, np.float32)

    dec_context = np.random.randn(batch_size, decoder_hidden_size)
    dec_context = tf.convert_to_tensor(dec_context, np.float32)

    generated = tokenization(attn_score, dec_hidden_state,
                             enc_context, dec_context, vsize)

    with tf.Session() as sess:
        v = sess.run(generated)
        print(v.shape)


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
