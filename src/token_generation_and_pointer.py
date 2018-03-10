'''
token_generation_and_pointer.py
'''
import tensorflow as tf
import numpy as np
import argparse

# below function will take parameters (self, decoder_outputs, hps, vsize, *extra_args):
def tokenization(params):
    '''
    Token generation and pointer (2.3, p.3). u_t = 1 if we
    want to pay attention to or copy the inputs and u_t = 0 if we do not. The
    tokenization mechanism allows our model to learn the representation of
    words it had not seen in training by copying the representation of an
    unknown word from its input (p.5 of article). p(u_t) = p_gen from Abi.

    Args: Dictionary params
        u_t:
        decoder_output: decoder state tensor at timestep t
        input_context: input context vector at timestep t
        decoder_context: decoder context vector at timestep t
        vocab_size: vocabulary size scalar
        temoral_attention_score: tensor of attenion scores from equation (4)
        use_pointer: boolean, True = pointer mechanism, False = no pointer

    Returns:
        dict(final_distrubution, vocab_score): token probability distribution final_distrubution
    '''
    temoral_attention_score = params['temoral_attention_scores']
    decoder_output = params['decoder_outputs']
    input_context = params['input_contexts']
    decoder_context = params['decoder_contexts']
    vocab_size = params['vocab_size']
    use_pointer = False
    if 'use_pointer' in params:
        use_pointer = params['use_pointer']

    # Variables
    attentions = tf.concat(values=[decoder_output, input_context, decoder_context], axis=1)

    # Hyperparameters
    # TODO: I am not sure whether row dim is vize or alpha_e_ti size
    attn_conc_size = attentions.get_shape().as_list()[1]
    attn_score_size = params["max_enc_steps"]

    # Initializations
    xavier_init = tf.contrib.layers.xavier_initializer()
    zeros_init = tf.zeros_initializer()

    # Tokenization with the pointer mechanism
    # Equation 10; alpha_e_ti is taken from Equation 4
    copy_distrubution = temoral_attention_score

    # Tokenization with the token-generation softmax layer
    # TODO: I need to decide between vsize vs attn_score size
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
    # TODO: check if vocab_scores is the real values we want
    vocab_scores = tf.nn.xw_plus_b(attentions, W_out, b_out)
    vocab_distribution = tf.nn.softmax(vocab_scores)

    # Probability of using copy mechanism for decoding step t
    # TODO: I need to decide between vocab_size vs attn_score size
    '''
    with tf.variable_scope("Copy_mechanism", reuse=tf.AUTO_REUSE):
        W_u = tf.get_variable('W_u',
                              shape=[attn_conc_size, attn_score_size],
                              initializer=xavier_init)
        b_u = tf.get_variable("b_u",
                              shape=[attn_score_size],
                              initializer=zeros_init)

    # Equation 11
    z_u = tf.nn.xw_plus_b(attentions, W_u, b_u)
    pointer = tf.nn.sigmoid(z_u)
    '''
    # Toggle pointer mechanism Equation 12
    # TODO: This can be simplified into 1 step when we decide row dims
    #if use_pointer:
        # Final probability distribution for output token y_t (Equation 12)
        # TODO: Test whether I should be doing this in the TensorFlow API
    vocab_dists = vocab_distribution #tf.add(pointer * copy_distrubution, (1 - pointer) * vocab_distribution)
    #else:
    #    vocab_dists = copy_distrubution

    return vocab_dists, vocab_scores

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
    vsize = 50000

    attn_score = np.random.randn(batch_size, decoder_t)
    attn_score = tf.convert_to_tensor(attn_score, np.float32)

    dec_hidden_state = np.random.randn(batch_size, decoder_hidden_size)
    dec_hidden_state = tf.convert_to_tensor(dec_hidden_state, np.float32)

    enc_context = np.random.randn(batch_size, encoder_hidden_size)
    enc_context = tf.convert_to_tensor(enc_context, np.float32)

    dec_context = np.random.randn(batch_size, decoder_hidden_size)
    dec_context = tf.convert_to_tensor(dec_context, np.float32)

    generated = tokenization({"temoral_attention_scores": attn_score, "decoder_outputs": dec_hidden_state,
                              "input_contexts": enc_context, "decoder_contexts": dec_context, "vocab_size": vsize})

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ret = sess.run(generated)
        print(ret["vocab_dists"].shape)


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
