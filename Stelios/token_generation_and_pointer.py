'''
token_generation_and_pointer.py
'''
import tensorflow as tf
import argparse

def tokenization(u_t, attn_score, h_d_t, c_e_t, c_d_t):
    '''
    Implementation of token generation and pointer (2.3, p.3). u_t = 1 if we
    want to pay attention to or copy the inputs and u_t = 0 if we do not. The
    tokenization mechanism allows our model to learn the representation of
    words it had not seen in training by copying the representation of an
    unknown word from its input (p.5 of article). p(u_t) = p_gen from Abi.

    Args:
        u_t: binary value, 1 for pointer mechanism, 0 for token generation
        attn_score: tensor of attenion scores from equation (4)
        h_d_t: decoder state tensor at timestep t
        c_e_t: input context vector at timestep t
        c_d_t: decoder context vector at timestep t
        vsize: vocabulary size

    Returns:
        y_t_distn: probability distribution for the output token y_t
    '''
    # Variables
    # TODO: these are tensors in which case this may not work as expected
    attn_conc = tf.concat(values=[h_d_t, c_e_t, c_d_t], axis=0)

    # Hyperparameters
    # TODO: Change this to appropriate dims when the other parts are known
    attn_conc_size = max(attn_conc.get_shape().as_list())
    attn_score_size = attn_score.get_shape().as_list()

    # Initializations
    xavier_init = tf.contrib.layers.xavier_initializer()
    zeros_init = tf.zeros_initializer()

    # Tokenization with the pointer mechanism
    if u_t:
        # Equation 10
        y_t_u_zero = attn_score

    # Tokenization with the token-generation softmax layer
    else:
        with tf.variable_scope("Tokenization"):
            W_out = tf.get_variable('W_out',
                                    shape=[attn_conc_size, attn_score_size],
                                    initializer=xavier_init)
            b_out = tf.get_variable("b_r",
                                    shape=[attn_score_size],
                                    initializer=zeros_init)

        # Reuse variables across timesteps
        tf.get_variable_scope().reuse_variables()

        # Equation 9
        z_out = tf.nn.xw_plus_b(attn_conc, W_out, b_out)
        y_t_u_one = tf.nn.softmax(z_out)

    # Probability of using copy mechanism for decoding step t
    with tf.variable_scope("Copy_mechanism"):
        W_u = tf.get_variable('W_u',
                              shape=[attn_conc_size, attn_score_size],
                              initializer=xavier_init)
        b_u = tf.get_variable("b_u",
                              shape=[attn_score_size],
                              initializer=zeros_init)

    # Equation 11
    z_u = tf.nn.xw_plus_b(attn_conc, W_u, b_u)
    u_t_one = tf.nn.sigmoid(z_u)

    # Final probability distribution for output token y_t (Equation 12)
    # TODO: Test whether I should be doing this in the TensorFlow API
    y_t = tf.add(u_t_one * y_t_u_one, (1 - u_t_one) * y_t_u_zero)

    return y_t

def test_tokenization(args):
    ''' test tokenization function
    python token_generation_and_pointer.py test1
    :param args:
    :return:
    '''
    # To-do: use np.random.randn to generate data with correct dimension to pass into tokenization function
    with tf.Session() as sess:
        tf.logging.info("put test code here.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Test the tokenization function to matching dimensions of parameters')
  subparsers = parser.add_subparsers()

  command_parser = subparsers.add_parser('test1', help='Test the tokenization function')
  command_parser.set_defaults(func=test_tokenization)

  ARGS = parser.parse_args()
  if not hasattr(ARGS, 'func'):
      parser.print_help()
  else:
      ARGS.func(ARGS)