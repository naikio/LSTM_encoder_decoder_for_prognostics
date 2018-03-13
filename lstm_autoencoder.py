import tensorflow as tf


class LSTMAutoencoder:

    """
    Basic LSTM autoencoder
    
    source:     https://arxiv.org/pdf/1608.06154.pdf
    """

    def __init__(self, hidden_n, inputs, learning_rate=0.001):

        self.n_components = inputs[0].get_shape().as_list()[1]  # n_components of last dense layer (from PCA components)
        self.hidden_n = hidden_n
        self.inputs = inputs
        self.learning_rate = learning_rate

        self._inference = None
        self._loss = None
        self._training = None
        self._summaries = None   # TensorBoard Summaries

        self._inference = self.inference
        self._loss = self.loss
        self._training = self.training
        self._summaries = self.summaries

    @property
    def inference(self):
        if self._inference is None:
            with tf.variable_scope('encoder'):
                lstm_units_enc = [tf.nn.rnn_cell.LSTMCell(self.hidden_n, use_peepholes=True) for _ in range(self.hidden_n)]
                multi_lstm_enc = tf.nn.rnn_cell.MultiRNNCell(lstm_units_enc)
                # DROPOUT non-recurrent connections of LSTM cells
                # encoder_cell = tf.nn.rnn_cell.DropoutWrapper(encoder_cell, output_keep_prob=0.85)
                encoder_output, encoder_state = tf.nn.dynamic_rnn(multi_lstm_enc, self.inputs, dtype=tf.float32)

            with tf.variable_scope('decoder'):
                lstm_units_dec = [tf.nn.rnn_cell.LSTMCell(self.hidden_n, use_peepholes=True) for _ in range(self.hidden_n)]
                multi_lstm_dec = tf.nn.rnn_cell.MultiRNNCell(lstm_units_dec)
                # DROPOUT non-recurrent connections of LSTM cells
                # decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob=0.85)
                decoder_output, decoder_state = tf.nn.dynamic_rnn(multi_lstm_dec, inputs=encoder_output,
                                                                  initial_state=encoder_state)

            self._inference = tf.layers.dense(inputs=decoder_output, units=self.n_components)

        return self._inference

    @property
    def loss(self):
        if self._loss is None:
            with tf.variable_scope('loss'):
                # Loss is reconstruction error
                self._loss = tf.norm((self.inputs - self._inference), ord=2)

        return self._loss

    @property
    def training(self):
        if self._training is None:
            with tf.variable_scope('training'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                global_step = tf.Variable(0, name='global_step', trainable=False)
                self._training = optimizer.minimize(self._loss, global_step=global_step)
        return self._training

    @property
    def summaries(self):
        if self._summaries is None:
            # Add TensorBoard Summaries
            # --- scalar summaries
            tf.summary.scalar('loss', self._loss)

            # --- merge all summaries and initialize the summary writer
            self._summaries = tf.summary.merge_all()
        return self._summaries
