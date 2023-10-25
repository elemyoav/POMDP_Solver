import tensorflow as tf

class RNNModel(tf.keras.Model):
    def __init__(self, n_actions, hidden_units=32):
        super(RNNModel, self).__init__()
        self.lstm = tf.keras.layers.LSTM(hidden_units, return_state=True, return_sequences=False)
        self.linear = tf.keras.layers.Dense(n_actions, activation='softmax')

    def __call__(self, inputs, initial_state=None):
        lstm_output, state_h, state_c = self.lstm(inputs, initial_state=initial_state)
        x = self.linear(lstm_output)
        return x, [state_h, state_c]

        