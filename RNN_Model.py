import tensorflow as tf

class RNNModel(tf.keras.Model):
    def __init__(self, n_actions, hidden_units=32):
        super(RNNModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='tanh')
        self.lstm = tf.keras.layers.LSTM(hidden_units, return_state=True, return_sequences=False)
        self.linear = tf.keras.layers.Dense(n_actions, activation=None)

    def __call__(self, inputs, initial_state=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        lstm_output, state_h, state_c = self.lstm(x, initial_state=initial_state)
        x = self.linear(lstm_output)
        return x, [state_h, state_c]