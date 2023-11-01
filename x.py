import tensorflow as tf
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, Dense

def create_attention_network(input_shape, num_heads, key_dim, your_output_units, num_layers=4):
    # Input layer
    inputs = Input(shape=input_shape)
    
    x = inputs
    for _ in range(num_layers):
    # Multi-Head Attention
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x, x, use_causal_mask=True)
        
        # Normalize the attention output
        x = LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    # Global Average Pooling
    mean_output = tf.reduce_mean(x, axis=1)
    
    # Output layer
    output = Dense(units=your_output_units, activation=None)(mean_output)
    
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    return model

# Example usage
input_shape = (20, 8)
num_heads = 4
key_dim = 32
your_output_units = 125  # Adjust as needed

attention_model = create_attention_network(input_shape, num_heads, key_dim, your_output_units)

attention_model.summary()

input = tf.random.uniform(shape=(1, 20, 8))
output = attention_model(input)
input = tf.random.uniform(shape=(32, 20, 8))
output = attention_model(input)
print(output.shape)