import tensorflow as tf

from args import args

class QMixer(tf.keras.Model):
    def __init__(self):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(tf.reduce_prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = tf.keras.layers.Dense(self.embed_dim * self.n_agents)
            self.hyper_w_final = tf.keras.layers.Dense(self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = args.hypernet_embed
            self.hyper_w_1 = tf.keras.Sequential([
                tf.keras.layers.Dense(hypernet_embed),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(self.embed_dim * self.n_agents)
            ])
            self.hyper_w_final = tf.keras.Sequential([
                tf.keras.layers.Dense(hypernet_embed),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(self.embed_dim)
            ])
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting the number of hypernet layers.")

        self.hyper_b_1 = tf.keras.layers.Dense(self.embed_dim)
        self.V = tf.keras.Sequential([
            tf.keras.layers.Dense(self.embed_dim),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1)
        ])

    def call(self, agent_qs, states):
        bs = agent_qs.shape[0]
        states = tf.reshape(states, (-1, self.state_dim))
        agent_qs = tf.expand_dims(agent_qs, axis=1)
        
        # First layer
        w1 = tf.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = tf.reshape(w1, (-1, self.n_agents, self.embed_dim))
        b1 = tf.reshape(b1, (-1, 1, self.embed_dim))
        hidden = tf.nn.elu(tf.linalg.matmul(agent_qs, w1) + b1)

        # Second layer
        w_final = tf.abs(self.hyper_w_final(states))
        w_final = tf.reshape(w_final, (-1, self.embed_dim, 1))

        # State-dependent bias
        v = self.V(states)
        v = tf.expand_dims(v, axis=1)
        
        # Compute final output
        y = tf.linalg.matmul(hidden, w_final) + v

        # Reshape and return
        q_tot = tf.reshape(y, (bs, -1, 1))
        return q_tot