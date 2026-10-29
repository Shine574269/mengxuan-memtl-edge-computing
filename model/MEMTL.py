# file: model/MEMTL.py

import tensorflow as tf
from tensorflow.keras import Model
from utils.loss_function import custom_loss


class MEMTL(Model):
    """
    An extensive work for the article
    "Computation Offloading in Multi-Access Edge Computing: A Multi-Task Learning Approach" 2021.
    Multi-head Ensemble Multi-Task Learning.

    This version is enhanced with self-contained Adaptive Gradient Clipping for improved training stability.
    """

    def __init__(self, mu_num, head_num):
        """
        Initialize a MEMTL model with specified mu_num and head_num.
        """
        assert head_num > 1
        super(MEMTL, self).__init__()

        self.mu_num = mu_num
        self.head_num = head_num


        self.clip_norm = 1.0  # Common default value for gradient clipping constraint that can be tuned

        self.d1 = tf.keras.layers.Dense(60, input_dim=mu_num * 6, activation='relu', trainable=True,
                                        kernel_initializer='he_uniform', bias_initializer='he_uniform')
        self.d2 = tf.keras.layers.Dense(10 * mu_num, activation='relu', trainable=True,
                                        kernel_initializer='random_uniform', bias_initializer='random_uniform')
        heads_1_list = []
        for i in range(self.head_num):
            heads_1_list.append(tf.keras.layers.Dense(40, activation='relu', trainable=True))
        self.heads_1 = heads_1_list

        heads_2_list = []
        for i in range(self.head_num):
            heads_2_list.append(tf.keras.layers.Dense(self.mu_num, activation='sigmoid', trainable=True))
        self.heads_2 = heads_2_list


    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x1s = []
        for layer in self.heads_1:
            x1s.append(layer(x))
        outputs = []
        for i in range(self.head_num):
            outputs.append(self.heads_2[i](x1s[i]))
        return outputs

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            if isinstance(y_pred, list):
                # Completely bypass Keras compiled_loss structure mappers to prevent 'No gradients'
                # and simply apply custom_loss perfectly to all predicting heads
                loss = tf.constant(0.0)
                for yp in y_pred:
                    loss += tf.reduce_mean(custom_loss(y, yp))
                if self.losses:
                    loss += tf.add_n(self.losses)
            else:
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Calculate gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Robustly filter out None gradients (variables that were dynamically untouched or frozen in cache)
        valid_grads_and_vars = [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]
        
        if valid_grads_and_vars:
            valid_grads = [g for g, v in valid_grads_and_vars]
            valid_vars = [v for g, v in valid_grads_and_vars]
            clipped_grads, _ = tf.clip_by_global_norm(valid_grads, self.clip_norm)
            self.optimizer.apply_gradients(zip(clipped_grads, valid_vars))

        # Update metrics (Reconstruct y_target to match y_pred list structure if necessary)
        if isinstance(y_pred, list) and not isinstance(y, (list, tuple)):
            y_target = [y] * len(y_pred)
        else:
            y_target = y
        self.compiled_metrics.update_state(y_target, y_pred)
        
        # Return metrics results as a dictionary
        return {m.name: m.result() for m in self.metrics}

    def set_backbone(self, trainable):
        """
        Set the backbone to participate in training.
        """
        self.d1.trainable = trainable
        self.d2.trainable = trainable

    def set_trainable(self, i):
        """
        Set the i th prediction head to participate in training.
        """
        # Allow i=-1 to freeze all heads
        if not (-1 <= i < self.head_num):
            raise ValueError(f"Index {i} is out of bounds for head_num {self.head_num}")

        for j in range(self.head_num):
            is_target = (j == i)
            self.heads_1[j].trainable = is_target
            self.heads_2[j].trainable = is_target

    def add_head(self):
        """
        Add a new prediction head. Uses tracker unlock/lock to bypass Keras 3.x state freeze.
        """
        new_head1 = tf.keras.layers.Dense(40, activation='relu', trainable=True)
        new_head2 = tf.keras.layers.Dense(self.mu_num, activation='sigmoid', trainable=True)
        
        # Build layers with dummy input to ensure variables are created
        dummy_input = tf.keras.Input(shape=(self.mu_num * 6,))
        _ = self.d1(dummy_input)
        dummy_backbone_out = self.d2(_)
        dummy_head1_out = new_head1(dummy_backbone_out)
        _ = new_head2(dummy_head1_out)

        # Temporarily unlock the Keras 3.x state tracker to allow dynamic layer addition
        self._tracker.unlock()
        self.heads_1.append(new_head1)
        self.heads_2.append(new_head2)
        self._tracker.lock()
        
        self.head_num += 1

    def copy_add_head(self, i):
        """
        Copy the i th prediction head and add the copy.
        """
        # Ensure layers are built before setting weights
        new_head1 = tf.keras.layers.Dense(40, activation='relu', trainable=False)
        new_head2 = tf.keras.layers.Dense(self.mu_num, activation='sigmoid', trainable=False)

        # Process a dummy input tensor to build the layers
        dummy_input = tf.keras.Input(shape=(self.mu_num * 6,))
        _ = self.d1(dummy_input)
        dummy_backbone_out = self.d2(_)
        dummy_head1_out = new_head1(dummy_backbone_out)
        _ = new_head2(dummy_head1_out)

        new_head1.set_weights(self.heads_1[i].get_weights())
        new_head2.set_weights(self.heads_2[i].get_weights())

        # Temporarily unlock the Keras 3.x state tracker
        self._tracker.unlock()
        self.heads_1.append(new_head1)
        self.heads_2.append(new_head2)
        self._tracker.lock()
        self.head_num += 1

    def reset_heads(self):
        """
        Reinitialize all prediction heads.
        """
        self._tracker.unlock()
        for i in range(self.head_num):
            self.heads_1[i] = tf.keras.layers.Dense(40, activation='relu')
            self.heads_2[i] = tf.keras.layers.Dense(self.mu_num, activation='sigmoid')
        self._tracker.lock()

    def reset_head(self, i):
        """
        Reinitialize the specified prediction head.
        """
        assert i < self.head_num
        self._tracker.unlock()
        self.heads_1[i] = tf.keras.layers.Dense(40, activation='relu')
        self.heads_2[i] = tf.keras.layers.Dense(self.mu_num, activation='sigmoid')
        self._tracker.lock()

    def get_config(self):
        return {"mu_num": self.mu_num, "head_num": self.head_num}
