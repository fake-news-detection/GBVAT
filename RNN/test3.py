import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential

nltk_data = np.load('GBVAT/data/npy/nltk/data.npy', allow_pickle=True)
nltk_label = np.load('GBVAT/data/npy/nltk/label.npy', allow_pickle=True)
spacy_data = np.load('GBVAT/data/npy/spacy/data.npy', allow_pickle=True)
spacy_label = np.load('GBVAT/data/npy/spacy/label.npy', allow_pickle=True)

TEST_SPLIT = 0.2
EMBEDDING_DIM = 300
MAX_DOC_LENGTH = 500

# NLTK Variant
num_test_samples = int(TEST_SPLIT * nltk_data.shape[0])
x_train = nltk_data[:-num_test_samples]
y_train = nltk_label[:-num_test_samples]
x_test = nltk_data[-num_test_samples:]
y_test = nltk_label[-num_test_samples:]

num_val_samples = int(TEST_SPLIT * x_train.shape[0])
x_train = x_train[:-num_val_samples]
y_train = y_train[:-num_val_samples]
x_val = x_train[-num_val_samples:]
y_val = y_train[-num_val_samples:]

# Develop DNN
model = Sequential()
model.add(Embedding(input_dim=20568,
                    output_dim=EMBEDDING_DIM,
                    input_length=MAX_DOC_LENGTH,
                    trainable=False))

model.add(Bidirectional(LSTM(units=64)))
model.add(Dense(32, activation='relu'))
model.summary()


def make_unit_norm(r):
    return r / (tf.reshape(tf.sqrt(tf.reduce_sum(tf.pow(r, 2.0), axis=1)), [-1, 1]) + 1e-16)


def compute_kld(p, q):
    p = tf.nn.softmax(p)
    q = tf.nn.softmax(q)
    return tf.reduce_sum(p * (tf.math.log(p + 1e-16) - tf.math.log(q + 1e-16)), axis=1)


# Perturbations
clean_input = Input(shape=(MAX_DOC_LENGTH,))  # original input
r = tf.random.uniform(shape=tf.shape(clean_input))
r = make_unit_norm(r)  # perturbation with constraint of L2 norm
adv_input = clean_input + 10 * r  # perturbed input

p_logit = model(clean_input)  # clean logit
p = Dense(2, activation='sigmoid')(p_logit)  # Should this be 2 ???

q_logit = model(adv_input)  # perturbed logit
# q = Dense(2, activation='sigmoid')(q_logit)

kl = tf.reduce_mean(compute_kld(p_logit, q_logit))

tf.compat.v1.disable_eager_execution()


# grad_kl = tf.gradients( kl , [r])[0]


def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]


grad_kl = _compute_gradients(kl, [r])[0]  # JM

r_vadv = tf.stop_gradient(grad_kl)
r_vadv = make_unit_norm(r_vadv) / len(x_train)  # checking dividing by lenght of labelled data

p_logit_no_gradient = tf.stop_gradient(p_logit)
p_logit_r_vadv = model(clean_input + r_vadv)
vat_loss = tf.reduce_mean(compute_kld(p_logit_no_gradient, p_logit_r_adv))

#############################################
# network = Sequential()
# network.add( Dense(100 ,activation='relu' ,  input_shape=(2,)))
# network.add( Dense(2   ))

# model_input = Input((2,))
# p_logit = network( model_input )
# p = Activation('softmax')( p_logit )
# r = tf.random_normal(shape=tf.shape( model_input ))
# r = make_unit_norm( r )
# p_logit_r = network( model_input + 10*r  )

kl = tf.reduce_mean(compute_kld(p_logit, p_logit_r))
grad_kl = tf.gradients(kl, [r])[0]
r_vadv = tf.stop_gradient(grad_kl)
r_vadv = make_unit_norm(r_vadv) / 3.0

p_logit_no_gradient = tf.stop_gradient(p_logit)
p_logit_r_adv = network(model_input + r_vadv)
vat_loss = tf.reduce_mean(compute_kld(p_logit_no_gradient, p_logit_r_adv))

model_vat = Model(model_input, p)
model_vat.add_loss(vat_loss)

model_vat.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

model_vat.metrics_names.append('vat_loss')
model_vat.metrics_tensors.append(vat_loss)

# Train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # only compilation
history = model.fit(x_train, y_train, epochs=2, batch_size=160, validation_data=(x_val, y_val))
# evaluating model
score, acc = model.evaluate(x_test, y_test)
print('Test score:', score)
print('Test accuracy:', acc)
