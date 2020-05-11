from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer

max_features = 20000
maxlen = 200
batch_size = 32

import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Lambda
from tensorflow.keras.models import Model
import numpy as np
import datetime
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import regularizers
#this is to enable eager execution
tf.compat.v1.enable_eager_execution()

batch_size = 128
epochs=10
alpha_vat=0.5
lr=0.0005

def compute_kld(p_logit, q_logit):
    p = tf.nn.softmax(p_logit)
    q = tf.nn.softmax(q_logit)
    return tf.reduce_sum(p*(tf.math.log(p + 1e-16) - tf.math.log(q + 1e-16)), axis=1)


def make_unit_norm(x):
    return x/(tf.reshape(tf.sqrt(tf.reduce_sum(tf.pow(x, 2.0), axis=1)), [-1, 1]) + 1e-16)

tf.keras.backend.clear_session()
tf.compat.v1.disable_eager_execution()
def BiLstmModel(maxlen, vocab_size):
  inputs =tf.keras.Input(shape=(maxlen,))
  x=Embedding(vocab_size, 128, input_length=None)(inputs)
# x=GaussianNoise(0.01, input_shape=(None,))(x)
  x=Bidirectional(LSTM(128))(x)
#   x=Dense(64, activation= 'relu')(x)
  x=Dropout(0.1)(x)
  outputs =Dense(2, activation='sigmoid')(x)
  outputs =Dense(1)(outputs)
  return Model(inputs,outputs)

model_vat = BiLstmModel(maxlen, vocab_size)

model_vat.summary()

tf.compat.v1.disable_eager_execution()
tf.keras.backend.clear_session()

inputs = Input(shape=(maxlen,))
model_vat= Sequential()
model_vat.add(Embedding(vocab_size, 128, input_length=None))
p_logit = model_vat(inputs)

x=Bidirectional(LSTM(128))(p_logit)
x=Dropout(0.1)(x)
x =Dense(2, activation='sigmoid')(x)
p =Dense(1)(x)

# p = Activation('sigmoid')( p_logit )

r = tf.random.uniform(shape=tf.shape(inputs))
r = make_unit_norm(r)
p_logit_r = model_vat( inputs + 10*r  )

def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var)for var, grad in zip(var_list, grads)]
kl = tf.reduce_mean(compute_kld( p_logit , p_logit_r ))
grad_kl = tf.gradients( kl , [r])[0]
grad_kl =_compute_gradients(kl, [r])[0]

r_vadv = tf.stop_gradient(grad_kl)
r_vadv = make_unit_norm( r_vadv )/3

p_logit_no_gradient = tf.stop_gradient(p_logit)
p_logit_r_adv = model_vat( inputs  + r_vadv)
vat_loss =  tf.reduce_mean(compute_kld( p_logit_no_gradient, p_logit_r_adv ))

'''try adding here too '''
# x=Bidirectional(LSTM(128))(p_logit)
# #   x=Dense(64, activation= 'relu')(x)
# x=Dropout(0.1)(x)
# x =Dense(2, activation='sigmoid')(x)
# p =Dense(1)(x)
model_vat = Model(inputs , p )
model_vat.add_loss(vat_loss )

model_vat.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss= 'binary_crossentropy',metrics=['accuracy'])

model_vat.metrics_names.append('vat_loss')
model_vat.metrics.append(vat_loss)

keras.utils.plot_model(model_vat, show_shapes=True, show_layer_names=True )

test_scores = model_vat.evaluate(x_test_seq, y_test, batch_size=batch_size, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

#prediction
y_hat= model_vat.predict(x_test)
y_bool=(np.greater_equal(y_hat,0.51)).astype(int)
# y_bool_predicted=y_bool.flatten()
Accuracy = np.count_nonzero((np.equal(y_bool,y_test)).astype(int))/len(y_test)
print("Test Accuracy:",Accuracy)



# tokenizer=Tokenizer (num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
# full_article= np.hstack((x_train, x_test,x_unlabel))
# tokenizer.fit_on_texts(full_article)
# x_train_token=tokenizer.texts_to_sequences(x_train)
# x_test_token=tokenizer.texts_to_sequences(x_test)
# x_unlabel_token= tokenizer.texts_to_sequences(x_unlabel)
#
# x_train_seq = sequence.pad_sequences(x_train_token, maxlen=maxlen)
# x_test_seq=sequence.pad_sequences(x_test_token, maxlen=maxlen)
# x_unlabel_tar= sequence.pad_sequences(x_unlabel_token, maxlen= maxlen)
# #defining vocalbury size
# vocab_size = len(tokenizer.word_index)+1
#
# x_train= x_train_seq
# x_test= x_test_seq
# vocab_size,np.shape(x_train_seq), np.shape(y_train),np.shape(x_unlabel_tar),np.shape(x_test),np.shape(y_test)