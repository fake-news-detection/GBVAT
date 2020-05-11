

# # Spacy Variant
# num_test_samples = int(TEST_SPLIT * spacy_data.shape[0])
# x_train = spacy_data[:-num_test_samples]
# y_train = spacy_label[:-num_test_samples]
# x_test = spacy_data[-num_test_samples:]
# y_test = spacy_label[-num_test_samples:]
#
# # Develop DNN
# model = Sequential()
# model.add(Embedding(input_dim=11925,
#                     output_dim=EMBEDDING_DIM,
#                     input_length=MAX_DOC_LENGTH,
#                     trainable=False))
#
# model.add(LSTM(units=256))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()
#
# # Train the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # only compilation
# history = model.fit(x_train, y_train, epochs=3, batch_size=40, validation_split=0.2)
# # evaluating model
# score, acc = model.evaluate(x_test, y_test, batch_size=10)
# print('Test score:', score)
# print('Test accuracy:', acc)