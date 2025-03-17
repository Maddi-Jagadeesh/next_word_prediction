# next_word_prediction
# Next Word Prediction using LSTM

This project implements a "Next Word Prediction" model using an "LSTM-based Neural Network". The model is trained on a text corpus and predicts the next word in a given sequence.

---Features---

- Uses **LSTM layers** for sequence modeling.
- **Word embeddings** using Keras Embedding layer.
- **Dropout layers** to prevent overfitting.
- **Softmax activation** for multi-class word prediction.
- Trained on a **large text corpus** to learn sentence structures.



---Model Architecture---

```python
model = Sequential()
model.add(Embedding(input_dim=len(T.word_index)+1, output_dim=100, input_length=26 - 1))
model.build((None,26-1)) # Initialize embedding layer weights
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(len(T.word_index)+1, activation='softmax'))
```


- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Epochs: 100 (can be adjusted based on performance)


---Improvements & Future Work---

- Use **pre-trained word embeddings** like GloVe or Word2Vec.
- Implement **beam search** for better predictions.
- Train on **larger datasets** for improved accuracy.


📜 License
This project is open-source and available under the **MIT License**.

---
Author: Maddi Jagadeesh 
GitHub: [Maddi-Jagadeesh](https://github.com/Maddi-Jagadeesh)


