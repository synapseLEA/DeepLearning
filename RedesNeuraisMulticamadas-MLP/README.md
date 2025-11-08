
# Redes Neurais Multicamadas (MLP)

A **Rede Neural Multicamadas (MLP)**, ou Perceptron Multicamadas, é a forma mais fundamental do Deep Learning. Ideal para dados tabulares (colunas e linhas) onde a ordem das features não importa.

## 📝 Conceito Chave

* É uma rede **feedforward** (alimentação direta), onde a informação flui apenas da entrada para a saída.
* **Aprendizado (Backpropagation):** O erro da previsão é propagado de volta para ajustar os **pesos** dos neurônios (via otimizadores como **Adam**) e reduzir a **perda (*loss*)**.

### 🧱 Estrutura e Comandos Keras Essenciais

Esta seção detalha os principais comandos do Keras, usando o MLP como exemplo.

#### A. Camada `Dense()`

A camada **`Dense`** (totalmente conectada) é a base do MLP.

| Parâmetro | Descrição | Possibilidades Chave | Exemplo |
| :--- | :--- | :--- | :--- |
| **`units`** | **Número de Neurônios** na camada. | Qualquer número inteiro (Ex: 32, 64). | `units=64` |
| **`activation`** | **Função de Ativação** do neurônio. | **`'relu'`** (ocultas), **`'sigmoid'`** (saída binária), **`'softmax'`** (saída multi-classe). | `activation='relu'` |
| **`input_shape`** | **Dimensão da Entrada.** **Só é necessário na PRIMEIRA camada.** | Tupla contendo o número de *features* (Ex: `(10,)`). | `input_shape=(10,)` |
| **`kernel_regularizer`** | Aplica **regularização L2** ou **L1** para mitigar o *overfitting*. | `'l1'`, `'l2'`. | `kernel_regularizer='l2'` |

**Exemplo**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
# PRIMEIRA CAMADA (define input_shape)
model.add(Dense(units=64, activation='relu', input_shape=(10,)))
# Adiciona Regularização (Dropout desliga neurônios aleatoriamente)
model.add(Dropout(0.2)) 
# CAMADA OCULTA
model.add(Dense(units=32, activation='relu'))
# CAMADA DE SAÍDA
model.add(Dense(units=1, activation='sigmoid'))
```

#### Configuração do Treinamento: `model.compile()`

Define como o modelo será treinado.

| Parâmetro | Descrição | Possibilidades Chave | Exemplo |
| :--- | :--- | :--- | :--- |
| **`optimizer`** | **Algoritmo de Otimização**. | **`'adam'`**, `'sgd'`, `'rmsprop'`. | `optimizer='adam'` |
| **`loss`** | **Função de Perda** a ser minimizada. | **`'binary_crossentropy'`** (classificação binária), `'mse'` (regressão), `'categorical_crossentropy'` (multi-classe). | `loss='binary_crossentropy'` |
| **`metrics`** | **Métricas** de avaliação. | `['accuracy']`, `['mae']` (regressão), `['Precision', 'Recall']`. | `metrics=['accuracy', 'Precision']` |

**Exemplo**
```python
from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy',
              metrics=['accuracy', 'Precision'])
```

#### Execução do Treinamento: `model.fit()`

Inicia o processo de aprendizado.

| Parâmetro | Descrição | Possibilidades Chave | Exemplo |
| :--- | :--- | :--- | :--- |
| **`x`, `y`** | Dados de *features* (`x`) e *labels* (`y`) de treinamento (NumPy arrays). | `x=X_train, y=y_train` |
| **`epochs`** | **Número de épocas** (voltas completas no *dataset*). | Inteiros (Ex: 10, 50). | `epochs=50` |
| **`batch_size`** | **Número de amostras** processadas antes de uma atualização de peso. | Potências de 2 (Ex: 32, 64). | `batch_size=64` |
| **`validation_split`** | Fração dos dados de treino a ser usada como validação (0 a 1). | `validation_split=0.15` | `validation_split=0.15` |
| **`callbacks`** | Funções chamadas durante o treino. | **`EarlyStopping`** (parada antecipada), **`ModelCheckpoint`** (salva o melhor modelo). | `callbacks=[es_callback]` |

```python
from tensorflow.keras.callbacks import EarlyStopping

# Define o Callback para Parada Antecipada
es_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Treinamento
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=64,
                    validation_split=0.15,
                    callbacks=[es_callback])
```

---
