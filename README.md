# F1 Lap-Time Prediction Model

This project trains a deep learning model to **predict Formula 1 lap times** using
telemetry sequences and contextual information such as tyre compound, driver,
season, and lap number. The model uses a combination of **LSTM sequence modeling**
and **embedding-based context features** to analyze the time-series structure of
each lap.

---

## Model Inputs

### **1. Telemetry Sequence**
- Shape: `(batch_size, seq_len, num_features)`
- Includes per-sample values such as:
  - Speed  
  - Throttle  
  - Brake  
  - Gear  
  - RPM  
- Laps have variable length, so telemetry is **padded** and fed into  
  `pack_padded_sequence` to correctly mask padding.

### **2. Context Features (Embedded)**
Each categorical field is passed through an embedding layer:

| Feature | Description | Example Values |
|--------|-------------|----------------|
| **Tyre** | Compound for that lap | 0–3 |
| **Driver** | Driver ID | 0–8 |
| **Year** | Season index | 0–2 |
| **Lap Number** | Lap index in stint | Integer |

All embeddings are concatenated with the LSTM’s output before final prediction.

---

## Model Architecture

### **1. LSTM Encoder**
- Processes the time-series telemetry sequence  
- Uses `pack_padded_sequence` so padding is ignored  
- Final hidden state represents the lap’s temporal features

### **2. Embedding Layers**
- `tyre_embedding`  
- `driver_embedding`  
- `year_embedding`  
- `lap_embedding`

These capture categorical relationships that influence lap time.

### **3. Fully Connected Regressor**
The model concatenates:
- LSTM output  
- Tyre embedding  
- Driver embedding  
- Year embedding  
- Lap embedding  

This combined vector is fed through an MLP:

```
[LSTM hidden state] + [context embeddings]
                ↓
     Linear → ReLU → Dropout
                ↓
              Linear
                ↓
         Predicted Lap Time
```

Loss function: **MSELoss** (or equivalent regression loss)  
Optimizer: **Adam**

---

## Training Progress

Example train loss progression (20 epochs):

```
Epoch  1: 0.0620  
Epoch  2: 0.0470  
Epoch  3: 0.0439  
Epoch  4: 0.0411  
Epoch  5: 0.0390  
Epoch  6: 0.0391  
Epoch  7: 0.0363  
Epoch  8: 0.0371  
Epoch  9: 0.0359  
Epoch 10: 0.0362  
Epoch 11: 0.0347  
Epoch 12: 0.0376  
Epoch 13: 0.0323  
Epoch 14: 0.0346  
Epoch 15: 0.0325  
Epoch 16: 0.0318  
Epoch 17: 0.0315  
Epoch 18: 0.0308  
Epoch 19: 0.0304  
Epoch 20: 0.0305  
```

Test Loss:
```
0.0281
```

Detected input ID ranges during training:
```
Tyre:   0 → 3
Driver: 0 → 8
Year:   0 → 2
```

---

## How Sequence Packing Works

Because laps have different numbers of telemetry samples:

1. Telemetry is padded to the maximum length in the batch  
2. A `lengths` tensor stores each lap’s true sequence length  
3. `pack_padded_sequence` removes padding so the LSTM sees only real data  
4. After LSTM processing, the final hidden state is used for prediction  

This allows the model to treat short and long laps correctly.

---

## Next Steps
- Fine Tune for other tracks
- Add weather context (rain %, temperature, wind)
- Add pit stop flags and stint age
- Improve multi-year generalization with richer driver embeddings
- Add error-based metrics like MAE and RMSE
- Predict multi-step deltas (tyre degradation modeling)

---
