# Statistical Methods in Artificial Intelligence

Assignment 5

2022101104

## KDE

Dataset creation:

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/fc0cacc9-69d1-4081-83f6-a69f4d2e30e5.png)

Fitting KDE Model:

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/image.png)

Fitting GMM Models:

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/image%201.png)

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/image%202.png)

```python
Log-Likelihood for GMM with 2 Components: -2.6382
Log-Likelihood for GMM with 3 Components: -2.5905
Log-Likelihood for GMM with 4 Components: -2.5473
Log-Likelihood for GMM with 5 Components: -2.5318
```

## HMM

Dataset:

![mfcc_0.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/df24e48a-31d5-4a22-a52f-a0a212c3b29c.png)

![mfcc_1.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/0dc4f3e1-cd1d-4c4d-acb4-c64fffe43bc3.png)

![mfcc_2.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/beb145d7-c8ba-4a98-a9d1-2bee6dc16c09.png)

![mfcc_3.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/149a7120-9adb-4885-b5e4-24ac17fc0b9d.png)

![mfcc_4.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/7131bfa1-5226-4292-ad77-759d79445fc0.png)

![mfcc_5.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/c555d4d1-f47b-4fb5-a884-bde9cf73a2b5.png)

![mfcc_6.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/fe59307e-ef96-4516-9f17-b0aac6295b2e.png)

![mfcc_7.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/fe4358aa-fcc0-4ca4-83e0-3b5eacccdc9d.png)

![mfcc_8.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/4c0bc51f-cca1-4f22-8fda-d0cba04e3617.png)

![mfcc_9.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/9b466165-fec0-4a3a-aa95-b0077e660c4b.png)

- **MFCC Coefficients**: The heatmap visualizes the Mel-Frequency Cepstral Coefficients (MFCCs), a widely used feature extraction technique for audio signals. Each row represents a frame of audio, and each column corresponds to an individual MFCC. These coefficients are effective in capturing the spectral envelope, which relates to the perceived sound quality. Generally, using 13 MFCCs is sufficient to encapsulate the key features of the audio.
- **Color Intensity**: The intensity of the colors in the heatmap indicates the magnitude of the MFCC values at specific frames and frequencies. Warmer colors (yellow) denote higher values, while cooler colors (such as blue and purple) signify lower values.

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/image%203.png)

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/image%204.png)

- MFCCs capture the temporal changes in the frequency spectrum of the audio, with each row in the MFCC matrix representing a short-time segment of the sound.
- These patterns vary across different spoken digits, enabling the model to distinguish between them, as illustrated in the figure above.
- When the same digits are spoken by different individuals, their MFCC features and spectrograms show notable similarities. Three examples of such digits are displayed below, highlighting the consistent patterns.

### Why are HMMs well-suited for this task?

- **Sequential Data Handling**: Audio signals are sequential by nature, with each frame depending on the previous one. HMMs are designed to model such temporal dependencies, making them an effective choice for capturing the transitions between different sound states in speech.
- **Adaptability to Variability**: People speak at varying speeds and pitches, which can affect the audio. HMMs are capable of accounting for such variability by probabilistically modeling the sequence of sound states, adapting to different speaking styles.
- **Handling Variable-Length Sequences**: Since audio recordings can have different lengths, with varying numbers of MFCC frames, HMMs are a good fit as they can process sequences of different durations without issue.
- **Modeling Hidden States**: The time-varying nature of the audio's spectral envelope suggests it can be modeled as a series of hidden states. These states could correspond to phonemes or groups of phonemes, which form the structure of a spoken digit.
- **Capturing State Transitions**: HMMs can learn the transition probabilities between hidden states, effectively modeling how certain MFCC patterns are likely to follow others in the sequence of sounds that make up a digit.
- **Emission Probabilities for Discrimination**: HMMs use emission probabilities to link hidden states with observed features. The patterns seen in the MFCC heatmaps suggest that emission probabilities can be leveraged to distinguish between different spoken digits, enabling the model to accurately classify them.

Test Set Performance:

```python
0.89
```

Own audio clips:

```
0.24
```

Comparison:

- **Noise or Recording Quality**: Own recordings may include background noise, echo, or variations in microphone quality that differ from the dataset used to train the model.
- **Accent or Pronunciation**: Differences in how I pronounce the digits compared to the dataset could affect recognition accuracy.
- **Overfitting to Train Data**: The model might be overfitted to the provided train set, making it less effective at generalizing to new inputs like own recordings.

## RNN

Dataset:

```python
Seqs:
[[0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
[0, 0, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0, 1, 1],
[0, 0, 0, 1, 1, 1, 0, 1, 0],
[0, 0, 1, 0, 0, 0, 0]]
Labels:
[8, 1, 4, 4, 1]

```

The complete dataset has 100k sequences. This is split into train, test and val in the ratio 0.8,0.1 and 0.1.

Architecture:

```python

class BitCounterRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.3):
        super(BitCounterRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)  # Add feature dimension
        output, hidden = self.rnn(x)
        last_output = output[:, -1, :]  # Get the last output state
        # Apply batch normalization
        # last_output = self.batch_norm(last_output)
        count_output = self.fc(last_output).squeeze(-1)
        return count_output
# Model Hyperparameters

input_size = 1
hidden_size = 64
num_layers = 2
dropout = 0.3
num_epochs=20
```

Training:

```python
Training on cuda.
Epoch 1/20, Train Loss: 1.9849, Val Loss: 1.3452
Epoch 2/20, Train Loss: 1.0931, Val Loss: 0.8941
Epoch 3/20, Train Loss: 0.8083, Val Loss: 0.6656
Epoch 4/20, Train Loss: 0.6201, Val Loss: 0.5005
Epoch 5/20, Train Loss: 0.4960, Val Loss: 0.3733
Epoch 6/20, Train Loss: 0.4215, Val Loss: 0.3285
Epoch 7/20, Train Loss: 0.3800, Val Loss: 0.2997
Epoch 8/20, Train Loss: 0.3476, Val Loss: 0.2367
Epoch 9/20, Train Loss: 0.3255, Val Loss: 0.2118
Epoch 10/20, Train Loss: 0.3018, Val Loss: 0.2024
Epoch 11/20, Train Loss: 0.2860, Val Loss: 0.1713
Epoch 12/20, Train Loss: 0.2736, Val Loss: 0.1617
Epoch 13/20, Train Loss: 0.2584, Val Loss: 0.1435
Epoch 14/20, Train Loss: 0.2481, Val Loss: 0.1407
Epoch 15/20, Train Loss: 0.2314, Val Loss: 0.1260
Epoch 16/20, Train Loss: 0.2248, Val Loss: 0.1173
Epoch 17/20, Train Loss: 0.2143, Val Loss: 0.1209
Epoch 18/20, Train Loss: 0.2073, Val Loss: 0.1210
Epoch 19/20, Train Loss: 0.1993, Val Loss: 0.0970
Epoch 20/20, Train Loss: 0.1914, Val Loss: 0.1143
Train Accuracy: 0.9453
Validation Accuracy: 0.9705
```

Test:

```python
Test Loss (MAE): 0.1184
Random Baseline MAE: 4.1239
```

```python
Sequence: 001101100111
True Count of '1's: 7
Predicted Count of '1's: 6.84
--------------------------------------------------
Sequence: 100011101100
True Count of '1's: 6
Predicted Count of '1's: 6.20
--------------------------------------------------
Sequence: 101
True Count of '1's: 2
Predicted Count of '1's: 2.09
--------------------------------------------------
Sequence: 01
True Count of '1's: 1
Predicted Count of '1's: 0.98
--------------------------------------------------
Sequence: 11
True Count of '1's: 2
Predicted Count of '1's: 2.09
--------------------------------------------------
```

Generalisation:

```python
MAE for 1 : 0.05493014238097451
MAE for 2 : 0.053913251920179886
MAE for 3 : 0.06430068063108545
MAE for 4 : 0.059000526603899504
MAE for 5 : 0.059906311937280604
MAE for 6 : 0.06986619808055737
MAE for 7 : 0.07084470476422991
MAE for 8 : 0.08721187285014562
MAE for 9 : 0.10142284631729126
MAE for 10 : 0.08714775244394939
MAE for 11 : 0.09344311356544495
MAE for 12 : 0.11417567729949951
MAE for 13 : 0.1451510488986969
MAE for 14 : 0.1795724974738227
MAE for 15 : 0.25226230091518825
MAE for 16 : 0.2949311539933488
MAE for 17 : 0.32004747023949254
MAE for 18 : 0.45303959846496583
MAE for 19 : 0.5373978018760681
MAE for 20 : 1.3515126137506395
MAE for 21 : 1.1247583495246039
MAE for 22 : 1.832932676587786
MAE for 23 : 1.9167664127965127
MAE for 24 : 2.3714290800548734
MAE for 25 : 2.195332497358322
MAE for 26 : 2.695166219364513
MAE for 27 : 3.351618528366089
MAE for 28 : 3.951860179071841
MAE for 29 : 4.010487965175083
MAE for 30 : 4.924534221028173
MAE for 31 : 4.777349933501212
MAE for 32 : 5.854572334289551
```

![image.png](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/image%205.png)

### Observations:

1. **Good Generalization Within the Training Range (1–16):**

   The model performs well (low MAE) for sequence lengths within the training range. This indicates that the model has effectively learned to process these lengths and generalizes well to data it was trained on.

2. **Degradation Beyond the Training Range (17–32):**

   There is a noticeable and significant increase in MAE for sequence lengths beyond 16. The error grows progressively larger as the sequence length increases. This suggests that the model struggles to generalize to OOD sequence lengths, particularly as they diverge further from the training range.

3. **Sharp Growth in Error Beyond ~20:**

   The error remains somewhat stable after 16 but starts increasing rapidly around sequence length 20 and becomes very large by 30–32. This indicates a compounding inability to handle longer sequences, likely due to limitations in the model's architecture (e.g., memory or representational capacity).

**Possible reasons for poor OOD performance:**
Capacity Constraints: The RNN may lack sufficient capacity to handle longer dependencies.
Overfitting to the Training Distribution: The model might rely heavily on patterns specific to sequence lengths it has seen, rather than learning generalizable features.
Vanishing Gradients: For longer sequences, the RNN's ability to retain information may degrade due to vanishing gradient issues.

## OCR

### Dataset

The dataset consists of 100k words printed on white background of size $256 \times 64$. The data is the image and the labels are the corresponding words. An example of 2 word-label pairs are shown below:

![image.jpeg](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/sample_data.jpeg)

### Architecture

```python
 class OCRModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(OCRModel, self).__init__()
        self.encoder = CNNEncoder()
        self.decoder = RNNDecoder(input_size=256, hidden_size=hidden_size, output_size=vocab_size)

    def forward(self, images, target=None, target_length=None, teacher_forcing_ratio=0.5):
        batch_size = images.size(0)
        encoder_outputs = self.encoder(images)

        if self.training and target is not None:
            outputs, _ = self.decoder(encoder_outputs, target, hidden=None)
            return outputs
        else:
            # Inference mode
            max_length = 25  # Maximum sequence length
            outputs = torch.zeros(batch_size, max_length, self.decoder.output_size).to(images.device)

            # Initialize with <sos>
            decoder_input = torch.ones(batch_size, 1).long().to(images.device) * 27  # <sos> token
            hidden = None

            for t in range(max_length):
                out, hidden = self.decoder(encoder_outputs, decoder_input, hidden)
                outputs[:, t:t+1] = out
                # Get the most likely next character
                _, topi = out.max(2)
                decoder_input = topi.squeeze(-1).unsqueeze(1)

                # Stop if all sequences have generated <eos>
                if (decoder_input == 28).all():  # 28 is <eos> token
                    break

            return outputs
```

```python
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(0.2)
        )

    def forward(self, x):
        x = self.features(x)
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(batch_size, -1, 256)
        return x

class RNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.RNN(hidden_size + input_size, hidden_size, n_layers,
                         batch_first=True, dropout=0.2 if n_layers > 1 else 0)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, target, hidden=None):
        batch_size = encoder_outputs.size(0)
        max_length = target.size(1)

        outputs = torch.zeros(batch_size, max_length, self.output_size).to(encoder_outputs.device)

        embedded = self.embedding(target)

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_size).to(encoder_outputs.device)

        # For each time step
        for t in range(max_length):
            context = encoder_outputs[:, t:t+1, :]

            rnn_input = torch.cat((embedded[:, t:t+1, :], context), dim=2)

            output, hidden = self.rnn(rnn_input, hidden)

            output = self.out(output.squeeze(1))
            outputs[:, t] = output

        return outputs, hidden
```

### 4.2.3 Task 3: Training

```python
Epoch 1/5:
Train Loss: 0.6465, Train Acc: 0.9194
Val Loss: 0.0005, Val Acc: 0.9090
Epoch 2/5:
Train Loss: 0.0067, Train Acc: 1.0000
Val Loss: 0.0002, Val Acc: 0.9999
Epoch 3/5:
Train Loss: 0.0035, Train Acc: 1.0000
Val Loss: 0.0001, Val Acc: 1.0000
Epoch 4/5:
Train Loss: 0.0027, Train Acc: 1.0000
Val Loss: 0.0000, Val Acc: 1.0000
Epoch 5/5:
Train Loss: 0.0024, Train Acc: 1.0000
Val Loss: 0.0000, Val Acc: 1.0000
```

```python
Test Accuracy: 0.9998
```

```python
Random Baseline Accuracy: 0.0125
```

![image.jpeg](Statistical%20Methods%20in%20Artificial%20Intelligence%2013f17aaec13280028444eb0f6904cd46/predicted.jpeg)
