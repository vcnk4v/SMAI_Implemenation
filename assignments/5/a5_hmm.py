import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Step 1: Load the dataset
def load_dataset(data_path):
    audio_data = []
    labels = []
    for filename in os.listdir(data_path):
        if filename.endswith(".wav"):
            digit = int(filename.split("_")[0])

            file_path = os.path.join(data_path, filename)
            # Load audio file
            y, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024)

            audio_data.append(mfcc.T)  # Transpose to have time steps on rows
            labels.append(digit)
    return audio_data, labels


def visualize_mfcc(mfcc, digit, speaker=None):
    plt.figure(figsize=(10, 4))
    sns.heatmap(mfcc, cmap="viridis", cbar=True)
    if speaker:
        plt.title(f"MFCC Heatmap for Digit {digit} (Speaker: {speaker})")
    else:
        plt.title(f"MFCC Heatmap for Digit {digit}")
    plt.xlabel("MFCC Coefficients")
    plt.ylabel("Time Frames")
    # plt.show()
    if speaker:
        plt.savefig(f"figures/mfcc_{speaker}_{digit}.png")
    else:
        plt.savefig(f"figures/mfcc_{digit}.png")


def train_hmm_models(X_train, y_train, n_components=5):
    models = {}
    # Train a separate HMM for each digit
    for digit in range(10):
        digit_data = [X_train[i] for i in range(len(y_train)) if y_train[i] == digit]

        # Convert list of sequences into a 2D array suitable for hmmlearn
        # Flatten each sequence to make it compatible with HMM input
        digit_data = [seq.reshape(-1, 1) for seq in digit_data]

        # Create and train the HMM model
        model = hmm.GaussianHMM(
            n_components=n_components, covariance_type="diag", n_iter=100
        )
        model.fit(
            np.concatenate(digit_data)
        )  # Train on all sequences of the current digit

        # Save the trained model
        models[digit] = model

    return models


def predict_digit(test_file, digit_hmms):
    mfcc_features = extract_mfcc(test_file)
    max_prob = float("-inf")
    best_digit = None

    for digit, model in digit_hmms.items():
        try:
            log_prob = model.score(mfcc_features)
            if log_prob > max_prob:
                max_prob = log_prob
                best_digit = digit
        except:
            pass  # Handle exceptions if model fails to score

    return best_digit


def extract_mfcc(audio_path, n_mfcc=13):
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Extract the MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    return mfccs.T


# Function to prepare data by extracting features from all files in a directory
def prepare_data(data_dir):
    digit_features = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".wav"):
            digit = int(filename.split("_")[0])  # Extract digit label from filename
            file_path = os.path.join(data_dir, filename)
            mfcc_features = extract_mfcc(file_path)
            if digit not in digit_features:
                digit_features[digit] = []
            digit_features[digit].append(mfcc_features)
    return digit_features


def main():
    train_path = "../../data/external/fsdd/train-recordings/"
    test_path = "../../data/external/fsdd/test-recordings/"
    test_path2 = "../data/external/fsdd/own_voice/"
    train_audio, train_labels = load_dataset(train_path)
    test_audio, test_labels = load_dataset(test_path)

    # sample_index = train_labels.index(0)
    # visualize_mfcc(train_audio[sample_index].T, digit=0)

    for i in range(10):
        visualize_mfcc(train_audio[i].T, digit=i)

    digit_hmms = {}
    train_digit_features = prepare_data(train_path)

    n_components = 5  # Adjust based on performance
    covariance_type = "diag"
    n_iter = 100

    # Train an HMM for each digit
    for digit, features_list in train_digit_features.items():
        # Stack all feature arrays for the current digit into one array for fitting
        X = np.vstack(features_list)
        lengths = [len(features) for features in features_list]

        # Initialize and train the HMM model
        model = hmm.GaussianHMM(
            n_components=n_components, covariance_type=covariance_type, n_iter=n_iter
        )
        model.fit(X, lengths)

        # Store the trained model
        digit_hmms[digit] = model

    print("Training complete for all digits.")

    correct_predictions = 0
    total_predictions = 0
    sample_count = 0
    for filename in os.listdir(test_path):
        if filename.endswith(".wav"):
            true_digit = int(filename.split("_")[0])
            test_file_path = os.path.join(test_path, filename)

            predicted_digit = predict_digit(test_file_path, digit_hmms)
            if predicted_digit == true_digit:
                correct_predictions += 1
            total_predictions += 1

        if sample_count < 5:
            print(
                f"Sample {sample_count + 1}: Actual: {true_digit}, Predicted: {predicted_digit}"
            )
            sample_count += 1

    accuracy = correct_predictions / total_predictions * 100
    print(f"Accuracy on test set: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
