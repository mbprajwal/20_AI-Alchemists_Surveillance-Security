import pandas as pd
import numpy as np
import streamlit as st
from collections import Counter
from itertools import groupby
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


# Your model and data preprocessing code
df = pd.read_csv(r"C:\Users\Aditi\Desktop\hackathon\extracted_data.csv")

# Define the mapping of labels to numbers
label_mapping = {
    'Simple': 1,
    'Vigenere': 2,
    'Column': 3,
    'Playfair': 4,
    'Hill': 5
}

# Apply the mapping to the 'label' column
df["1"] = df["1"].map(label_mapping)
df = df.sample(frac=1)
y = df['1']
x = df.iloc[:, 3:]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Initialize the Gradient Boosting model
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# Cipher detection function
def detect_cipher_type(encoded_text):
    # Feature extraction logic
    english_freq_list = [4, 19, 0, 14, 8, 13, 18, 17, 7, 11, 3, 20, 2, 12, 6, 5, 24, 15, 22, 1, 21, 10, 23, 9, 25, 16]
    english_freq_list = [x + 1 for x in english_freq_list]

    log_table = np.array([  # Log Table for digraph frequency score
        4,7,8,7,4,6,7,5,7,3,6,8,7,9,3,7,3,9,8,9,6,7,6,5,7,4,
        7,4,2,0,8,1,1,1,6,3,0,7,2,1,7,1,0,6,5,3,7,1,2,0,6,0,
        8,2,5,2,7,3,2,8,7,2,7,6,2,1,8,2,2,6,4,7,6,1,3,0,4,0,
        7,6,5,6,8,6,5,5,8,4,3,6,6,5,7,5,3,6,7,7,6,5,6,0,6,2,
        9,7,8,8,8,7,6,6,7,4,5,8,7,9,7,7,5,9,9,8,5,7,7,6,7,3,
        7,4,5,3,7,6,4,4,7,2,2,6,5,3,8,4,0,7,5,7,6,2,4,0,5,0,
        7,5,5,4,7,5,5,7,7,3,2,6,5,5,7,5,2,7,6,6,6,3,5,0,5,1,
        8,5,4,4,9,4,3,4,8,3,1,5,5,4,8,4,2,6,5,7,6,2,5,0,5,0,
        7,5,8,7,7,7,7,4,4,2,5,8,7,9,7,6,4,7,8,8,4,7,3,5,0,5,
        5,0,0,0,4,0,0,0,3,0,0,0,0,0,5,0,0,0,0,0,6,0,0,0,0,0,
        5,4,3,2,7,4,2,4,6,2,2,4,3,6,5,3,1,3,6,5,3,0,4,0,5,0,
        8,5,5,7,8,5,4,4,8,2,5,8,5,4,8,5,2,4,6,6,6,5,5,0,7,1,
        8,6,4,3,8,4,2,4,7,1,0,4,6,4,7,6,1,3,6,5,6,1,4,0,6,0,
        8,6,7,8,8,6,9,6,8,4,6,6,5,6,8,5,3,5,8,9,6,5,6,3,6,2,
        6,6,7,7,6,8,6,6,6,3,6,7,8,9,7,7,3,9,7,8,9,6,8,4,5,3,
        7,3,3,3,7,3,2,6,7,2,1,7,3,2,7,6,0,7,6,6,6,0,3,0,4,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,0,
        8,6,6,7,9,6,6,5,8,3,6,6,6,6,8,6,3,6,8,8,6,5,6,0,7,1,
        8,6,7,6,8,6,5,7,8,4,6,6,6,6,8,7,4,5,8,9,7,4,7,0,6,2,
        8,6,6,5,8,6,5,9,8,3,3,6,6,5,9,6,2,7,8,8,7,4,7,0,7,2,
        6,6,7,6,6,4,6,4,6,2,3,7,7,8,5,6,0,8,8,8,3,3,4,3,4,3,
        6,1,0,0,8,0,0,0,7,0,0,0,0,0,5,0,0,0,1,0,2,1,0,0,3,0,
        7,3,3,4,7,3,2,8,7,2,2,4,4,6,7,3,0,5,5,5,2,1,4,0,3,1,
        4,1,4,2,4,2,0,3,5,1,0,1,1,0,3,5,0,1,2,5,2,0,2,2,3,0,
        6,6,6,6,6,6,5,5,6,3,3,5,6,5,8,6,3,5,7,6,4,3,6,2,4,2,
        4,0,0,0,5,0,0,0,3,0,0,2,0,0,3,0,0,0,1,0,2,0,0,0,4,4]).reshape(26,26)
    logdi = pd.DataFrame(log_table, index=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), columns=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

    # Vector of frequency of letters in English language
    rel_freq = {
        'A': 0.08167, 'B': 0.01492, 'C': 0.02782, 'D': 0.04253, 'E': 0.12702,
        'F': 0.02228, 'G': 0.02015, 'H': 0.06094, 'I': 0.06966, 'J': 0.00153,
        'K': 0.00772, 'L': 0.04025, 'M': 0.02406, 'N': 0.06749, 'O': 0.07507,
        'P': 0.01929, 'Q': 0.00095, 'R': 0.05987, 'S': 0.06327, 'T': 0.09056,
        'U': 0.07294, 'V': 0.01178, 'W': 0.01722, 'X': 0.00150, 'Y': 0.01974,
        'Z': 0.00074
    }

    # Count unique characters in a string
    def char_ct_unique(ciphertext):
        return len(set(ciphertext))

    # Measure similarity of frequency distribution to uniform distribution (Index of Coincidence)
    def get_IC(ciphertext):
        f = Counter(ciphertext)
        L = len(ciphertext)
        return sum(v * (v - 1) for v in f.values()) / (L * (L - 1)) if L>2 else 0


    # Get letters from every period interval starting at start
    def gather_letters(ciphertext, start, period):
        return ''.join([ciphertext[i] for i in range(start, len(ciphertext), period)])


    # Log Digraph Frequency Score (Placeholder: Replace with actual scoring function)
    def get_LDI_Score(ciphertext, pos, logdi):
        return logdi[ciphertext[pos]][ciphertext[pos + 1]]


    # Reverse Log Digraph Frequency Score (Placeholder: Replace with actual scoring function)
    def get_RDI_Score(ciphertext, pos, logdi):
        return logdi[ciphertext[pos + 1]][ciphertext[pos]]

    # Main feature extraction
    def extract_features(ciphertexts, english_freq_list, logdi):
        features = []
        for ct in ciphertexts:
            L = len(ct)
            nuc = char_ct_unique(ct)
            ic = get_IC(ct)
            mic = max(
                np.mean([get_IC(gather_letters(ct, start, period)) for start in range(period)])
                for period in range(1, 16)
            )
            dic = sum(f * (f - 1) for f in Counter([ct[i:i+2] for i in range(L - 1)]).values()) / ((L - 1) * (L - 2))
            edi = sum(f * (f - 1) for f in Counter([ct[i:i+2] for i in range(0, L - 1, 2)]).values()) / ((L // 2) * (L // 2 - 1))
            ldi = np.mean([get_LDI_Score(ct, pos, logdi) for pos in range(L - 1)])
            rdi = np.mean([get_RDI_Score(ct, pos, logdi) for pos in range(0, L - 1, 2)])

            features.append({
                "NUC": nuc,
                "IC": ic,
                "MIC": mic,
                "DIC": dic,
                "EDI": edi,
                "LDI": ldi,
                "RDI": rdi
            })

        return pd.DataFrame(features)
    FEATURES = extract_features([encoded_text.upper()], english_freq_list, logdi)


    prediction = gb.predict(FEATURES)
    # Original label mapping
    label_mapping = {
        'simple substitution': 1,
        'vigenere': 2,
        'column transposition': 3,
        'playfair': 4,
        'hill cipher': 5
    }

    # Reverse the mapping: mapping numbers back to labels
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    label = reverse_label_mapping[prediction[0]]

    return label


# Vigenère decryption function
def vigenere_decode(ciphertext, key):
    # Create a mapping for letters to indices
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    letter_to_index = {letter: idx for idx, letter in enumerate(alphabet)}
    index_to_letter = {idx: letter for idx, letter in enumerate(alphabet)}

    # Normalize ciphertext and key to uppercase
    ciphertext = ciphertext.upper()
    key = key.upper()

    # Decode each letter
    plaintext = []
    key_length = len(key)
    for i, char in enumerate(ciphertext):
        if char in letter_to_index:
            # Get the numerical position of ciphertext char and key char
            cipher_idx = letter_to_index[char]
            key_idx = letter_to_index[key[i % key_length]]
            # Reverse the Vigenère formula to decode
            plain_idx = (cipher_idx - key_idx) % 26
            plaintext.append(index_to_letter[plain_idx])
        else:
            # Keep non-alphabet characters as-is
            plaintext.append(char)

    return ''.join(plaintext)


# Set page config as the first Streamlit command
st.set_page_config(page_title="Cipher Detection Tool", page_icon="🔐", layout="centered")


# Streamlit interface
st.sidebar.title('Navigation')
st.sidebar.markdown("""
- Enter the encoded text
- Detect cipher type
- If Vigenère cipher is detected, provide a decryption key
""")

# Main content
st.title('Cipher Detection Tool')
encoded_text = st.text_area("Enter the encoded text here:")

if encoded_text:
    # Logic to process the encoded text and detect cipher type
    detected_cipher = detect_cipher_type(encoded_text)
    st.write(f"Detected cipher type: {detected_cipher}")

    if detected_cipher == 'vigenere':
        key = st.text_input("Enter the decryption key:")
        if key:
            decrypted_text = vigenere_decode(encoded_text, key)
            st.write(f"Decrypted text: {decrypted_text}")
