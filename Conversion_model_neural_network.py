'''
Attempt of application of Pytorch as the model to predict frequencies and volumes based on the emotion vectors.
'''
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(46)

'''
Setup feature matrix X and response Y
'''

# temporary mapping
emotion_chord_mapping = {
    'Admiration': [60, 64, 67, 71],  # Cmaj7 (C Major 7th chord)
    'Adoration': [57, 60, 64, 67],  # Amin7 (A minor 7th chord)
    'Aesthetic Appreciation': [62, 66, 69, 73],  # Dmaj7 (D Major 7th chord)
    'Amusement': [59, 62, 65, 69],  # Bmin7 (B minor 7th chord)
    'Anger': [61, 64, 68, 71],  # C#min7 (C# minor 7th chord)
    'Annoyance': [63, 67, 70, 74],  # Emaj7 (E Major 7th chord)
    'Anxiety': [58, 62, 65, 69],  # Bmin7 (B minor 7th chord)
    'Awe': [64, 67, 71, 74],  # Emin7 (E minor 7th chord)
    'Awkwardness': [59, 62, 66, 69],  # Bmin7 (B minor 7th chord)
    'Boredom': [60, 63, 67, 70],  # Cmin7 (C minor 7th chord)
    'Calmness': [60, 64, 67, 71],  # Cmaj7 (C Major 7th chord)
    'Concentration': [62, 66, 69, 73],  # Dmaj7 (D Major 7th chord)
    'Confusion': [59, 62, 66, 70],  # Bmin7 (B minor 7th chord)
    'Contemplation': [61, 64, 68, 71],  # C#min7 (C# minor 7th chord)
    'Contempt': [62, 65, 69, 72],  # Dmin7 (D minor 7th chord)
    'Contentment': [60, 64, 67, 71],  # Cmaj7 (C Major 7th chord)
    'Craving': [57, 60, 64, 67],  # Amin7 (A minor 7th chord)
    'Determination': [62, 65, 69, 72],  # Dmin7 (D minor 7th chord)
    'Disappointment': [60, 63, 67, 70],  # Cmin7 (C minor 7th chord)
    'Disapproval': [62, 65, 69, 72],  # Dmin7 (D minor 7th chord)
    'Disgust': [61, 64, 68, 71],  # C#min7 (C# minor 7th chord)
    'Distress': [61, 64, 68, 71],  # C#min7 (C# minor 7th chord)
    'Doubt': [62, 65, 69, 72],  # Dmin7 (D minor 7th chord)
    'Ecstasy': [64, 67, 71, 74],  # Emin7 (E minor 7th chord)
    'Embarrassment': [61, 64, 68, 71],  # C#min7 (C# minor 7th chord)
    'Empathic Pain': [62, 65, 69, 72],  # Dmin7 (D minor 7th chord)
    'Enthusiasm': [64, 67, 71, 74],  # Emin7 (E minor 7th chord)
    'Entrancement': [62, 65, 69, 72],  # Dmin7 (D minor 7th chord)
    'Envy': [61, 64, 68, 71],  # C#min7 (C# minor 7th chord)
    'Excitement': [64, 67, 71, 74],  # Emin7 (E minor 7th chord)
    'Fear': [61, 64, 67, 71],  # C#min7 (C# minor 7th chord)
    'Gratitude': [60, 63, 67, 70],  # Cmin7 (C minor 7th chord)
    'Guilt': [61, 64, 68, 71],  # C#min7 (C# minor 7th chord)
    'Horror': [61, 64, 67, 71],  # C#min7 (C# minor 7th chord)
    'Interest': [64, 67, 71, 74],  # Emin7 (E minor 7th chord)
    'Joy': [62, 66, 69, 73],  # Dmaj7 (D Major 7th chord)
    'Love': [57, 60, 64, 67],  # Amin7 (A minor 7th chord)
    'Nostalgia': [60, 63, 67, 70],  # Cmin7 (C minor 7th chord)
    'Pain': [60, 63, 67, 70],  # Cmin7 (C minor 7th chord)
    'Pride': [62, 65, 69, 72],  # Dmin7 (D minor 7th chord)
    'Realization': [61, 64, 68, 71],  # C#min7 (C# minor 7th chord)
    'Relief': [61, 64, 67, 71],  # C#min7 (C# minor 7th chord)
    'Romance': [62, 65, 69, 72],  # Dmin7 (D minor 7th chord)
    'Sadness': [60, 63, 67, 70],  # Cmin7 (C minor 7th chord)
    'Sarcasm': [62, 65, 68, 71],  # Dmin7 (D minor 7th chord)
    'Satisfaction': [62, 65, 69, 72],  # Dmin7 (D minor 7th chord)
    'Desire': [57, 60, 64, 67],  # Amin7 (A minor 7th chord)
    'Shame': [60, 63, 67, 70],  # Cmin7 (C minor 7th chord)
    'Surprise (negative)': [62, 65, 69, 72],  # Dmin7 (D minor 7th chord)
    'Surprise (positive)': [62, 65, 69, 72],  # Dmin7 (D minor 7th chord)
    'Sympathy': [62, 65, 69, 72],  # Dmin7 (D minor 7th chord)
    'Tiredness': [59, 62, 65, 69],  # Bmin7 (B minor 7th chord)
    'Triumph': [61, 64, 67, 71]  # C#min7 (C# minor 7th chord)
}

emotions = list(emotion_chord_mapping.keys())

# conversion from note (MIDI) to frequency
def note_to_frequency(note_number):
    A4_frequency = 440  # Frequency of A4 note in Hz
    return A4_frequency * (2 ** ((note_number - 69) / 12.0))

# Normalize a sample (x0 -> (X_i, y_i))
def normalize(x0):
    # take a high dimensional vector and normalize the highest score to be the absolute volume
    maxscore = max(x0, key=lambda x: x['score'])
    s = 0
    X_scores = []
    for tmp in x0:
        for key, item in tmp.items():
            if key == "score":
                s += item
                # record the scores only as the training features
                X_scores.append(item)
    y = emotion_chord_mapping[maxscore["name"]][:]
    y.append(maxscore["score"] / s + maxscore["score"] / 100)
    return y, X_scores

# generate more samples (x0s) using emotiosn and uniform distribution
def generate_samples(num_samples=100, num_emotions=len(emotions)):
    samples = []
    for _ in range(num_samples):
        sample = []
        for _ in range(num_emotions):
            emotion = random.choice(emotions)
            score = random.uniform(0, 1)
            sample.append({'name': emotion, 'score': score})
        samples.append(sample)
    return samples


X = [[{'name': 'Admiration', 'score': 0.06379243731498718}, {'name': 'Adoration', 'score': 0.07222934812307358}, {'name': 'Aesthetic Appreciation', 'score': 0.02808445133268833}, {'name': 'Amusement', 'score': 0.027589013800024986}, {'name': 'Anger', 'score': 0.0120259253308177}, {'name': 'Annoyance', 'score': 0.025653120130300522}, {'name': 'Anxiety', 'score': 0.004923961125314236}, {'name': 'Awe', 'score': 0.025031352415680885}, {'name': 'Awkwardness', 'score': 0.061385106295347214}, {'name': 'Boredom', 'score': 0.05333968624472618}, {'name': 'Calmness', 'score': 0.135557159781456}, {'name': 'Concentration', 'score': 0.010018930770456791}, {'name': 'Confusion', 'score': 0.09115109592676163}, {'name': 'Contemplation', 'score': 0.020809845998883247}, {'name': 'Contempt', 'score': 0.030744805932044983}, {'name': 'Contentment', 'score': 0.060751479119062424}, {'name': 'Craving', 'score': 0.008604105561971664}, {'name': 'Determination', 'score': 0.010685051791369915}, {'name': 'Disappointment', 'score': 0.03298037126660347}, {'name': 'Disapproval', 'score': 0.022201286628842354}, {'name': 'Disgust', 'score': 0.010582842864096165}, {'name': 'Distress', 'score': 0.011887000873684883}, {'name': 'Doubt', 'score': 0.01733436994254589}, {'name': 'Ecstasy', 'score': 0.014498891308903694}, {'name': 'Embarrassment', 'score': 0.01016779150813818}, {'name': 'Empathic Pain', 'score': 0.020176580175757408}, {'name': 'Enthusiasm', 'score': 0.02454438991844654}, {'name': 'Entrancement', 'score': 0.031809959560632706}, {'name': 'Envy', 'score': 0.0078100417740643024}, {'name': 'Excitement', 'score': 0.01086737122386694}, {'name': 'Fear', 'score': 0.002841347362846136}, {'name': 'Gratitude', 'score': 0.012725913897156715}, {'name': 'Guilt', 'score': 0.005438251420855522}, {'name': 'Horror', 'score': 0.0024618145544081926}, {'name': 'Interest', 'score': 0.06195246800780296}, {'name': 'Joy', 'score': 0.056471534073352814}, {'name': 'Love', 'score': 0.14727146923542023}, {'name': 'Nostalgia', 'score': 0.01013017725199461}, {'name': 'Pain', 'score': 0.020555714145302773}, {'name': 'Pride', 'score': 0.01391206681728363}, {'name': 'Realization', 'score': 0.021947279572486877}, {'name': 'Relief', 'score': 0.006897658109664917}, {'name': 'Romance', 'score': 0.09956834465265274}, {'name': 'Sadness', 'score': 0.031556904315948486}, {'name': 'Sarcasm', 'score': 0.02911987341940403}, {'name': 'Satisfaction', 'score': 0.03389870375394821}, {'name': 'Desire', 'score': 0.08503230661153793}, {'name': 'Shame', 'score': 0.019161522388458252}, {'name': 'Surprise (negative)', 'score': 0.047956954687833786}, {'name': 'Surprise (positive)', 'score': 0.030542362481355667}, {'name': 'Sympathy', 'score': 0.03246130049228668}, {'name': 'Tiredness', 'score': 0.03606246039271355}, {'name': 'Triumph', 'score': 0.01235896535217762}],
[{'name': 'Admiration', 'score': 0.04051987826824188}, {'name': 'Adoration', 'score': 0.009848205372691154}, {'name': 'Aesthetic Appreciation', 'score': 0.07604563981294632}, {'name': 'Amusement', 'score': 0.021876158192753792}, {'name': 'Anger', 'score': 0.002392953261733055}, {'name': 'Annoyance', 'score': 0.020589718595147133}, {'name': 'Anxiety', 'score': 0.0038202449213713408}, {'name': 'Awe', 'score': 0.03062158264219761}, {'name': 'Awkwardness', 'score': 0.015123735181987286}, {'name': 'Boredom', 'score': 0.10465148836374283}, {'name': 'Calmness', 'score': 0.14533281326293945}, {'name': 'Concentration', 'score': 0.05680052563548088}, {'name': 'Confusion', 'score': 0.14072860777378082}, {'name': 'Contemplation', 'score': 0.08013628423213959}, {'name': 'Contempt', 'score': 0.019430125132203102}, {'name': 'Contentment', 'score': 0.06073099002242088}, {'name': 'Craving', 'score': 0.006501526106148958}, {'name': 'Determination', 'score': 0.010847826488316059}, {'name': 'Disappointment', 'score': 0.01760883629322052}, {'name': 'Disapproval', 'score': 0.009595056995749474}, {'name': 'Disgust', 'score': 0.003568764077499509}, {'name': 'Distress', 'score': 0.004913835320621729}, {'name': 'Doubt', 'score': 0.020732788369059563}, {'name': 'Ecstasy', 'score': 0.0070318495854735374}, {'name': 'Embarrassment', 'score': 0.002762148156762123}, {'name': 'Empathic Pain', 'score': 0.004658285528421402}, {'name': 'Enthusiasm', 'score': 0.020970847457647324}, {'name': 'Entrancement', 'score': 0.029176874086260796}, {'name': 'Envy', 'score': 0.009052899666130543}, {'name': 'Excitement', 'score': 0.011401181109249592}, {'name': 'Fear', 'score': 0.0018732709577307105}, {'name': 'Gratitude', 'score': 0.007437469437718391}, {'name': 'Guilt', 'score': 0.0009437649860046804}, {'name': 'Horror', 'score': 0.0012521253665909171}, {'name': 'Interest', 'score': 0.10246434807777405}, {'name': 'Joy', 'score': 0.016914378851652145}, {'name': 'Love', 'score': 0.0034126662649214268}, {'name': 'Nostalgia', 'score': 0.008389955386519432}, {'name': 'Pain', 'score': 0.004401583690196276}, {'name': 'Pride', 'score': 0.012128319591283798}, {'name': 'Realization', 'score': 0.076319120824337}, {'name': 'Relief', 'score': 0.01443533692508936}, {'name': 'Romance', 'score': 0.0022930167615413666}, {'name': 'Sadness', 'score': 0.006305092945694923}, {'name': 'Sarcasm', 'score': 0.018501555547118187}, {'name': 'Satisfaction', 'score': 0.07013726979494095}, {'name': 'Desire', 'score': 0.004992782603949308}, {'name': 'Shame', 'score': 0.004370391834527254}, {'name': 'Surprise (negative)', 'score': 0.028048569336533546}, {'name': 'Surprise (positive)', 'score': 0.05226348340511322}, {'name': 'Sympathy', 'score': 0.007420244161039591}, {'name': 'Tiredness', 'score': 0.02038390189409256}, {'name': 'Triumph', 'score': 0.02609831653535366}]]

fake_X = generate_samples(num_samples=10000)
X.extend(fake_X)
X_scores = [[] for _ in range(len(X))]
Y = [[] for _ in range(len(X))]

for i in range(len(X)):
    y, x = normalize(X[i])
    X_scores[i] = x
    Y[i] = y
    
# Standardize 4th dimension of Y
Y = np.array(Y)
mean_val = np.mean(Y[:, 4])
std_dev = np.std(Y[:, 4])
min_val = mean_val - 2 * std_dev
max_val = mean_val + 2 * std_dev
for i in range(len(X)):
    if Y[i, 4] >= min_val and Y[i, 4] <= max_val:
        Y[i, 4] = (Y[i, 4] - min_val) / (max_val - min_val)
    elif Y[i, 4] < min_val:
        Y[i, 4] = min_val
    else: 
        Y[i, 4] = max_val



'''
Actual training of model and predictions
'''


X_train, X_test, y_train, y_test = train_test_split(X_scores, Y, test_size=0.2, random_state=46)

# PCA
pca = PCA(n_components=10)  
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# Define the neural network model class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)  # Output layer with output_size units
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer does not have an activation function for regression tasks
        # Round the first 4 dimensions to integers
        x[:, :4] = torch.round(x[:, :4])
        return x

# Initialize the model, loss function, and optimizer
input_size = 10  # Number of PCA components
output_size = 5  # 5-dimensional output with the first 4 dimensions rounded to integers
neural_network_model = NeuralNetwork(input_size, output_size)
criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
optimizer = optim.Adam(neural_network_model.parameters(), lr=0.01)  # Adam optimizer

# Convert data to PyTorch tensors (assuming X_train_pca and y_train are already PyTorch tensors)
X_train_pca = torch.tensor(X_train_pca, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Training the neural network model
num_epochs = 50
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(X_train_pca), batch_size):
        inputs = X_train_pca[i:i+batch_size]
        targets = y_train[i:i+batch_size]
        
        # Forward pass
        outputs = neural_network_model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print progress (optional)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Convert test data to PyTorch tensors (assuming X_test_pca and y_test are already PyTorch tensors)
X_test_pca = torch.tensor(X_test_pca, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Evaluate the model on test data
neural_network_model.eval()
with torch.no_grad():
    predictions = neural_network_model(X_test_pca)
    nn_mse = criterion(predictions, y_test)
    print("Neural Network Mean Squared Error:", nn_mse.item())


# Convert MIDI notes to frequencies
# for i in range(4):
#     y_test[:, i] = note_to_frequency(y_test[:, i])
#     predictions[:, i] = note_to_frequency(predictions[:, i])


for i in range(5):
    plt.figure(figsize=(8, 6))
    y_test_i = [y_test[_][i] for _ in range(len(y_test))]
    predictions_i = [predictions[_][i] for _ in range(len(y_test))]
    
    # Add small random noise to the points for better visibility
    # noise = np.random.normal(0, 0.01, len(y_test_i))
    # y_test_i += noise
    # predictions_i += noise
    
    plt.scatter(y_test_i, predictions_i, color='black', label='Actual vs. Predicted', s = 5)
    plt.plot(y_test_i, y_test_i, color='red', linewidth=0.5, label='Perfect Prediction (y=x)')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs. Predicted Values for Dimension {i+1}')
    plt.legend()
    plt.grid(True)
    plt.show()