from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import math

def note_to_frequency(note_number):
    A4_frequency = 440  # Frequency of A4 note in Hz
    return A4_frequency * (2 ** ((note_number - 69) / 12.0))




X = [[{'name': 'Admiration', 'score': 0.06379243731498718}, {'name': 'Adoration', 'score': 0.07222934812307358}, {'name': 'Aesthetic Appreciation', 'score': 0.02808445133268833}, {'name': 'Amusement', 'score': 0.027589013800024986}, {'name': 'Anger', 'score': 0.0120259253308177}, {'name': 'Annoyance', 'score': 0.025653120130300522}, {'name': 'Anxiety', 'score': 0.004923961125314236}, {'name': 'Awe', 'score': 0.025031352415680885}, {'name': 'Awkwardness', 'score': 0.061385106295347214}, {'name': 'Boredom', 'score': 0.05333968624472618}, {'name': 'Calmness', 'score': 0.135557159781456}, {'name': 'Concentration', 'score': 0.010018930770456791}, {'name': 'Confusion', 'score': 0.09115109592676163}, {'name': 'Contemplation', 'score': 0.020809845998883247}, {'name': 'Contempt', 'score': 0.030744805932044983}, {'name': 'Contentment', 'score': 0.060751479119062424}, {'name': 'Craving', 'score': 0.008604105561971664}, {'name': 'Determination', 'score': 0.010685051791369915}, {'name': 'Disappointment', 'score': 0.03298037126660347}, {'name': 'Disapproval', 'score': 0.022201286628842354}, {'name': 'Disgust', 'score': 0.010582842864096165}, {'name': 'Distress', 'score': 0.011887000873684883}, {'name': 'Doubt', 'score': 0.01733436994254589}, {'name': 'Ecstasy', 'score': 0.014498891308903694}, {'name': 'Embarrassment', 'score': 0.01016779150813818}, {'name': 'Empathic Pain', 'score': 0.020176580175757408}, {'name': 'Enthusiasm', 'score': 0.02454438991844654}, {'name': 'Entrancement', 'score': 0.031809959560632706}, {'name': 'Envy', 'score': 0.0078100417740643024}, {'name': 'Excitement', 'score': 0.01086737122386694}, {'name': 'Fear', 'score': 0.002841347362846136}, {'name': 'Gratitude', 'score': 0.012725913897156715}, {'name': 'Guilt', 'score': 0.005438251420855522}, {'name': 'Horror', 'score': 0.0024618145544081926}, {'name': 'Interest', 'score': 0.06195246800780296}, {'name': 'Joy', 'score': 0.056471534073352814}, {'name': 'Love', 'score': 0.14727146923542023}, {'name': 'Nostalgia', 'score': 0.01013017725199461}, {'name': 'Pain', 'score': 0.020555714145302773}, {'name': 'Pride', 'score': 0.01391206681728363}, {'name': 'Realization', 'score': 0.021947279572486877}, {'name': 'Relief', 'score': 0.006897658109664917}, {'name': 'Romance', 'score': 0.09956834465265274}, {'name': 'Sadness', 'score': 0.031556904315948486}, {'name': 'Sarcasm', 'score': 0.02911987341940403}, {'name': 'Satisfaction', 'score': 0.03389870375394821}, {'name': 'Desire', 'score': 0.08503230661153793}, {'name': 'Shame', 'score': 0.019161522388458252}, {'name': 'Surprise (negative)', 'score': 0.047956954687833786}, {'name': 'Surprise (positive)', 'score': 0.030542362481355667}, {'name': 'Sympathy', 'score': 0.03246130049228668}, {'name': 'Tiredness', 'score': 0.03606246039271355}, {'name': 'Triumph', 'score': 0.01235896535217762}],
[{'name': 'Admiration', 'score': 0.04051987826824188}, {'name': 'Adoration', 'score': 0.009848205372691154}, {'name': 'Aesthetic Appreciation', 'score': 0.07604563981294632}, {'name': 'Amusement', 'score': 0.021876158192753792}, {'name': 'Anger', 'score': 0.002392953261733055}, {'name': 'Annoyance', 'score': 0.020589718595147133}, {'name': 'Anxiety', 'score': 0.0038202449213713408}, {'name': 'Awe', 'score': 0.03062158264219761}, {'name': 'Awkwardness', 'score': 0.015123735181987286}, {'name': 'Boredom', 'score': 0.10465148836374283}, {'name': 'Calmness', 'score': 0.14533281326293945}, {'name': 'Concentration', 'score': 0.05680052563548088}, {'name': 'Confusion', 'score': 0.14072860777378082}, {'name': 'Contemplation', 'score': 0.08013628423213959}, {'name': 'Contempt', 'score': 0.019430125132203102}, {'name': 'Contentment', 'score': 0.06073099002242088}, {'name': 'Craving', 'score': 0.006501526106148958}, {'name': 'Determination', 'score': 0.010847826488316059}, {'name': 'Disappointment', 'score': 0.01760883629322052}, {'name': 'Disapproval', 'score': 0.009595056995749474}, {'name': 'Disgust', 'score': 0.003568764077499509}, {'name': 'Distress', 'score': 0.004913835320621729}, {'name': 'Doubt', 'score': 0.020732788369059563}, {'name': 'Ecstasy', 'score': 0.0070318495854735374}, {'name': 'Embarrassment', 'score': 0.002762148156762123}, {'name': 'Empathic Pain', 'score': 0.004658285528421402}, {'name': 'Enthusiasm', 'score': 0.020970847457647324}, {'name': 'Entrancement', 'score': 0.029176874086260796}, {'name': 'Envy', 'score': 0.009052899666130543}, {'name': 'Excitement', 'score': 0.011401181109249592}, {'name': 'Fear', 'score': 0.0018732709577307105}, {'name': 'Gratitude', 'score': 0.007437469437718391}, {'name': 'Guilt', 'score': 0.0009437649860046804}, {'name': 'Horror', 'score': 0.0012521253665909171}, {'name': 'Interest', 'score': 0.10246434807777405}, {'name': 'Joy', 'score': 0.016914378851652145}, {'name': 'Love', 'score': 0.0034126662649214268}, {'name': 'Nostalgia', 'score': 0.008389955386519432}, {'name': 'Pain', 'score': 0.004401583690196276}, {'name': 'Pride', 'score': 0.012128319591283798}, {'name': 'Realization', 'score': 0.076319120824337}, {'name': 'Relief', 'score': 0.01443533692508936}, {'name': 'Romance', 'score': 0.0022930167615413666}, {'name': 'Sadness', 'score': 0.006305092945694923}, {'name': 'Sarcasm', 'score': 0.018501555547118187}, {'name': 'Satisfaction', 'score': 0.07013726979494095}, {'name': 'Desire', 'score': 0.004992782603949308}, {'name': 'Shame', 'score': 0.004370391834527254}, {'name': 'Surprise (negative)', 'score': 0.028048569336533546}, {'name': 'Surprise (positive)', 'score': 0.05226348340511322}, {'name': 'Sympathy', 'score': 0.007420244161039591}, {'name': 'Tiredness', 'score': 0.02038390189409256}, {'name': 'Triumph', 'score': 0.02609831653535366}]]

y1_volume = max(X[0], key=lambda x: x['score'])
y2_volume = max(X[1], key=lambda x: x['score'])
note_number = 60  
frequency = note_to_frequency(note_number)

# Y = [[, y1_volume], [, y2_volume]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

pca = PCA(n_components=10)  
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

model = LinearRegression()

model.fit(X_train_pca, y_train)

predictions = model.predict(X_test_pca)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

new_data_pca = pca.transform(new_data)
predicted_values = model.predict(new_data_pca)
print("Predicted [frequency, amplitude]:", predicted_values)
