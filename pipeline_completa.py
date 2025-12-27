import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from tqdm import tqdm
import multiprocessing

'''Prima parte di GENERAZIONE DEL DATASET. Impostare i parametri n_samples e p_y1 in base a quanto si vuole 
rendere sbilanciato il dataset nelle due classi. '''

'''print("Generazione del dataset...")'''

'''n_samples = 10000
prop_target1 = 0.3'''

'''def generate_dgp(n_samples, p_y1, random_state):
    rng = np.random.default_rng(random_state)'''
    
    #PRIMO DGP
'''X1 = rng.normal(2.0, 1.0, n_samples)
    X2 = rng.normal(2.0, 1.0, n_samples)
    X3 = rng.exponential(1.0, n_samples)
    X4 = rng.exponential(1.0, n_samples)
    X5 = rng.uniform(0.0, 4.0, n_samples)
    X6 = rng.uniform(0.0, 1.0, n_samples) * X5
    X7 = rng.integers(0, 2, n_samples)
    
    # Targets
    Y0 = X1 + X2 * X7 - 3.0 * (X6 ** 2)
    Y1 = X4 + X2 * X3 * X6 - (X5 ** 3)'''

    #SECONDO DGP
'''X1  = rng.normal(2.0, 1.0, n_samples)                       # N(2,1)
    X2  = rng.normal(1.0, np.abs(X1), n_samples)                # N(1, |X1|)
    X3  = rng.exponential(1.0, n_samples)                       # Exp(1)
    X4  = rng.poisson(4.0, n_samples)                           # Poi(4)
    X5  = rng.random(n_samples) * X4                            # Unif[0,X4]
    X6  = rng.random(n_samples) * (X4 - X5) + X5                # Unif[X5,X4]
    X7  = rng.random(n_samples)                                 # Unif[0,1]
    X8  = (rng.random(n_samples) < X7).astype(int)              # Ber(X7)
    X9  = np.where(                                             # mixture
            X8 == 1,
            rng.normal(-5.0, 3.0, n_samples),                   #   N(−5,3)
            rng.binomial(X4, X7)                                #   Bin(X7, X4)
          )
    X10 = rng.normal(X2, X3)                                    # N(X2, X3)
    X11 = rng.poisson(6.0, n_samples)                           # Poi(6)

    # ---- Potential outcomes ----
    Y0 = (X10 ** np.abs(X3)) + X6 * X1 - X3 * X4
    Y1 = (X4  ** np.abs(X2)) + X9 + X1 * X2'''

    #TERZO DGP
'''X1  = rng.normal(0.0, 1.0, n_samples)
    X2  = rng.lognormal(mean=0.0, sigma=0.5, size=n_samples)          
    X3  = rng.beta(2.0, 5.0, n_samples)                               
    X4  = rng.poisson(3.0, n_samples)
    X5  = rng.uniform(-3.0, 3.0, n_samples)
    X6  = rng.random(n_samples) < 0.4                                 

    p_clip = np.clip(X3, 0.05, 0.95)                                  
    X7  = rng.binomial(X4, p_clip)
    X8  = rng.normal(X2 * X3, 0.5, n_samples)
    X9  = rng.uniform(0.0, X2 + 1.0)
    X10 = rng.exponential(5.0, n_samples)                             
    X11 = rng.chisquare(df=3, size=n_samples)
    X12 = np.where(X6 == 1, rng.laplace(0.0, 1.0, n_samples), 0.0)

    # Variabili inutili / distrattori
    X13 = rng.random(n_samples)
    X14 = (rng.random(n_samples) < 0.5).astype(int)
    X15 = rng.poisson(1.0, n_samples)
    X16 = rng.normal(0.0, 1.0, n_samples)

    # target
    Y0 = (np.log1p(X2) +
          np.sqrt(X11) +
          X6 * X5**2 -
          0.1 * X10 +
          np.sin(X1))

    Y1 = (np.sqrt(X4 + 1) +
          np.cos(X5) +
          X7 / (1.0 + X2) +
          X12 +
          0.5 * X8**2)'''
    
'''choose_y1 = rng.random(n_samples) < p_y1
    Y = np.where(choose_y1, Y1, Y0)
    target = choose_y1.astype(int)
    
    data = np.column_stack([X1, X2, X3, X4, X5, X6, X7, Y, target])
    columns = [f"X{i}" for i in range(1, 8)] + ["Y", "target"]
    df = pd.DataFrame(data, columns=columns)
    return df

# Generate dataset
df = generate_dgp(n_samples, prop_target1, random_state=2025)

#Salvo il dataset generato
df.to_csv('dataset_tot.csv', index=False)'''

'''SUDDIVISIONE DEL DATASET in Train Set - Test Set - Bayes Test Set. Utilizzeremo il primo per addestrare
100 modelli, il secondo per calcolare le loro performance medie, il terzo per calcolare le performance 
congiunte mediante un approccio Bayesiano.'''

print("\nSuddivisione del dataset in Train - Test - Bayes Test...")

# Caricamento del dataset
df = pd.read_csv('spambase.csv', delimiter=',')

# Separazione delle features e del target
X = df.drop(['is_spam'], axis=1)  # Rimuoviamo la variabile target dalle features
y = df['is_spam']

# Prima divisione: 60% train, 40% per test e Bayes test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

# Seconda divisione: dividiamo il 40% rimanente in due parti uguali (20% test, 20% Bayes test)
X_test, X_bayes, y_test, y_bayes = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Salvataggio dei set di dati per uso futuro
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
bayes_data = pd.concat([X_bayes, y_bayes], axis=1)

train_data.to_csv('train_set.csv', index=False)
test_data.to_csv('test_set.csv', index=False)
bayes_data.to_csv('bayes_test_set.csv', index=False)

p_y1 = y_train.mean()
print(f"\nStima proporzione delle unità con target=1 nel dataset: {p_y1}")

'''ADDESTRAMENTO di 100 MODELLI di Random Forest costituiti ciascuno da un solo albero. Ogni modello è a 
priori diverso dall'altro in quanto vengono forniti random seed diversi. Se si calcolano le prestazioni di
classificazione prendendo come predizione il voto di maggioranza tra i 100 modelli, si ottengono performance
pressoché identiche ad un'unica random forest costituita da 100 alberi.'''

print("\nAddestramento e salvataggio dei 100 modelli Random Forest...")
print(f"Utilizzo {os.cpu_count()} core della CPU")

def train_and_save_models():
    # Caricamento del train set
    train_data = pd.read_csv('train_set.csv')
    
    # Separazione delle features e del target nel train set
    X_train = train_data.drop(['is_spam'], axis=1)
    y_train = train_data['is_spam']
    
    # Lista dei random_state da utilizzare
    random_states = range(1, 101)
    
    # Creazione della directory per i modelli se non esiste
    models_dir = 'saved_models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    
    # Addestramento e salvataggio dei modelli
    for random_state in tqdm(random_states, desc="Addestrando modelli"):
        # Creazione e addestramento del modello Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=1, 
            max_depth=None, 
            random_state=random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Salvataggio del modello
        model_filename = f'{models_dir}/model_{random_state}.joblib'
        joblib.dump(rf_model, model_filename)

    print(f"\nTutti i 100 modelli sono stati salvati nella directory '{models_dir}'")

train_and_save_models()

'''Funzione per INTERROGARE uno specifico modello.'''

def load_and_predict(X_test, random_state):
    # Caricamento del modello salvato
    model_filename = f'saved_models/model_{random_state}.joblib'
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Modello {random_state} non trovato. Esegui prima train_and_save_models.py")
    
    rf_model = joblib.load(model_filename)
    
    # Predizione
    prediction = rf_model.predict(X_test)
    return prediction[0] 

'''ANALISI ERRORI dei singoli modelli, calcolo della probabilità media sui dati presenti nel Test Set che un 
singolo modello commetta errori di falso positivo e falso negativo.'''

print("\nCALCOLO ERRORI singoli modelli...")

def analyze_model_errors():
    # Caricamento del test set
    test_data = pd.read_csv('test_set.csv')
    
    # Preparazione dei dati
    X_test = test_data.drop(['is_spam'], axis=1)
    y_true = test_data['is_spam']
    
    # Lista dei random_state da utilizzare
    random_states = range(1, 101)
    
    # Controllo se i modelli salvati esistono
    if not os.path.exists('saved_models/model_1.joblib'):
        print("Errore: Modelli salvati non trovati!")
        return
    
    # Inizializzazione delle variabili di conteggio
    count_target_0 = 0  # Conta unità con target = 0
    count_target_1 = 0  # Conta unità con target = 1
    total_false_positives = 0  # Conta falsi positivi totali
    total_false_negatives = 0  # Conta falsi negativi totali
    
    print(f"Numero totale di unità da analizzare: {len(X_test)}")
    print(f"Numero di modelli per unità: {len(random_states)}")
    
    # Analisi di ogni unità nel test set
    for i, (_, unit) in enumerate(tqdm(X_test.iterrows(), desc="Analizzando unità", unit="unità")):
        # Creo un DataFrame con la singola unità
        unit_to_predict = pd.DataFrame([unit], columns=X_test.columns)
        true_target = y_true.iloc[i]
        
        # Aggiorno i contatori per il target reale
        if true_target == 0:
            count_target_0 += 1
        else:
            count_target_1 += 1
        
        # Interrogo tutti i 100 modelli per questa unità
        for random_state in random_states:
            prediction = load_and_predict(unit_to_predict, random_state)
            
            # Analisi degli errori
            if true_target == 0 and prediction == 1:
                # Falso positivo: target = 0, predetto = 1
                total_false_positives += 1
            elif true_target == 1 and prediction == 0:
                # Falso negativo: target = 1, predetto = 0
                total_false_negatives += 1
    
    # Calcolo delle probabilità
    if count_target_0 > 0:
        prob_false_positive = total_false_positives / (count_target_0*100)
    else:
        prob_false_positive = 0
    
    if count_target_1 > 0:
        prob_false_negative = total_false_negatives / (count_target_1*100)
    else:
        prob_false_negative = 0
    
    # Stampa dei risultati
    print("\n" + "="*60)
    print("RISULTATI DELL'ANALISI DEGLI ERRORI")
    print("="*60)
    
    print(f"\nDistribuzione del target nel test set:")
    print(f"- Unità con target = 0: {count_target_0}")
    print(f"- Unità con target = 1: {count_target_1}")
    
    print(f"\nProbabilità di errore per singolo modello:")
    print(f"- Probabilità falso positivo: {prob_false_positive:.4f} ({prob_false_positive*100:.2f}%)")
    print(f"- Probabilità falso negativo: {prob_false_negative:.4f} ({prob_false_negative*100:.2f}%)")

    prob_true_positive = 1 - prob_false_negative
    prob_true_negative = 1 - prob_false_positive
    
    probabilities = [prob_true_negative, prob_false_positive, prob_false_negative, prob_true_positive]

    return probabilities

probabilities = analyze_model_errors()

'''SVILUPPO E LANCIO modello predittivo CLASIFICATORE BAYESIANO che utilizza le probabilità condizionali dei 
singoli modelli, cioè le probabilità di restituire falsi positivi e falsi negativi, per classificare le unità del 
bayes_test_set. Si utilizza dunque anche la probabilità a priori p_y1 per poter applicare il teorema di Bayes
e la legge delle probabilità totali e infine stimare per ogni unità la probabilità che questa appartenga
alla classe 1 dati i risultati delle predizioni dei 100 modelli. Se tale probabilità è > 0.5, l'unità viene
classificata nella classe 1, altrimenti nella classe 0.'''

print("\nCompilazione modello predittivo Bayesiano...")

def bayesian_classifier():
    # Caricamento del bayes test set
    bayes_test_data = pd.read_csv('bayes_test_set.csv')
    
    # Preparazione dei dati
    X_bayes_test = bayes_test_data.drop(['is_spam'], axis=1)
    y_true = bayes_test_data['is_spam']
    
    # Lista dei random_state da utilizzare
    random_states = range(1, 101)
    
    # Controllo se i modelli salvati esistono
    if not os.path.exists('saved_models/model_1.joblib'):
        print("Errore: Modelli salvati non trovati!")
        print("Esegui prima: python train_and_save_models.py")
        return
    
    # Definizione dei parametri del classificatore Bayesiano
    prior_prob_class0 = 1 - p_y1
    prior_prob_class1 = p_y1
    
    # Matrice di probabilità condizionali P(predizione|classe_reale)
    p00 = probabilities[0]  # P(pred=0|target=0) - True Negative rate
    p01 = probabilities[1]  # P(pred=1|target=0) - False Positive rate
    p10 = probabilities[2]  # P(pred=0|target=1) - False Negative rate
    p11 = probabilities[3]  # P(pred=1|target=1) - True Positive rate
    
    print(f"Probabilità a priori - Classe 0: {prior_prob_class0}, Classe 1: {prior_prob_class1}")
    
    # Lista per memorizzare le predizioni finali
    y_pred = []
    
    # Analisi di ogni unità nel bayes test set
    for i, (_, unit) in enumerate(tqdm(X_bayes_test.iterrows(), desc="Classificando unità", unit="unità")):
        # Creo un DataFrame con la singola unità
        unit_to_predict = pd.DataFrame([unit], columns=X_bayes_test.columns)
        
        # Interrogo tutti i 100 modelli per questa unità
        predictions = []
        for random_state in random_states:
            prediction = load_and_predict(unit_to_predict, random_state)
            predictions.append(prediction)
        
        # Calcolo della probabilità a posteriori P(target=1|predizioni)
        # Numeratore: P(predizioni|target=1) * P(target=1)
        numerator = prior_prob_class1
        # Denominatore: P(predizioni|target=0) * P(target=0)
        denominator = prior_prob_class0
        
        for pred in predictions:
            if pred == 0:
                # Se pred=0, moltiplico per p10 (se target=1) o p00 (se target=0)
                numerator *= p10
                denominator *= p00
            else:
                # Se pred=1, moltiplico per p11 (se target=1) o p01 (se target=0)
                numerator *= p11
                denominator *= p01
        
        # Calcolo della probabilità a posteriori
        post_prob_class1 = numerator / (numerator + denominator)
        
        # Classificazione finale
        if post_prob_class1 > 0.5:
            final_prediction = 1
        else:
            final_prediction = 0
        
        y_pred.append(final_prediction)
    
    # Calcolo delle metriche
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)
    
    # Stampa delle metriche
    print("\n" + "="*60)
    print("RISULTATI DEL CLASSIFICATORE BAYESIANO")
    print("="*60)
    
    print(f"\nMetriche di performance modello Bayesiano:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1score:.4f}")
    
    # Visualizzazione della matrice di confusione
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice di Confusione - Classificatore Bayesiano')
    plt.ylabel('Valori Reali')
    plt.xlabel('Valori Predetti')
    plt.savefig('bayesian_confusion_matrix.png')
    plt.close()

bayesian_classifier()

'''SVILUPPO E LANCIO del modello predittivo standard Random Forest, calcolo delle sue performance sul 
bayes_test_set per un confronto col modello predittivo bayesiano.'''

print("\nCompilazione modello predittivo Random Forest Standard...")

# Creazione e addestramento del modello Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=101)
rf_model.fit(X_train, y_train)

# Predizioni sul bayes_test_set
y_pred = rf_model.predict(X_bayes)

# Calcolo delle metriche
accuracy = accuracy_score(y_bayes, y_pred)
precision = precision_score(y_bayes, y_pred, zero_division=0)
recall = recall_score(y_bayes, y_pred, zero_division=0)
f1score = f1_score(y_bayes, y_pred, zero_division=0)
conf_matrix = confusion_matrix(y_bayes, y_pred)

# Stampa delle metriche
print("\nMetriche del modello Random Forest Standard:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1score:.4f}")

# Visualizzazione della matrice di confusione
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice di Confusione')
plt.ylabel('Valori Reali')
plt.xlabel('Valori Predetti')
plt.savefig('confusion_matrix.png')
plt.close()


