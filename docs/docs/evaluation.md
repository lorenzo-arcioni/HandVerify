# Evaluation

Guida completa alla valutazione dei modelli di verifica della calligrafia.

## 📋 Indice

- [Metriche Biometriche](#metriche-biometriche)
- [Metriche di Classificazione](#metriche-di-classificazione)
- [Curve ROC e DET](#curve-roc-e-det)
- [Operating Points](#operating-points)
- [Interpretazione Risultati](#interpretazione-risultati)
- [Confronto Modelli](#confronto-modelli)
- [Visualizzazioni](#visualizzazioni)

## 📊 Metriche Biometriche

Le metriche biometriche sono progettate specificamente per sistemi di verifica dell'identità.

### Equal Error Rate (EER)

**Definizione:** Punto in cui False Accept Rate (FAR) = False Reject Rate (FRR).

```
EER = punto dove FAR = FRR
```

**Calcolo:**
```python
from src.evaluation import compute_eer

# Da curva ROC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
eer, eer_threshold = compute_eer(fpr, tpr, thresholds)

print(f"EER: {eer:.4f} ({eer*100:.2f}%)")
print(f"Threshold @ EER: {eer_threshold:.4f}")
```

**Interpretazione:**

| EER | Qualità | Contesto |
|-----|---------|----------|
| < 1% | Eccellente | Sistemi commerciali top-tier |
| 1-3% | Molto buono | Applicazioni pratiche |
| 3-5% | Buono | Ricerca, prototipi |
| 5-10% | Accettabile | Baseline, primi esperimenti |
| > 10% | Scarso | Richiede miglioramenti |

**Vantaggi:**
- ✅ Singola metrica riassuntiva
- ✅ Indipendente dalla distribuzione delle classi
- ✅ Standard nell'industria biometrica

**Svantaggi:**
- ❌ Non sempre corrisponde all'operating point reale
- ❌ Non cattura la forma completa della curva ROC

### Area Under ROC Curve (AUC)

**Definizione:** Area sotto la curva ROC (Receiver Operating Characteristic).

```
AUC ∈ [0, 1]
AUC = 1.0 → Classificatore perfetto
AUC = 0.5 → Classificatore casuale
```

**Calcolo:**
```python
from sklearn.metrics import roc_auc_score, roc_curve, auc

# Metodo 1: diretto
auc_score = roc_auc_score(y_true, y_scores)

# Metodo 2: da curva ROC
fpr, tpr, _ = roc_curve(y_true, y_scores)
auc_score = auc(fpr, tpr)

print(f"AUC: {auc_score:.4f}")
```

**Interpretazione:**

| AUC | Qualità | Significato |
|-----|---------|-------------|
| 0.99-1.00 | Eccellente | Separazione quasi perfetta |
| 0.95-0.99 | Molto buono | Separazione molto buona |
| 0.90-0.95 | Buono | Separazione accettabile |
| 0.80-0.90 | Discreto | Necessita miglioramenti |
| < 0.80 | Scarso | Inadeguato |

**Vantaggi:**
- ✅ Invariante alla scala
- ✅ Invariante al threshold
- ✅ Valuta prestazioni su tutti i threshold

**Svantaggi:**
- ❌ Può essere ottimistico se classi sbilanciate
- ❌ Non fornisce threshold operativo

### False Accept Rate (FAR) e False Reject Rate (FRR)

**Definizioni:**
```
FAR = False Positives / Total Impostors
    = P(accept | impostor)
    = Probabilità di accettare un impostore

FRR = False Negatives / Total Genuines
    = P(reject | genuine)
    = Probabilità di rigettare un genuino
```

**Relazione con threshold:**
```
Threshold basso  → FAR alto, FRR basso (permissivo)
Threshold alto   → FAR basso, FRR alto (restrittivo)
EER             → FAR = FRR (bilanciato)
```

**Visualizzazione:**
```python
import matplotlib.pyplot as plt

# Calcola FAR e FRR per vari threshold
thresholds = np.linspace(0, 1, 100)
far_values = []
frr_values = []

for threshold in thresholds:
    predictions = (y_scores >= threshold).astype(int)
    
    # FAR = FP / (FP + TN)
    fp = np.sum((predictions == 1) & (y_true == 0))
    tn = np.sum((predictions == 0) & (y_true == 0))
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # FRR = FN / (FN + TP)
    fn = np.sum((predictions == 0) & (y_true == 1))
    tp = np.sum((predictions == 1) & (y_true == 1))
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    far_values.append(far)
    frr_values.append(frr)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, far_values, label='FAR', linewidth=2)
plt.plot(thresholds, frr_values, label='FRR', linewidth=2)
plt.xlabel('Threshold')
plt.ylabel('Error Rate')
plt.title('FAR vs FRR')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('far_frr_curve.png', dpi=150)
```

### d-prime (d')

**Definizione:** Misura la separabilità tra le distribuzioni genuine e impostor.

```
d' = (μ_impostor - μ_genuine) / σ_pooled

σ_pooled = sqrt((σ²_genuine + σ²_impostor) / 2)
```

**Calcolo:**
```python
from src.evaluation import compute_verification_metrics

metrics = compute_verification_metrics(genuine_dists, impostor_dists)

print(f"d-prime: {metrics['d_prime']:.4f}")
print(f"Decidability: {metrics['decidability']:.4f}")  # d' * sqrt(2)
```

**Interpretazione:**

| d' | Qualità | Significato |
|----|---------|-------------|
| > 4.0 | Eccellente | Distribuzioni molto separate |
| 3.0-4.0 | Molto buono | Buona separazione |
| 2.0-3.0 | Buono | Separazione moderata |
| 1.0-2.0 | Discreto | Sovrapposizione significativa |
| < 1.0 | Scarso | Alta sovrapposizione |

**Vantaggi:**
- ✅ Indipendente dal threshold
- ✅ Interpretazione intuitiva (teoria dei segnali)
- ✅ Robusto a distribuzioni non perfettamente gaussiane

**Esempio visualizzazione:**
```python
import matplotlib.pyplot as plt
import numpy as np

# Distribuzioni
plt.figure(figsize=(12, 6))
plt.hist(genuine_dists, bins=50, alpha=0.6, label='Genuine', density=True)
plt.hist(impostor_dists, bins=50, alpha=0.6, label='Impostor', density=True)

# Medie
plt.axvline(np.mean(genuine_dists), color='blue', linestyle='--', 
            label=f'μ_genuine = {np.mean(genuine_dists):.3f}')
plt.axvline(np.mean(impostor_dists), color='red', linestyle='--',
            label=f'μ_impostor = {np.mean(impostor_dists):.3f}')

plt.xlabel('Distance')
plt.ylabel('Density')
plt.title(f"Distance Distributions (d' = {metrics['d_prime']:.3f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('distance_distributions.png', dpi=150)
```

## 📈 Metriche di Classificazione

### Accuracy

**Definizione:** Frazione di predizioni corrette.

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Calcolo:**
```python
from sklearn.metrics import accuracy_score

predictions = (y_scores >= threshold).astype(int)
accuracy = accuracy_score(y_true, predictions)

print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
```

**Attenzione:** Può essere fuorviante con classi sbilanciate!

### Precision, Recall, F1-Score

**Definizioni:**
```
Precision = TP / (TP + FP)
          = Accuratezza delle predizioni positive

Recall = TP / (TP + FN)
       = Frazione di positivi catturati

F1 = 2 * (Precision * Recall) / (Precision + Recall)
   = Media armonica di precision e recall
```

**Calcolo:**
```python
from sklearn.metrics import precision_score, recall_score, f1_score

predictions = (y_scores >= threshold).astype(int)

precision = precision_score(y_true, predictions)
recall = recall_score(y_true, predictions)
f1 = f1_score(y_true, predictions)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

**Interpretazione nel contesto biometrico:**

| Metrica | Significato | Trade-off |
|---------|-------------|-----------|
| **Precision alta** | Pochi falsi positivi | Utente raramente accetta impostori |
| **Recall alto** | Pochi falsi negativi | Utente genuino raramente rigettato |
| **F1 alto** | Bilanciamento | Equilibrio tra i due |

### Confusion Matrix

**Visualizzazione:**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calcola confusion matrix
predictions = (y_scores >= threshold).astype(int)
cm = confusion_matrix(y_true, predictions)

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Impostor', 'Genuine'],
            yticklabels=['Impostor', 'Genuine'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Threshold={threshold:.3f})')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)

# Interpretazione
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")
```

## 📉 Curve ROC e DET

### ROC Curve (Receiver Operating Characteristic)

**Asse X:** False Positive Rate (FPR) = FAR  
**Asse Y:** True Positive Rate (TPR) = 1 - FRR

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calcola curva ROC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, linewidth=2, label=f'Model (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

# EER point
eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
plt.plot(fpr[eer_idx], tpr[eer_idx], 'ro', markersize=10, 
         label=f'EER = {fpr[eer_idx]:.4f}')

plt.xlabel('False Positive Rate (FAR)', fontsize=12)
plt.ylabel('True Positive Rate (1 - FRR)', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
```

**Interpretazione:**
- **Curva vicino all'angolo superiore sinistro**: Prestazioni ottime
- **Curva sulla diagonale**: Performance casuale
- **Area sotto la curva (AUC)**: Qualità globale

### DET Curve (Detection Error Tradeoff)

**Asse X:** False Accept Rate (FAR) - scala logaritmica  
**Asse Y:** False Reject Rate (FRR) - scala logaritmica

```python
# Calcola FRR
fnr = 1 - tpr

# Plot DET
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr, fnr, linewidth=2, label='Model')
ax.plot([0.001, 1], [0.001, 1], 'k--', linewidth=1, label='EER Line')

# EER point
ax.plot(fpr[eer_idx], fnr[eer_idx], 'ro', markersize=10,
        label=f'EER = {fpr[eer_idx]:.4f}')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('False Accept Rate (FAR)', fontsize=12)
ax.set_ylabel('False Reject Rate (FRR)', fontsize=12)
ax.set_title('DET Curve', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('det_curve.png', dpi=150)
```

**Vantaggi DET rispetto ROC:**
- ✅ Scala logaritmica evidenzia differenze a bassi error rates
- ✅ EER è sulla diagonale (più chiaro)
- ✅ Meglio per confrontare sistemi biometrici

## 🎯 Operating Points

Gli operating points sono threshold specifici scelti per applicazioni reali.

### Operating Points Standard

```python
from src.evaluation import compute_verification_metrics

metrics = compute_verification_metrics(genuine_dists, impostor_dists)

# Operating points calcolati automaticamente
print(f"\n{'='*60}")
print("OPERATING POINTS")
print(f"{'='*60}")

# EER point (bilanciato)
print(f"\n1. EER Point (Balanced):")
print(f"   Threshold: {metrics['eer_threshold']:.4f}")
print(f"   FAR = FRR: {metrics['eer']:.4f}")
print(f"   Accuracy:  {metrics['accuracy']:.4f}")

# High security (FAR = 0.1%)
print(f"\n2. High Security (FAR = 0.1%):")
print(f"   Threshold: {metrics['threshold_far_0.001']:.4f}")
print(f"   FRR:       {metrics['frr_far_0.001']:.4f}")
print(f"   Accuracy:  {metrics['acc_far_0.001']:.4f}")

# Moderate security (FAR = 1.0%)
print(f"\n3. Moderate Security (FAR = 1.0%):")
print(f"   Threshold: {metrics['threshold_far_0.01']:.4f}")
print(f"   FRR:       {metrics['frr_far_0.01']:.4f}")
print(f"   Accuracy:  {metrics['acc_far_0.01']:.4f}")

print(f"{'='*60}\n")
```

### Scegliere Operating Point

**Criteri di scelta:**

| Applicazione | Priorità | Operating Point |
|--------------|----------|-----------------|
| **Banking/Finance** | Minimizza FAR | FAR = 0.01% - 0.1% |
| **Access Control** | Bilancia FAR/FRR | EER point |
| **User Experience** | Minimizza FRR | FAR = 1% - 5% |
| **Forensics** | Massima accuratezza | Caso per caso |

**Esempio personalizzato:**
```python
# Trova threshold per target FAR
def find_threshold_for_far(fpr, thresholds, target_far=0.001):
    idx = np.argmin(np.abs(fpr - target_far))
    return thresholds[idx], fpr[idx]

# Per FAR = 0.01% (1 in 10,000)
target_far = 0.0001
threshold, actual_far = find_threshold_for_far(fpr, thresholds, target_far)

print(f"Target FAR: {target_far:.5f}")
print(f"Threshold:  {threshold:.4f}")
print(f"Actual FAR: {actual_far:.5f}")
```

## 🔍 Interpretazione Risultati

### Output Completo Valutazione

```python
from src.evaluation import compute_verification_metrics, print_verification_results

# Calcola metriche
metrics = compute_verification_metrics(genuine_dists, impostor_dists)

# Stampa report formattato
print_verification_results(metrics, dataset_name="Test Set")
```

**Output:**
```
======================================================================
VERIFICATION METRICS: Test Set
======================================================================
📊 PRIMARY METRICS:
  EER (Equal Error Rate):     0.0245 (2.45%)
  AUC-ROC:                     0.9912
  Accuracy @ EER threshold:    0.9755

🎯 CLASSIFICATION METRICS (@ EER threshold):
  Precision:                   0.9782
  Recall:                      0.9734
  F1-Score:                    0.9758

📈 DISCRIMINABILITY:
  d-prime (d'):                3.2456
  Decidability Index:          4.5912

📏 DISTANCE STATISTICS:
  Genuine:  μ=0.3456, σ=0.1234
  Impostor: μ=0.8912, σ=0.1567

⚙️ OPERATING POINTS:
  EER Threshold:               0.4567
  Accuracy @ FAR=0.1%:         0.9623
  Accuracy @ FAR=1.0%:         0.9789
  Threshold @ FAR=0.1%:        0.6789
  Threshold @ FAR=1.0%:        0.5234
======================================================================
```

### Interpretazione Step-by-Step

#### 1. Guarda EER e AUC
```python
if metrics['eer'] < 0.02 and metrics['auc'] > 0.99:
    print("✅ Prestazioni ECCELLENTI - production ready")
elif metrics['eer'] < 0.05 and metrics['auc'] > 0.95:
    print("✅ Prestazioni BUONE - applicabile")
elif metrics['eer'] < 0.10:
    print("⚠️ Prestazioni ACCETTABILI - necessita miglioramenti")
else:
    print("❌ Prestazioni INSUFFICIENTI - riprogettare")
```

#### 2. Verifica d-prime
```python
if metrics['d_prime'] > 4.0:
    print("✅ Separazione ECCELLENTE tra genuine/impostor")
elif metrics['d_prime'] > 2.0:
    print("✅ Separazione BUONA")
else:
    print("⚠️ Sovrapposizione SIGNIFICATIVA - rivedere features")
```

#### 3. Analizza Distribuzioni
```python
# Overlap delle distribuzioni
genuine_max = np.max(genuine_dists)
impostor_min = np.min(impostor_dists)

if genuine_max < impostor_min:
    print("✅ Nessuna sovrapposizione - separazione perfetta")
else:
    overlap = (genuine_max - impostor_min) / (impostor_max - genuine_min)
    print(f"⚠️ Overlap: {overlap:.2%}")
```

#### 4. Confronta Operating Points
```python
# Se FRR @ FAR=0.1% è troppo alto
if metrics['frr_far_0.001'] > 0.1:
    print("⚠️ High security mode rifiuta troppi genuini")
    print("   → Considera threshold più permissivo o miglioramenti al modello")
```

## 📊 Confronto Modelli

### Confronto Multiplo

```python
# scripts/compare_models.py
import pandas as pd
import matplotlib.pyplot as plt

# Carica metriche di vari modelli
models = {
    'ResNet18': pd.read_csv('results/resnet18/resnet18_final_metrics.csv'),
    'EfficientNet-B0': pd.read_csv('results/efficientnet/efficientnet_final_metrics.csv'),
    'MobileNetV3': pd.read_csv('results/mobilenet/mobilenet_final_metrics.csv'),
}

# Estrai metriche chiave
comparison = []
for model_name, metrics_df in models.items():
    metrics = metrics_df.iloc[0]
    comparison.append({
        'Model': model_name,
        'EER (%)': metrics['eer'] * 100,
        'AUC': metrics['auc'],
        'Accuracy': metrics['accuracy'],
        'd-prime': metrics['d_prime']
    })

df = pd.DataFrame(comparison)

# Tabella
print(f"\n{'='*70}")
print("MODEL COMPARISON")
print(f"{'='*70}")
print(df.to_string(index=False))
print(f"{'='*70}\n")

# Plot confronto
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# EER
axes[0].bar(df['Model'], df['EER (%)'])
axes[0].set_ylabel('EER (%)')
axes[0].set_title('Equal Error Rate')
axes[0].tick_params(axis='x', rotation=45)

# AUC
axes[1].bar(df['Model'], df['AUC'])
axes[1].set_ylabel('AUC')
axes[1].set_title('Area Under ROC')
axes[1].set_ylim([0.9, 1.0])
axes[1].tick_params(axis='x', rotation=45)

# d-prime
axes[2].bar(df['Model'], df['d-prime'])
axes[2].set_ylabel("d'")
axes[2].set_title('Discriminability (d-prime)')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
```

### ROC Curves Comparison

```python
# Plot ROC curve per ciascun modello
plt.figure(figsize=(10, 8))

for model_name, metrics_df in models.items():
    metrics = metrics_df.iloc[0]
    fpr = eval(metrics['fpr'])  # Salvato come string in CSV
    tpr = eval(metrics['tpr'])
    auc_score = metrics['auc']
    
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc_score:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_comparison.png', dpi=150)
```

### Statistical Significance

Test se la differenza tra modelli è statisticamente significativa:

```python
from scipy import stats

# McNemar's test per confrontare due modelli
def mcnemar_test(y_true, pred1, pred2):
    """
    Test if two models have significantly different error rates.
    """
    # Tabella contingenza
    both_correct = np.sum((pred1 == y_true) & (pred2 == y_true))
    both_wrong = np.sum((pred1 != y_true) & (pred2 != y_true))
    pred1_correct = np.sum((pred1 == y_true) & (pred2 != y_true))
    pred2_correct = np.sum((pred1 != y_true) & (pred2 == y_true))
    
    # Chi-square statistic
    statistic = (pred1_correct - pred2_correct)**2 / (pred1_correct + pred2_correct)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return statistic, p_value

# Test
statistic, p_value = mcnemar_test(y_true, predictions_model1, predictions_model2)

if p_value < 0.05:
    print(f"✅ Differenza SIGNIFICATIVA (p={p_value:.4f})")
else:
    print(f"⚠️ Differenza NON significativa (p={p_value:.4f})")
```

## 📈 Visualizzazioni

### 1. Distance Distribution Plot

```python
def plot_distance_distributions(genuine_dists, impostor_dists, save_path='dist_plot.png'):
    """Plot genuine vs impostor distance distributions."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histograms
    ax.hist(genuine_dists, bins=50, alpha=0.6, label='Genuine', 
            color='green', density=True, edgecolor='black')
    ax.hist(impostor_dists, bins=50, alpha=0.6, label='Impostor', 
            color='red', density=True, edgecolor='black')
    
    # Statistics
    mu_g, sigma_g = np.mean(genuine_dists), np.std(genuine_dists)
    mu_i, sigma_i = np.mean(impostor_dists), np.std(impostor_dists)
    
    ax.axvline(mu_g, color='green', linestyle='--', linewidth=2,
               label=f'μ_genuine = {mu_g:.3f}')
    ax.axvline(mu_i, color='red', linestyle='--', linewidth=2,
               label=f'μ_impostor = {mu_i:.3f}')
    
    # d-prime
    d_prime = (mu_i - mu_g) / np.sqrt((sigma_g**2 + sigma_i**2) / 2)
    
    ax.set_xlabel('Distance', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f"Distance Distributions (d' = {d_prime:.3f})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# Utilizzo
plot_distance_distributions(
    metrics['genuine_dists'],
    metrics['impostor_dists'],
    'distance_distributions.png'
)
```

### 2. Error Rates vs Threshold

```python
def plot_error_rates(fpr, fnr, thresholds, eer_idx, save_path='error_rates.png'):
    """Plot FAR and FRR vs threshold."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(thresholds, fpr, label='FAR (False Accept Rate)', 
            linewidth=2, color='red')
    ax.plot(thresholds, fnr, label='FRR (False Reject Rate)', 
            linewidth=2, color='blue')
    
    # EER point
    ax.plot(thresholds[eer_idx], fpr[eer_idx], 'go', markersize=12,
            label=f'EER = {fpr[eer_idx]:.4f}')
    ax.axvline(thresholds[eer_idx], color='green', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title('Error Rates vs Decision Threshold', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# Calcola FNR
fnr = 1 - metrics['tpr']
eer_idx = np.argmin(np.abs(metrics['fpr'] - fnr))

plot_error_rates(
    metrics['fpr'],
    fnr,
    metrics['thresholds'],
    eer_idx,
    'error_rates.png'
)
```

### 3. Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall(y_true, y_scores, save_path='pr_curve.png'):
    """Plot Precision-Recall curve."""
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(recall, precision, linewidth=2, 
            label=f'Model (AP = {ap_score:.4f})')
    ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1, label='Baseline')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

plot_precision_recall(y_true, y_scores, 'pr_curve.png')
```

## ✅ Best Practices

### 1. Validazione Completa

```python
# Non limitarti solo a EER/AUC
metrics = compute_verification_metrics(genuine_dists, impostor_dists)

# Controlla TUTTE le metriche
checks = {
    'EER < 5%': metrics['eer'] < 0.05,
    'AUC > 0.95': metrics['auc'] > 0.95,
    "d' > 2.0": metrics['d_prime'] > 2.0,
    'Genuine μ < Impostor μ': metrics['mu_genuine'] < metrics['mu_impostor'],
}

print("\nValidation Checks:")
for check, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"{status} {check}")
```

### 2. Test su Dati Indipendenti

```python
# Usa sempre un test set separato (MAI visto durante training)
train_loader, test_loader, train_ds, test_ds = create_dataloaders(
    ...,
    test_size=0.2,  # 20% per test
    random_state=42  # Fisso per riproducibilità
)

# Non usare validation set per metriche finali!
```

### 3. Multiple Runs per Robustezza

```python
# Esegui training multiple volte
results = []

for seed in [42, 123, 456, 789, 1011]:
    set_seed(seed)
    # Training...
    metrics = validate(model, test_ds)
    results.append(metrics['eer'])

# Report con incertezza
mean_eer = np.mean(results)
std_eer = np.std(results)
print(f"EER: {mean_eer:.4f} ± {std_eer:.4f}")
```

### 4. Confronti Equi

```python
# SEMPRE stesso test set per confrontare modelli
# SEMPRE stessi hyperparameter quando possibile
# SEMPRE stesso numero di epoche o early stopping

# ❌ SBAGLIATO
model1_metrics = validate(model1, test_set_1)
model2_metrics = validate(model2, test_set_2)  # Test set diverso!

# ✅ CORRETTO
model1_metrics = validate(model1, test_set)
model2_metrics = validate(model2, test_set)  # Stesso test set
```

### 5. Documenta Operating Point

```python
# Specifica SEMPRE quale threshold usi in produzione
CONFIG = {
    'model': 'resnet18',
    'operating_point': 'EER',  # o 'FAR_0.001'
    'threshold': 0.4567,
    'expected_far': 0.0245,
    'expected_frr': 0.0245,
}

# Salva configurazione
with open('deployment_config.json', 'w') as f:
    json.dump(CONFIG, f, indent=2)
```

## 🔍 Troubleshooting

### Problema: EER alto ma AUC buono

**Causa:** Distribuzioni ben separate ma EER point non ottimale.

**Soluzione:**
```python
# Analizza la curva ROC completa
# Forse un altro operating point è migliore
print(f"Accuracy @ FAR=1%: {metrics['acc_far_0.01']:.4f}")
```

### Problema: d-prime basso ma EER accettabile

**Causa:** Distribuzioni sovrapposte ma classificatore riesce a separare.

**Soluzione:**
```python
# Visualizza distribuzioni
plot_distance_distributions(genuine_dists, impostor_dists)
# Considera feature engineering o architettura diversa
```

### Problema: Varianza alta tra fold

**Causa:** Dataset piccolo o pochi writer per fold.

**Soluzione:**
```python
# Aumenta n_splits
trainer.train_kfold(..., n_splits=10)  # invece di 5

# Oppure usa stratified split per bilanciare
```

### Problema: Performance test << validation

**Causa:** Overfitting o data leakage.

**Soluzione:**
```python
# 1. Verifica data leakage
# Stessi writer in train e val?

# 2. Aumenta regolarizzazione
# Dropout, weight decay, data augmentation

# 3. Early stopping più aggressivo
trainer.train(..., patience=5)  # invece di 10
```

## 📚 Risorse Aggiuntive

- **[Training](training.md)**: Come ottimizzare per metriche migliori
- **[Results](results.md)**: Benchmark e confronti
- **[Notebooks](notebooks.md)**: Tutorial su analisi risultati

---

**Prossimo:** [Notebooks →](notebooks.md)