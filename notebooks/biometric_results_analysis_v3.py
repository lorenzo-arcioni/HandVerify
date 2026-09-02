# %% [markdown]
# # Analisi biometrica + Estrazione tabelle per la sezione Results
#
# Notebook esteso rispetto alla versione precedente. Oltre a tutta
# l'analisi gia' presente (bootstrap CI, normalita', ROC/DET, Wilcoxon,
# confondimenti noti, SOTA, failure analysis, ecc.) aggiunge il blocco di
# **estrazione dati puro** richiesto per riscrivere "Experimental Setup" e
# "Results": legge i log di training grezzi + i CSV per-esperimento e
# produce le Tabelle 1-5 richieste, in Markdown copia-incollabile.
#
# Nuovo in questa versione:
# 1. Parsing dei log di training (writer count, coppie genuine/impostor,
#    epoche, early stopping) — la versione precedente usava solo i CSV.
# 2. Le metriche puntuali (Tabelle 2-4) sono lette DIRETTAMENTE dal CSV
#    (`_final_metrics.csv`), non ricalcolate via bootstrap, per rispettare
#    la regola "CSV e' source of truth" della spec di estrazione. Il
#    ricalcolo via bootstrap (gia' presente) resta disponibile per le CI e
#    per i plot, come sezione separata.
# 3. Tabella 1 (protocollo writer/coppie) per split, dedotta dai log.
# 4. Tabelle 2/3 (AUC/EER, backbone x loss, per split) in Markdown.
# 5. Tabella 4 (long format completa) in Markdown, con colonna note per i
#    modelli "collassati" (AUC ~ 0.5).
# 6. Tabella 5 (aggregato same/cross per loss) SOLO se tutti e 6 i
#    backbone sono presenti per quella combinazione (loss, split);
#    altrimenti riporta esplicitamente quali backbone mancano invece di
#    calcolare una media parziale silenziosa.
# 7. Sezione "Missing configurations" esplicita (mai inventare/omettere
#    silenziosamente una configurazione mancante).
#
# Tutte le tabelle vengono anche salvate su disco come file `.md`
# indipendenti, pronti per essere incollati in LaTeX/paper.

# %%
import warnings
warnings.filterwarnings('ignore')

import ast
import re
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except (OSError, ValueError):
    plt.style.use('default')
sns.set_palette("husl")

Path('./images').mkdir(exist_ok=True)
Path('./tables').mkdir(exist_ok=True)

# %% [markdown]
# ## Configurazione

# %%
results_base = Path('../results')

loss_types = ['bce', 'contrastive', 'triplet']
dataset_combinations = ['iam_to_iam', 'iam_to_rimes', 'rimes_to_iam', 'rimes_to_rimes']

MODELS_TO_ANALYZE = [
    'efficientnet_b0', 'efficientnet_b1',
    'mobilenet_v3_large', 'mobilenet_v3_small',
    'resnet18', 'resnet34',
]

REPRESENTATIVE_MODEL = 'resnet34'

N_BOOTSTRAP = 1000
CI_LEVEL = 95
BOOTSTRAP_SEED = 42

# Soglia sotto la quale un AUC viene considerato "collassato" (chance level)
AUC_COLLAPSE_THRESHOLD = 0.55

# %% [markdown]
# ## Caricamento dati: risoluzione path per CSV metriche/history + log grezzo
#
# Come nella versione precedente, prova prima
# `results_base/{loss}/{loss}_experiments/...`, poi come fallback
# `results_base/{loss}_experiments/...`. Per il log grezzo (necessario
# SOLO per l'estrazione delle statistiche di protocollo/writer e delle
# dinamiche di training) si provano diversi nomi file plausibili: se il
# tuo script di training usa un nome diverso, aggiungi il pattern alla
# lista `log_name_candidates` qui sotto.

# %%
def resolve_experiment_path(loss, model_name, dataset_comb):
    experiment_name = f"{model_name}_{loss}_{dataset_comb}"
    candidates = [
        results_base / loss / f"{loss}_experiments" / experiment_name,
        results_base / f"{loss}_experiments" / experiment_name,
    ]
    for c in candidates:
        if (c / f"{experiment_name}_final_metrics.csv").exists():
            return c, experiment_name
    return None, experiment_name


def resolve_log_file(experiment_path, experiment_name):
    """Cerca il file di log grezzo (stdout capture) dell'esperimento.
    Ritorna il Path se trovato, altrimenti None. Adatta i pattern se il
    tuo script di training salva i log con un nome diverso."""
    if experiment_path is None:
        return None
    log_name_candidates = [
        f"{experiment_name}.log",
        f"{experiment_name}_log.txt",
        f"{experiment_name}_train.log",
        f"{experiment_name}_training.log",
        "train.log",
        "training.log",
        "log.txt",
        "stdout.log",
    ]
    for name in log_name_candidates:
        p = experiment_path / name
        if p.exists():
            return p
    # fallback: qualunque *.log nella cartella dell'esperimento
    logs = list(experiment_path.glob("*.log"))
    if logs:
        return logs[0]
    return None


# %% [markdown]
# ## Parsing dei log grezzi
#
# I log NON sono strutturati come JSON: si fa parsing testuale con regex
# sui pattern descritti nella spec di estrazione. Il parser e' tollerante
# (ritorna `None`/`np.nan` per i campi che non trova invece di sollevare
# eccezioni) cosi' che un formato di log leggermente diverso non blocchi
# l'intero notebook — i campi mancanti finiscono comunque nella sezione
# "Missing configurations" / vengono segnalati come `NaN`.

# %%
def _find_int(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    if m:
        return int(m.group(1).replace(',', ''))
    return None


def parse_writer_counts(log_text):
    """Estrae il numero di writer per TRAIN/VAL/TEST.
    Pattern atteso: blocchi tipo
        Initializing TRAIN Siamese Dataset
        ...
        Writers: N
    Si assume che il blocco TEST non sia necessariamente etichettato come
    "Siamese Dataset" (spesso e' un blocco di valutazione separato): si
    cerca quindi anche un pattern generico "Writers: N" nei pressi della
    parola TEST/Evaluation, con fallback sul terzo valore "Writers: N"
    trovato nel file se non c'e' un'etichetta esplicita.
    """
    result = {'train_writers': None, 'val_writers': None, 'test_writers': None}

    block_pattern = re.compile(
        r"Initializing\s+(TRAIN|VAL|TEST)\s+Siamese\s+Dataset.*?Writers:\s*(\d+)",
        re.IGNORECASE | re.DOTALL,
    )
    for m in block_pattern.finditer(log_text):
        split_name = m.group(1).upper()
        n_writers = int(m.group(2))
        if split_name == 'TRAIN':
            result['train_writers'] = n_writers
        elif split_name == 'VAL':
            result['val_writers'] = n_writers
        elif split_name == 'TEST':
            result['test_writers'] = n_writers

    if result['test_writers'] is None:
        # fallback: cerca "Writers: N" vicino a TEST/Evaluation/Evaluating
        m = re.search(
            r"(TEST|Evaluat\w*)[^\n]{0,200}?Writers:\s*(\d+)",
            log_text, re.IGNORECASE | re.DOTALL,
        )
        if m:
            result['test_writers'] = int(m.group(2))

    return result


def parse_pair_counts(log_text):
    """Estrae i conteggi di coppie generate/campionate per TRAIN e VAL,
    piu' le coppie effettivamente valutate sul TEST (dal blocco FINAL
    COMPREHENSIVE VALIDATION)."""

    def counts_in_span(span_text):
        return {
            'genuine_pairs': _find_int(r"Generated\s+([\d,]+)\s+genuine\s+pairs", span_text, re.IGNORECASE),
            'impostor_pool': _find_int(r"Generated\s+([\d,]+)\s+total\s+impostor\s+pairs\s*\(pool\)", span_text, re.IGNORECASE),
            'sampled_impostors': _find_int(r"Sampled\s+([\d,]+)\s+impostors\s+from\s+pool\s+of\s+([\d,]+)", span_text, re.IGNORECASE),
        }

    result = {
        'train': {'genuine_pairs': None, 'impostor_pool': None, 'sampled_impostors': None},
        'val': {'genuine_pairs': None, 'impostor_pool': None, 'sampled_impostors': None},
        'test_genuine_evaluated': None,
        'test_impostor_evaluated': None,
    }

    # Split grezzo del log in blocchi TRAIN / VAL usando le stesse ancore
    # del parsing dei writer, cosi' i conteggi di coppie non si mescolano
    # tra train e val.
    train_block = re.search(
        r"Initializing\s+TRAIN\s+Siamese\s+Dataset(.*?)(?=Initializing\s+(VAL|TEST)\s+Siamese\s+Dataset|\Z)",
        log_text, re.IGNORECASE | re.DOTALL,
    )
    val_block = re.search(
        r"Initializing\s+VAL\s+Siamese\s+Dataset(.*?)(?=Initializing\s+(TRAIN|TEST)\s+Siamese\s+Dataset|\Z)",
        log_text, re.IGNORECASE | re.DOTALL,
    )
    if train_block:
        result['train'] = counts_in_span(train_block.group(1))
    if val_block:
        result['val'] = counts_in_span(val_block.group(1))

    m = re.search(
        r"Evaluated\s+([\d,]+)\s+genuine\s*\+\s*([\d,]+)\s+impostor\s+pairs",
        log_text, re.IGNORECASE,
    )
    if m:
        result['test_genuine_evaluated'] = int(m.group(1).replace(',', ''))
        result['test_impostor_evaluated'] = int(m.group(2).replace(',', ''))

    return result


def parse_training_dynamics(log_text):
    """Estrae epoche effettivamente eseguite, miglior val loss, e se e'
    scattato l'early stopping o e' stato raggiunto il cap di 50 epoche."""
    epoch_losses = re.findall(
        r"Train Loss:\s*([\d.]+)\s*\|\s*Val Loss:\s*([\d.]+)",
        log_text, re.IGNORECASE,
    )
    n_epochs_run = len(epoch_losses)
    best_val_loss = min((float(v) for _, v in epoch_losses), default=np.nan)

    early_stopped = bool(re.search(r"early\s*stop", log_text, re.IGNORECASE))
    hit_epoch_cap = (n_epochs_run >= 50) and not early_stopped

    return {
        'n_epochs_run': n_epochs_run if n_epochs_run > 0 else None,
        'best_val_loss': best_val_loss,
        'early_stopped': early_stopped,
        'hit_epoch_cap': hit_epoch_cap,
    }


def parse_log_full(log_path):
    if log_path is None or not log_path.exists():
        return None
    try:
        log_text = log_path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return None

    parsed = {}
    parsed.update(parse_writer_counts(log_text))
    parsed['pairs'] = parse_pair_counts(log_text)
    parsed['dynamics'] = parse_training_dynamics(log_text)
    return parsed


# %% [markdown]
# ## Colonne CSV attese (spec di estrazione) e mappatura
#
# Il CSV `_final_metrics.csv` per-esperimento contiene una riga con queste
# colonne (nomi esatti dalla spec). Le leggiamo cosi' come sono — nessun
# ricalcolo — perche' il CSV e' la source of truth per le Tabelle 2-4.

# %%
CSV_POINT_COLUMNS = [
    'auc', 'eer', 'eer_threshold', 'accuracy', 'precision', 'recall', 'f1',
    'd_prime', 'decidability', 'mu_genuine', 'mu_impostor',
    'sigma_genuine', 'sigma_impostor',
    'far_0.001_effective', 'frr_at_far_0.001', 'gar_at_far_0.001', 'threshold_at_far_0.001',
    'far_0.01_effective', 'frr_at_far_0.01', 'gar_at_far_0.01', 'threshold_at_far_0.01',
]


def load_point_metrics_from_csv(metrics_df):
    """Estrae la riga di metriche puntuali dal CSV, cosi' come salvata
    dallo script di valutazione (nessun ricalcolo). Colonne mancanti nel
    CSV vengono riportate come NaN esplicito, mai inventate."""
    row = metrics_df.iloc[0]
    out = {}
    for col in CSV_POINT_COLUMNS:
        out[col] = row[col] if col in metrics_df.columns else np.nan
    return out


def is_far_degenerate(value):
    """FAR effettivo NaN o chiaramente degenere (es. 0 o negativo, che non
    ha senso come FAR realizzato) -> punto operativo non raggiungibile."""
    if value is None:
        return True
    try:
        v = float(value)
    except (TypeError, ValueError):
        return True
    return np.isnan(v)


# %% [markdown]
# ## Caricamento completo: CSV (metriche + history) + log grezzo, per tutte
# ## le 72 configurazioni (backbone x loss x split)

# %%
all_results = {}          # per plotting/bootstrap (usa genuine_vals/impostor_vals)
point_metrics_csv = {}    # metriche puntuali lette DIRETTAMENTE dal CSV (source of truth)
protocol_by_split = {}    # statistiche di protocollo dai log, raccolte per split
training_dynamics = {}    # dinamiche di training dai log, per (loss, backbone, split)
missing_configs = []      # configurazioni (backbone,loss,split) del tutto assenti
missing_log_configs = []  # configurazioni con CSV presente ma log grezzo assente/non parsabile

for loss in loss_types:
    all_results.setdefault(loss, {})
    point_metrics_csv.setdefault(loss, {})
    training_dynamics.setdefault(loss, {})

    for model_name in MODELS_TO_ANALYZE:
        for dataset_comb in dataset_combinations:
            experiment_path, experiment_name = resolve_experiment_path(loss, model_name, dataset_comb)

            if experiment_path is None:
                missing_configs.append((model_name, loss, dataset_comb))
                continue

            metrics_file = experiment_path / f"{experiment_name}_final_metrics.csv"
            history_file = experiment_path / f"{experiment_name}_history.csv"

            if not (metrics_file.exists() and history_file.exists()):
                missing_configs.append((model_name, loss, dataset_comb))
                continue

            metrics_df = pd.read_csv(metrics_file)
            history_df = pd.read_csv(history_file)

            genuine_vals = np.array(ast.literal_eval(metrics_df['genuine_vals'].values[0]))
            impostor_vals = np.array(ast.literal_eval(metrics_df['impostor_vals'].values[0]))

            all_results[loss].setdefault(model_name, {})[dataset_comb] = {
                'metrics': metrics_df,
                'history': history_df,
                'genuine_vals': genuine_vals,
                'impostor_vals': impostor_vals,
            }

            point_metrics_csv[loss].setdefault(model_name, {})[dataset_comb] = \
                load_point_metrics_from_csv(metrics_df)

            # --- log grezzo (protocollo + dinamiche di training) ---
            log_path = resolve_log_file(experiment_path, experiment_name)
            parsed_log = parse_log_full(log_path)

            if parsed_log is None:
                missing_log_configs.append((model_name, loss, dataset_comb))
            else:
                training_dynamics[loss].setdefault(model_name, {})[dataset_comb] = parsed_log['dynamics']

                # Le statistiche di protocollo dipendono solo dallo split
                # (non da backbone/loss): raccogliamo tutte le osservazioni
                # per split e verifichiamo la coerenza invece di assumerla.
                protocol_by_split.setdefault(dataset_comb, []).append({
                    'source': f"{model_name}_{loss}_{dataset_comb}",
                    'train_writers': parsed_log.get('train_writers'),
                    'val_writers': parsed_log.get('val_writers'),
                    'test_writers': parsed_log.get('test_writers'),
                    'train_genuine_pairs': parsed_log['pairs']['train']['genuine_pairs'],
                    'train_impostor_pool': parsed_log['pairs']['train']['impostor_pool'],
                    'train_sampled_impostors': parsed_log['pairs']['train']['sampled_impostors'],
                    'val_genuine_pairs': parsed_log['pairs']['val']['genuine_pairs'],
                    'val_impostor_pool': parsed_log['pairs']['val']['impostor_pool'],
                    'test_genuine_evaluated': parsed_log['pairs']['test_genuine_evaluated'],
                    'test_impostor_evaluated': parsed_log['pairs']['test_impostor_evaluated'],
                })

n_loaded = sum(len(m) for l in all_results.values() for m in l.values())
print(f"Configurazioni caricate (CSV): {n_loaded} / 72")
print(f"Configurazioni completamente mancanti (CSV assente): {len(missing_configs)}")
print(f"Configurazioni con CSV presente ma log grezzo NON trovato/parsabile: {len(missing_log_configs)}")
if missing_configs:
    print("  Mancanti:", ", ".join(f"{b}/{l}/{s}" for b, l, s in missing_configs[:10]),
          "..." if len(missing_configs) > 10 else "")

# %% [markdown]
# ## Tabella 1 — Protocollo writer/coppie (una riga per split)
#
# Le statistiche dipendono solo dallo split, non da backbone/loss. Se piu'
# log per lo stesso split riportano numeri diversi (dovrebbero essere
# identici, essendo lo split deterministico), viene segnalato un
# disaccordo esplicito invece di prendere silenziosamente il primo valore.

# %%
def summarize_protocol_for_split(dataset_comb):
    obs_list = protocol_by_split.get(dataset_comb, [])
    if not obs_list:
        return None, ["nessun log parsabile trovato per questo split"]

    fields = ['train_writers', 'val_writers', 'test_writers',
              'train_genuine_pairs', 'train_impostor_pool', 'train_sampled_impostors',
              'val_genuine_pairs', 'val_impostor_pool',
              'test_genuine_evaluated', 'test_impostor_evaluated']

    summary = {}
    disagreements = []
    for f in fields:
        values = [o[f] for o in obs_list if o[f] is not None]
        if not values:
            summary[f] = None
            continue
        unique_vals = set(values)
        if len(unique_vals) > 1:
            disagreements.append(f"{f}: valori discordanti tra log {sorted(unique_vals)}")
            summary[f] = None  # non si inventa un valore in caso di disaccordo
        else:
            summary[f] = values[0]
    return summary, disagreements


table1_rows = []
table1_notes = []
for dataset_comb in dataset_combinations:
    summary, disagreements = summarize_protocol_for_split(dataset_comb)
    if summary is None:
        table1_rows.append({
            'split': dataset_comb, 'train_writers': '—', 'val_writers': '—',
            'test_writers': '—', 'test_genuine': '—', 'test_impostor': '—',
        })
        table1_notes.append(f"- **{dataset_comb}**: {disagreements[0]}")
        continue

    def fmt(v):
        return '—' if v is None else str(v)

    table1_rows.append({
        'split': dataset_comb,
        'train_writers': fmt(summary['train_writers']),
        'val_writers': fmt(summary['val_writers']),
        'test_writers': fmt(summary['test_writers']),
        'test_genuine': fmt(summary['test_genuine_evaluated']),
        'test_impostor': fmt(summary['test_impostor_evaluated']),
    })
    if disagreements:
        table1_notes.append(f"- **{dataset_comb}**: " + "; ".join(disagreements))

    # colonne extra (train/val genuine pairs, impostor pool, sampled) tenute
    # a parte in una tabella supplementare piu' estesa (vedi sotto)

table1_df = pd.DataFrame(table1_rows)

table1_md_lines = [
    "| split | train_writers | val_writers | test_writers | test_genuine | test_impostor |",
    "|---|---|---|---|---|---|",
]
for r in table1_rows:
    table1_md_lines.append(
        f"| {r['split']} | {r['train_writers']} | {r['val_writers']} | "
        f"{r['test_writers']} | {r['test_genuine']} | {r['test_impostor']} |"
    )
table1_md = "\n".join(table1_md_lines)

# Tabella supplementare con i conteggi di coppie (utile per lo stesso
# paragrafo di protocollo ma non richiesta esplicitamente nel formato
# minimo della Tabella 1)
table1b_rows = []
for dataset_comb in dataset_combinations:
    summary, _ = summarize_protocol_for_split(dataset_comb)
    if summary is None:
        continue
    def fmt(v):
        return '—' if v is None else str(v)
    table1b_rows.append(
        f"| {dataset_comb} | {fmt(summary['train_genuine_pairs'])} | "
        f"{fmt(summary['train_impostor_pool'])} | {fmt(summary['train_sampled_impostors'])} | "
        f"{fmt(summary['val_genuine_pairs'])} | {fmt(summary['val_impostor_pool'])} |"
    )
table1b_md = "\n".join([
    "| split | train_genuine_pairs | train_impostor_pool | train_sampled_impostors | val_genuine_pairs | val_impostor_pool |",
    "|---|---|---|---|---|---|",
] + table1b_rows)

print("### Tabella 1 — Protocollo writer/coppie\n")
print(table1_md)
print("\n### Tabella 1b — Coppie train/val (supplementare)\n")
print(table1b_md)
if table1_notes:
    print("\n**Note / disaccordi tra log per split:**")
    for n in table1_notes:
        print(n)

with open('./tables/table1_protocol.md', 'w', encoding='utf-8') as f:
    f.write("## Tabella 1 — Protocollo writer/coppie\n\n" + table1_md + "\n\n")
    f.write("## Tabella 1b — Coppie train/val (supplementare)\n\n" + table1b_md + "\n")
    if table1_notes:
        f.write("\n\n### Note / disaccordi\n" + "\n".join(table1_notes) + "\n")

# %% [markdown]
# ## Tabelle 2 & 3 — AUC / EER per backbone x loss, una tabella per split
#
# Valori presi direttamente dal CSV (`point_metrics_csv`), non ricalcolati.
# EER espresso in % (1 decimale), AUC a 3 decimali. `—` dove la
# configurazione manca.

# %%
def collapsed_flag(auc_value):
    if auc_value is None or (isinstance(auc_value, float) and np.isnan(auc_value)):
        return False
    return abs(float(auc_value) - 0.5) < (AUC_COLLAPSE_THRESHOLD - 0.5)


table2_md_blocks = []  # AUC
table3_md_blocks = []  # EER
collapsed_notes = []

for dataset_comb in dataset_combinations:
    header = "| backbone | " + " | ".join(loss_types) + " |"
    sep = "|---|" + "---|" * len(loss_types)

    auc_lines = [f"#### Split: `{dataset_comb}`", "", header, sep]
    eer_lines = [f"#### Split: `{dataset_comb}`", "", header, sep]

    for model_name in MODELS_TO_ANALYZE:
        auc_row_vals, eer_row_vals = [], []
        for loss in loss_types:
            m = point_metrics_csv.get(loss, {}).get(model_name, {}).get(dataset_comb)
            if m is None:
                auc_row_vals.append('—')
                eer_row_vals.append('—')
                continue
            auc_v = m['auc']
            eer_v = m['eer']
            if collapsed_flag(auc_v):
                collapsed_notes.append(f"{model_name}/{loss}/{dataset_comb}: AUC={auc_v:.3f} (collassato, ~chance level)")
                auc_row_vals.append(f"{auc_v:.3f} \u26a0\ufe0f")
            else:
                auc_row_vals.append('—' if (auc_v is None or (isinstance(auc_v, float) and np.isnan(auc_v))) else f"{auc_v:.3f}")
            eer_pct = eer_v * 100 if (eer_v is not None and not (isinstance(eer_v, float) and np.isnan(eer_v))) else None
            eer_row_vals.append('—' if eer_pct is None else f"{eer_pct:.1f}")
        auc_lines.append(f"| {model_name} | " + " | ".join(auc_row_vals) + " |")
        eer_lines.append(f"| {model_name} | " + " | ".join(eer_row_vals) + " |")

    table2_md_blocks.append("\n".join(auc_lines))
    table3_md_blocks.append("\n".join(eer_lines))

table2_md = "\n\n".join(table2_md_blocks)
table3_md = "\n\n".join(table3_md_blocks)

print("### Tabella 2 — AUC per backbone x loss (una tabella per split)\n")
print(table2_md)
print("\n\n### Tabella 3 — EER (%) per backbone x loss (una tabella per split)\n")
print(table3_md)
if collapsed_notes:
    print("\n**\u26a0\ufe0f Modelli con training collassato (AUC ~ chance level):**")
    for n in collapsed_notes:
        print(" -", n)

with open('./tables/table2_auc.md', 'w', encoding='utf-8') as f:
    f.write("## Tabella 2 — AUC per backbone x loss\n\n" + table2_md + "\n")
with open('./tables/table3_eer.md', 'w', encoding='utf-8') as f:
    f.write("## Tabella 3 — EER (%) per backbone x loss\n\n" + table3_md + "\n")

# %% [markdown]
# ## Tabella 4 — Metriche complete per esperimento (long format)
#
# Una riga per (backbone, loss, split), letta dal CSV. `far0.1pct_valid` =
# "no" se `far_0.001_effective` e' NaN/degenere per quella riga. Colonna
# `notes` aggiuntiva per segnalare i training collassati (non richiesta nel
# formato minimo, ma esplicitamente richiesta a parte dalla regola 5 della
# spec: viene qui integrata come colonna invece che in prosa sparsa).

# %%
table4_header = (
    "| backbone | loss | split | auc | eer | d_prime | accuracy | precision | recall | f1 | "
    "mu_genuine | mu_impostor | sigma_genuine | sigma_impostor | gar_far1pct | gar_far0.1pct | "
    "far0.1pct_valid | notes |"
)
table4_sep = "|" + "---|" * 18

table4_rows_md = []
table4_rows_records = []

for loss in loss_types:
    for model_name in MODELS_TO_ANALYZE:
        for dataset_comb in dataset_combinations:
            m = point_metrics_csv.get(loss, {}).get(model_name, {}).get(dataset_comb)
            if m is None:
                continue  # gia' elencato in "Missing configurations"

            def f3(v):
                return '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.3f}"

            def fpct(v):
                return '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v) * 100:.1f}"

            far01_valid = not is_far_degenerate(m.get('far_0.001_effective'))
            notes = ''
            if collapsed_flag(m['auc']):
                notes = 'COLLASSATO (AUC~chance)'

            row_md = (
                f"| {model_name} | {loss} | {dataset_comb} | {f3(m['auc'])} | {fpct(m['eer'])} | "
                f"{f3(m['d_prime'])} | {f3(m['accuracy'])} | {f3(m['precision'])} | {f3(m['recall'])} | "
                f"{f3(m['f1'])} | {f3(m['mu_genuine'])} | {f3(m['mu_impostor'])} | "
                f"{f3(m['sigma_genuine'])} | {f3(m['sigma_impostor'])} | "
                f"{fpct(m.get('gar_at_far_0.01'))} | {fpct(m.get('gar_at_far_0.001'))} | "
                f"{'yes' if far01_valid else 'no'} | {notes} |"
            )
            table4_rows_md.append(row_md)
            table4_rows_records.append({
                'backbone': model_name, 'loss': loss, 'split': dataset_comb,
                'auc': m['auc'], 'eer': m['eer'], 'd_prime': m['d_prime'],
                'accuracy': m['accuracy'], 'precision': m['precision'], 'recall': m['recall'], 'f1': m['f1'],
                'mu_genuine': m['mu_genuine'], 'mu_impostor': m['mu_impostor'],
                'sigma_genuine': m['sigma_genuine'], 'sigma_impostor': m['sigma_impostor'],
                'gar_far1pct': m.get('gar_at_far_0.01'), 'gar_far0.1pct': m.get('gar_at_far_0.001'),
                'far0.1pct_valid': far01_valid, 'notes': notes,
            })

table4_md = "\n".join([table4_header, table4_sep] + table4_rows_md)
table4_df = pd.DataFrame(table4_rows_records)
table4_df.to_csv('./tables/table4_full_export.csv', index=False)

print(f"### Tabella 4 — Metriche complete per esperimento ({len(table4_rows_md)} righe)\n")
print(table4_md)

with open('./tables/table4_full.md', 'w', encoding='utf-8') as f:
    f.write("## Tabella 4 — Metriche complete per esperimento\n\n" + table4_md + "\n")

# %% [markdown]
# ## Tabella 5 — Aggregazione same-dataset vs cross-dataset (mean +/- std)
#
# Calcolata SOLO se tutti e 6 i backbone sono presenti per quella
# combinazione (loss, split); altrimenti si riportano esplicitamente i
# backbone mancanti invece di una media parziale.

# %%
def aggregate_strict(dataset_comb):
    rows = []
    for loss in loss_types:
        present = {
            model_name: point_metrics_csv.get(loss, {}).get(model_name, {}).get(dataset_comb)
            for model_name in MODELS_TO_ANALYZE
        }
        available = {k: v for k, v in present.items() if v is not None}
        missing_backbones = [k for k, v in present.items() if v is None]

        if len(available) < len(MODELS_TO_ANALYZE):
            rows.append({
                'loss': loss, 'split': dataset_comb,
                'status': 'INCOMPLETO',
                'missing_backbones': ", ".join(missing_backbones) if missing_backbones else '—',
                'auc_mean': None, 'auc_std': None, 'eer_mean': None, 'eer_std': None,
            })
            continue

        auc_vals = np.array([v['auc'] for v in available.values()], dtype=float)
        eer_vals = np.array([v['eer'] for v in available.values()], dtype=float) * 100  # in %

        rows.append({
            'loss': loss, 'split': dataset_comb, 'status': 'OK',
            'missing_backbones': '—',
            'auc_mean': np.nanmean(auc_vals), 'auc_std': np.nanstd(auc_vals, ddof=1),
            'eer_mean': np.nanmean(eer_vals), 'eer_std': np.nanstd(eer_vals, ddof=1),
        })
    return rows


table5_records = []
for dataset_comb in dataset_combinations:
    table5_records.extend(aggregate_strict(dataset_comb))

table5_lines = [
    "| loss | split | auc_mean | auc_std | eer_mean | eer_std |",
    "|---|---|---|---|---|---|",
]
table5_incomplete_notes = []
for r in table5_records:
    if r['status'] == 'OK':
        table5_lines.append(
            f"| {r['loss']} | {r['split']} | {r['auc_mean']:.3f} | {r['auc_std']:.3f} | "
            f"{r['eer_mean']:.1f} | {r['eer_std']:.1f} |"
        )
    else:
        table5_lines.append(f"| {r['loss']} | {r['split']} | — | — | — | — |")
        table5_incomplete_notes.append(
            f"- **{r['loss']}/{r['split']}**: backbone mancanti -> {r['missing_backbones']}"
        )

table5_md = "\n".join(table5_lines)

print("### Tabella 5 — Aggregazione same/cross-dataset (solo se 6/6 backbone presenti)\n")
print(table5_md)
if table5_incomplete_notes:
    print("\n**Combinazioni (loss, split) incomplete — nessuna media calcolata:**")
    for n in table5_incomplete_notes:
        print(n)

with open('./tables/table5_aggregated.md', 'w', encoding='utf-8') as f:
    f.write("## Tabella 5 — Aggregazione same/cross-dataset\n\n" + table5_md + "\n")
    if table5_incomplete_notes:
        f.write("\n### Combinazioni incomplete\n" + "\n".join(table5_incomplete_notes) + "\n")

# %% [markdown]
# ## Sezione — Missing configurations (esplicita, mai omessa)

# %%
missing_lines = ["### Missing configurations\n"]
if not missing_configs and not missing_log_configs:
    missing_lines.append("Nessuna configurazione mancante: 72/72 CSV e log trovati e parsati.")
else:
    if missing_configs:
        missing_lines.append(f"**CSV/esperimento del tutto assente ({len(missing_configs)}):**")
        for b, l, s in missing_configs:
            missing_lines.append(f"- {b} / {l} / {s}")
    if missing_log_configs:
        missing_lines.append(f"\n**CSV presente ma log grezzo non trovato/parsabile ({len(missing_log_configs)}):**")
        missing_lines.append("(per queste configurazioni le metriche puntuali in Tabelle 2-4 sono comunque disponibili; "
                              "mancano solo i dati di protocollo/training-dynamics derivati dal log)")
        for b, l, s in missing_log_configs:
            missing_lines.append(f"- {b} / {l} / {s}")

missing_md = "\n".join(missing_lines)
print(missing_md)
with open('./tables/missing_configurations.md', 'w', encoding='utf-8') as f:
    f.write(missing_md + "\n")

# %% [markdown]
# ## Training dynamics (Tabella C, opzionale) — epoche, best val loss, early stopping
#
# Estratta dai log, per completezza (richiesta come "optional/only if
# asked" nella spec).

# %%
table_dyn_rows = []
for loss in loss_types:
    for model_name in MODELS_TO_ANALYZE:
        for dataset_comb in dataset_combinations:
            d = training_dynamics.get(loss, {}).get(model_name, {}).get(dataset_comb)
            if d is None:
                continue
            status = 'early_stopped' if d['early_stopped'] else ('epoch_cap' if d['hit_epoch_cap'] else 'unknown')
            table_dyn_rows.append({
                'backbone': model_name, 'loss': loss, 'split': dataset_comb,
                'n_epochs_run': d['n_epochs_run'], 'best_val_loss': d['best_val_loss'],
                'stop_reason': status,
            })

table_dyn_df = pd.DataFrame(table_dyn_rows)
table_dyn_df.to_csv('./tables/training_dynamics.csv', index=False)
print(f"Training dynamics estratte per {len(table_dyn_df)} configurazioni (vedi tables/training_dynamics.csv)")
if not table_dyn_df.empty:
    print(table_dyn_df.head(10).to_string(index=False))

# %% [markdown]
# ---
# # Da qui in poi: analisi completa gia' presente nella versione precedente
# ---
# Bootstrap CI, controllo normalita', ROC/DET, aggregati same/cross con
# statistiche t, heatmap, Wilcoxon appaiato, confondimenti noti, confronto
# SOTA, failure analysis, DET aggregata. Tutta questa parte usa i vettori
# genuine_vals/impostor_vals gia' caricati in `all_results` e RICALCOLA le
# metriche (necessario per bootstrap CI e per i plot, che richiedono le
# curve fpr/tpr complete — non presenti come tali nella Tabella 4 sopra).

# %% [markdown]
# ## Metriche biometriche (ricalcolate su score raw) + Bootstrap CI

# %%
def calculate_biometric_metrics(genuine_scores, impostor_scores):
    mu_genuine = np.mean(genuine_scores)
    mu_impostor = np.mean(impostor_scores)
    sigma_genuine = np.std(genuine_scores, ddof=1)
    sigma_impostor = np.std(impostor_scores, ddof=1)

    pooled_std = np.sqrt(0.5 * (sigma_genuine**2 + sigma_impostor**2))
    d_prime = (mu_genuine - mu_impostor) / pooled_std if pooled_std > 0 else 0.0
    decidability = abs(d_prime)

    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    y_scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    def operating_point(far_target):
        idxs = np.where(fpr <= far_target)[0]
        if len(idxs) == 0:
            return dict(far_eff=np.nan, frr=np.nan, gar=np.nan, thr=np.nan)
        idx = idxs[-1]
        return dict(far_eff=fpr[idx], frr=fnr[idx], gar=tpr[idx], thr=thresholds[idx])

    op_0001 = operating_point(0.0001)
    op_001 = operating_point(0.001)
    op_01 = operating_point(0.01)

    n_impostor_local = len(impostor_scores)
    op_0001_valid = n_impostor_local >= int(1 / 0.0001)
    if not op_0001_valid:
        op_0001 = dict(far_eff=np.nan, frr=np.nan, gar=np.nan, thr=np.nan)

    predictions = (y_scores >= eer_threshold).astype(int)
    accuracy = np.mean(predictions == y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'auc': roc_auc, 'eer': eer, 'eer_threshold': eer_threshold,
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
        'd_prime': d_prime, 'decidability': decidability,
        'mu_genuine': mu_genuine, 'mu_impostor': mu_impostor,
        'sigma_genuine': sigma_genuine, 'sigma_impostor': sigma_impostor,
        'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds,
        'far_0001_effective': op_0001['far_eff'], 'frr_at_far_0001': op_0001['frr'],
        'gar_at_far_0001': op_0001['gar'], 'threshold_at_far_0001': op_0001['thr'],
        'far_0001_valid': op_0001_valid,
        'far_001_effective': op_001['far_eff'], 'frr_at_far_001': op_001['frr'],
        'gar_at_far_001': op_001['gar'], 'threshold_at_far_001': op_001['thr'],
        'far_01_effective': op_01['far_eff'], 'frr_at_far_01': op_01['frr'],
        'gar_at_far_01': op_01['gar'], 'threshold_at_far_01': op_01['thr'],
    }


def bootstrap_ci(genuine, impostor, metric='auc', n_boot=N_BOOTSTRAP, ci=CI_LEVEL, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n_g, n_i = len(genuine), len(impostor)
    boot_vals = []
    for _ in range(n_boot):
        g_s = rng.choice(genuine, size=n_g, replace=True)
        i_s = rng.choice(impostor, size=n_i, replace=True)
        y_true = np.concatenate([np.ones(n_g), np.zeros(n_i)])
        y_scores = np.concatenate([g_s, i_s])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        if metric == 'auc':
            boot_vals.append(auc(fpr, tpr))
        else:
            fnr = 1 - tpr
            idx = np.nanargmin(np.abs(fnr - fpr))
            boot_vals.append((fpr[idx] + fnr[idx]) / 2)
    boot_vals = np.array(boot_vals)
    alpha = (100 - ci) / 2
    lower, upper = np.percentile(boot_vals, [alpha, 100 - alpha])
    point = np.mean(boot_vals)
    return point, lower, upper


# %%
computed_metrics = {}
for loss in loss_types:
    computed_metrics[loss] = {}
    for model_name, splits in all_results.get(loss, {}).items():
        computed_metrics[loss][model_name] = {}
        for dataset_comb, data in splits.items():
            genuine = data['genuine_vals']
            impostor = data['impostor_vals']
            m = calculate_biometric_metrics(genuine, impostor)
            _, auc_lo, auc_hi = bootstrap_ci(genuine, impostor, metric='auc')
            _, eer_lo, eer_hi = bootstrap_ci(genuine, impostor, metric='eer')
            m['auc_ci_low'], m['auc_ci_high'] = auc_lo, auc_hi
            m['eer_ci_low'], m['eer_ci_high'] = eer_lo, eer_hi
            computed_metrics[loss][model_name][dataset_comb] = m

print("Esempio con CI (loss, modello, split -> AUC [CI95%], EER [CI95%]):")
for loss in loss_types:
    for model_name in list(computed_metrics[loss].keys())[:1]:
        for dataset_comb, m in computed_metrics[loss][model_name].items():
            print(f"  {loss}/{model_name}/{dataset_comb}: "
                  f"AUC={m['auc']:.3f} [{m['auc_ci_low']:.3f}-{m['auc_ci_high']:.3f}]  "
                  f"EER={m['eer']:.3f} [{m['eer_ci_low']:.3f}-{m['eer_ci_high']:.3f}]")

# %% [markdown]
# ## Controllo di normalita' delle distribuzioni genuine/impostor

# %%
def normality_check(genuine, impostor, max_n=5000, seed=42):
    rng = np.random.default_rng(seed)
    g = genuine if len(genuine) <= max_n else rng.choice(genuine, max_n, replace=False)
    i = impostor if len(impostor) <= max_n else rng.choice(impostor, max_n, replace=False)
    _, p_g = stats.shapiro(g)
    _, p_i = stats.shapiro(i)
    return p_g, p_i


normality_rows = []
for loss in loss_types:
    for model_name, splits in computed_metrics[loss].items():
        for dataset_comb in splits:
            genuine = all_results[loss][model_name][dataset_comb]['genuine_vals']
            impostor = all_results[loss][model_name][dataset_comb]['impostor_vals']
            p_g, p_i = normality_check(genuine, impostor)
            normality_rows.append({
                'loss': loss, 'model': model_name, 'split': dataset_comb,
                'shapiro_p_genuine': p_g, 'shapiro_p_impostor': p_i,
                'genuine_normal_a5pct': p_g > 0.05,
                'impostor_normal_a5pct': p_i > 0.05,
            })

normality_df = pd.DataFrame(normality_rows)
normality_df.to_csv('./normality_check.csv', index=False)
pct_normal_genuine = normality_df['genuine_normal_a5pct'].mean() * 100
pct_normal_impostor = normality_df['impostor_normal_a5pct'].mean() * 100
print(f"Distribuzioni genuine compatibili con normalita' (Shapiro-Wilk, alpha=0.05): {pct_normal_genuine:.1f}%")
print(f"Distribuzioni impostor compatibili con normalita' (Shapiro-Wilk, alpha=0.05): {pct_normal_impostor:.1f}%")

# %% [markdown]
# ## Distribuzioni degli score (con overlay gaussiano) — modello rappresentativo

# %%
def plot_score_distributions(model_name):
    fig, axes = plt.subplots(len(loss_types), len(dataset_combinations),
                              figsize=(20, 4 * len(loss_types)))
    for i, loss in enumerate(loss_types):
        for j, dataset_comb in enumerate(dataset_combinations):
            ax = axes[i, j]
            data = all_results.get(loss, {}).get(model_name, {}).get(dataset_comb)
            if data is None:
                ax.axis('off')
                continue
            genuine = data['genuine_vals']
            impostor = data['impostor_vals']
            m = computed_metrics[loss][model_name][dataset_comb]
            ax.hist(impostor, bins=50, alpha=0.5, label='Impostor', color='red', density=True)
            ax.hist(genuine, bins=50, alpha=0.5, label='Genuine', color='green', density=True)
            x_range = np.linspace(min(impostor.min(), genuine.min()),
                                   max(impostor.max(), genuine.max()), 200)
            ax.plot(x_range, stats.norm.pdf(x_range, m['mu_genuine'], m['sigma_genuine']),
                    color='darkgreen', linewidth=1.5)
            ax.plot(x_range, stats.norm.pdf(x_range, m['mu_impostor'], m['sigma_impostor']),
                    color='darkred', linewidth=1.5)
            ax.axvline(m['eer_threshold'], color='blue', linestyle=':', linewidth=2,
                       label=f"soglia EER={m['eer_threshold']:.3f}")
            train_ds, test_ds = dataset_comb.split('_to_')
            ax.set_title(f"{loss.upper()} | {train_ds.upper()}\u2192{test_ds.upper()}\n"
                         f"EER={m['eer']:.3f} [{m['eer_ci_low']:.3f}-{m['eer_ci_high']:.3f}]  "
                         f"d'={m['d_prime']:.2f}", fontsize=10, fontweight='bold')
            ax.set_xlabel('Score (similarita\u0300)', fontsize=9)
            ax.set_ylabel('Densita\u0300', fontsize=9)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)
    fig.suptitle(f"Distribuzioni genuine/impostor con fit gaussiano \u2014 backbone: {model_name}",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f'./images/score_distributions_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()


if REPRESENTATIVE_MODEL in computed_metrics.get('bce', {}):
    plot_score_distributions(REPRESENTATIVE_MODEL)
else:
    print(f"Modello rappresentativo '{REPRESENTATIVE_MODEL}' non trovato, salto il plot dettagliato.")

# %% [markdown]
# ## ROC curve — modello rappresentativo

# %%
def plot_roc_curves(model_name):
    fig, axes = plt.subplots(len(loss_types), len(dataset_combinations),
                              figsize=(20, 4 * len(loss_types)))
    for i, loss in enumerate(loss_types):
        for j, dataset_comb in enumerate(dataset_combinations):
            ax = axes[i, j]
            m = computed_metrics.get(loss, {}).get(model_name, {}).get(dataset_comb)
            if m is None:
                ax.axis('off')
                continue
            ax.plot(m['fpr'], m['tpr'], linewidth=2,
                    label=f"AUC={m['auc']:.3f} [{m['auc_ci_low']:.3f}-{m['auc_ci_high']:.3f}]")
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
            eer_idx = np.nanargmin(np.abs((1 - m['tpr']) - m['fpr']))
            ax.plot(m['fpr'][eer_idx], m['tpr'][eer_idx], 'ro', markersize=7,
                    label=f"EER={m['eer']:.3f}")
            train_ds, test_ds = dataset_comb.split('_to_')
            ax.set_title(f"{loss.upper()} | {train_ds.upper()}\u2192{test_ds.upper()}",
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('FAR', fontsize=9)
            ax.set_ylabel('1 - FRR', fontsize=9)
            ax.legend(fontsize=8, loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
    fig.suptitle(f"Curve ROC \u2014 backbone: {model_name}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f'./images/roc_curves_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()


if REPRESENTATIVE_MODEL in computed_metrics.get('bce', {}):
    plot_roc_curves(REPRESENTATIVE_MODEL)

# %% [markdown]
# ## DET curve (log-log) — modello rappresentativo

# %%
def plot_det_curves(model_name):
    fig, axes = plt.subplots(len(loss_types), len(dataset_combinations),
                              figsize=(20, 4 * len(loss_types)))
    for i, loss in enumerate(loss_types):
        for j, dataset_comb in enumerate(dataset_combinations):
            ax = axes[i, j]
            m = computed_metrics.get(loss, {}).get(model_name, {}).get(dataset_comb)
            if m is None:
                ax.axis('off')
                continue
            frr = 1 - m['tpr']
            ax.loglog(m['fpr'], frr, linewidth=2)
            eer_idx = np.nanargmin(np.abs(frr - m['fpr']))
            ax.plot(m['fpr'][eer_idx], frr[eer_idx], 'ro', markersize=7,
                    label=f"EER={m['eer']:.3f}")
            ax.plot([1e-4, 1], [1e-4, 1], 'k--', linewidth=1, alpha=0.5)
            train_ds, test_ds = dataset_comb.split('_to_')
            ax.set_title(f"{loss.upper()} | {train_ds.upper()}\u2192{test_ds.upper()}",
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('FAR (log)', fontsize=9)
            ax.set_ylabel('FRR (log)', fontsize=9)
            ax.legend(fontsize=8, loc='lower left')
            ax.grid(True, alpha=0.3, which='both')
            ax.set_xlim([1e-4, 1])
            ax.set_ylim([1e-4, 1])
    fig.suptitle(f"Curve DET \u2014 backbone: {model_name}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f'./images/det_curves_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()


if REPRESENTATIVE_MODEL in computed_metrics.get('bce', {}):
    plot_det_curves(REPRESENTATIVE_MODEL)

# %% [markdown]
# ## Training history — modello rappresentativo

# %%
def plot_training_history(model_name):
    fig, axes = plt.subplots(len(loss_types), len(dataset_combinations),
                              figsize=(20, 4 * len(loss_types)))
    for i, loss in enumerate(loss_types):
        for j, dataset_comb in enumerate(dataset_combinations):
            ax = axes[i, j]
            data = all_results.get(loss, {}).get(model_name, {}).get(dataset_comb)
            if data is None:
                ax.axis('off')
                continue
            history = data['history']
            ax.plot(history['epoch'], history['train_loss'], linewidth=2, label='Train', marker='o', markersize=3)
            ax.plot(history['epoch'], history['val_loss'], linewidth=2, label='Validation', marker='s', markersize=3)
            train_ds, test_ds = dataset_comb.split('_to_')
            ax.set_title(f"{loss.upper()} | {train_ds.upper()}\u2192{test_ds.upper()}",
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('Epoca', fontsize=9)
            ax.set_ylabel('Loss', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    fig.suptitle(f"Curve di training \u2014 backbone: {model_name}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f'./images/training_history_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()


if REPRESENTATIVE_MODEL in computed_metrics.get('bce', {}):
    plot_training_history(REPRESENTATIVE_MODEL)

# %% [markdown]
# ## Tabelle aggregate Same-dataset / Cross-dataset (bootstrap-based, tutti i backbone)
#
# Complementari alla Tabella 5 "rigorosa" sopra: qui l'aggregazione usa la
# t-distribuzione su tutti i backbone disponibili (anche se < 6), utile
# come vista aggiuntiva ma NON sostituisce la Tabella 5 richiesta nella
# spec, che invece impone tutti e 6 i backbone.

# %%
def aggregate_across_models(splits_to_include, metric_keys):
    rows = []
    for loss in loss_types:
        for dataset_comb in splits_to_include:
            vals = {k: [] for k in metric_keys}
            for model_name in computed_metrics.get(loss, {}):
                m = computed_metrics[loss][model_name].get(dataset_comb)
                if m is None:
                    continue
                for k in metric_keys:
                    vals[k].append(m[k])
            n = len(vals[metric_keys[0]])
            if n == 0:
                continue
            row = {'loss': loss.upper(), 'split': dataset_comb, 'n_backbones': n}
            for k in metric_keys:
                arr = np.array(vals[k])
                mean = arr.mean()
                std = arr.std(ddof=1) if n > 1 else 0.0
                se = std / np.sqrt(n) if n > 1 else 0.0
                tcrit = stats.t.ppf(0.975, df=n - 1) if n > 1 else 0.0
                row[f'{k}_mean'] = mean
                row[f'{k}_std'] = std
                row[f'{k}_ci95_low'] = mean - tcrit * se
                row[f'{k}_ci95_high'] = mean + tcrit * se
            rows.append(row)
    return pd.DataFrame(rows)


metric_keys = ['auc', 'eer', 'd_prime', 'decidability', 'accuracy', 'f1',
               'gar_at_far_001', 'gar_at_far_01']

same_agg = aggregate_across_models(['iam_to_iam', 'rimes_to_rimes'], metric_keys)
cross_agg = aggregate_across_models(['iam_to_rimes', 'rimes_to_iam'], metric_keys)
same_agg.to_csv('./same_dataset_aggregated.csv', index=False)
cross_agg.to_csv('./cross_dataset_aggregated.csv', index=False)

print("SAME-DATASET (bootstrap-based, n=n. backbone disponibili):")
print(same_agg[['loss', 'split', 'n_backbones', 'auc_mean', 'auc_std', 'eer_mean', 'eer_std', 'd_prime_mean']].round(3).to_string(index=False))
print("\nCROSS-DATASET (bootstrap-based, n=n. backbone disponibili):")
print(cross_agg[['loss', 'split', 'n_backbones', 'auc_mean', 'auc_std', 'eer_mean', 'eer_std', 'd_prime_mean']].round(3).to_string(index=False))

# %% [markdown]
# ## Same vs Cross dataset — confronto aggregato (grafico)

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics_to_compare = ['auc', 'eer', 'd_prime']
titles = ['AUC (\u2191 meglio)', 'EER (\u2193 meglio)', "d' (\u2191 meglio)"]

for idx, (metric, title) in enumerate(zip(metrics_to_compare, titles)):
    ax = axes[idx]
    x_pos = 0
    x_ticks, x_labels = [], []
    for loss in loss_types:
        same_vals_all, cross_vals_all = [], []
        for model_name in computed_metrics.get(loss, {}):
            for ds in ['iam_to_iam', 'rimes_to_rimes']:
                m = computed_metrics[loss][model_name].get(ds)
                if m is not None:
                    same_vals_all.append(m[metric])
            for ds in ['iam_to_rimes', 'rimes_to_iam']:
                m = computed_metrics[loss][model_name].get(ds)
                if m is not None:
                    cross_vals_all.append(m[metric])
        if same_vals_all:
            ax.bar(x_pos, np.mean(same_vals_all), width=0.35,
                   label='Same-dataset' if x_pos == 0 else '', color='steelblue', alpha=0.85)
            ax.errorbar(x_pos, np.mean(same_vals_all), yerr=np.std(same_vals_all, ddof=1),
                        fmt='none', color='black', capsize=5)
        if cross_vals_all:
            ax.bar(x_pos + 0.4, np.mean(cross_vals_all), width=0.35,
                   label='Cross-dataset' if x_pos == 0 else '', color='coral', alpha=0.85)
            ax.errorbar(x_pos + 0.4, np.mean(cross_vals_all), yerr=np.std(cross_vals_all, ddof=1),
                        fmt='none', color='black', capsize=5)
        x_labels.append(loss.upper())
        x_ticks.append(x_pos + 0.2)
        x_pos += 1
    ax.set_xlabel('Loss function', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle("Confronto Same- vs Cross-dataset (barre = std tra backbone)", fontsize=12, y=1.03)
plt.tight_layout()
plt.savefig('./images/same_vs_cross_comparison_aggregated.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Heatmap AUC / EER (media sui backbone)

# %%
auc_matrix = np.full((len(loss_types), len(dataset_combinations)), np.nan)
eer_matrix = np.full((len(loss_types), len(dataset_combinations)), np.nan)
auc_std_matrix = np.full((len(loss_types), len(dataset_combinations)), np.nan)

for i, loss in enumerate(loss_types):
    for j, dataset_comb in enumerate(dataset_combinations):
        vals = [computed_metrics[loss][model_name][dataset_comb]['auc']
                for model_name in computed_metrics.get(loss, {})
                if dataset_comb in computed_metrics[loss][model_name]]
        eer_vals = [computed_metrics[loss][model_name][dataset_comb]['eer']
                    for model_name in computed_metrics.get(loss, {})
                    if dataset_comb in computed_metrics[loss][model_name]]
        if vals:
            auc_matrix[i, j] = np.mean(vals)
            auc_std_matrix[i, j] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            eer_matrix[i, j] = np.mean(eer_vals)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
labels_x = [dc.replace('_to_', '\u2192').upper() for dc in dataset_combinations]
labels_y = [l.upper() for l in loss_types]
annot_auc = np.array([[f"{auc_matrix[i,j]:.3f}\n\u00b1{auc_std_matrix[i,j]:.3f}"
                        for j in range(len(dataset_combinations))]
                       for i in range(len(loss_types))])

sns.heatmap(auc_matrix, annot=annot_auc, fmt='', cmap='RdYlGn',
            xticklabels=labels_x, yticklabels=labels_y,
            ax=axes[0], cbar_kws={'label': 'AUC (media sui backbone)'}, vmin=0.5, vmax=1.0)
axes[0].set_title("AUC \u2014 media \u00b1 std sui backbone", fontsize=13, fontweight='bold')

sns.heatmap(eer_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
            xticklabels=labels_x, yticklabels=labels_y,
            ax=axes[1], cbar_kws={'label': 'EER (media sui backbone)'}, vmin=0.0, vmax=0.5)
axes[1].set_title("EER \u2014 media sui backbone", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('./images/metrics_heatmap_aggregated.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Test di Wilcoxon a coppie appaiate tra loss (su AUC)

# %%
def build_paired_auc(loss_a, loss_b):
    pairs = []
    for model_name in computed_metrics.get(loss_a, {}):
        for dataset_comb in computed_metrics[loss_a][model_name]:
            m_b = computed_metrics.get(loss_b, {}).get(model_name, {}).get(dataset_comb)
            if m_b is not None:
                pairs.append((computed_metrics[loss_a][model_name][dataset_comb]['auc'], m_b['auc']))
    if not pairs:
        return np.array([]), np.array([])
    x, y = zip(*pairs)
    return np.array(x), np.array(y)


def wilcoxon_with_effect(x, y):
    diff = x - y
    diff_nz = diff[diff != 0]
    if len(diff_nz) < 4:
        return dict(n=len(diff), n_nonzero=len(diff_nz), statistic=np.nan, p_value=np.nan, effect_size=np.nan)
    ranks = stats.rankdata(np.abs(diff_nz))
    r_plus = ranks[diff_nz > 0].sum()
    r_minus = ranks[diff_nz < 0].sum()
    stat, p = stats.wilcoxon(x, y)
    effect = (r_plus - r_minus) / (r_plus + r_minus)
    return dict(n=len(diff), n_nonzero=len(diff_nz), statistic=stat, p_value=p, effect_size=effect)


wilcoxon_rows = []
for loss_a, loss_b in combinations(loss_types, 2):
    x, y = build_paired_auc(loss_a, loss_b)
    if len(x) == 0:
        continue
    res = wilcoxon_with_effect(x, y)
    res['loss_a'] = loss_a
    res['loss_b'] = loss_b
    res['mean_auc_a'] = x.mean()
    res['mean_auc_b'] = y.mean()
    wilcoxon_rows.append(res)
    sig = "significativo (p<0.05)" if (not np.isnan(res['p_value']) and res['p_value'] < 0.05) else "non significativo"
    print(f"{loss_a.upper()} vs {loss_b.upper()} (n={res['n']}, n_diff\u22600={res['n_nonzero']}): "
          f"AUC {res['mean_auc_a']:.3f} vs {res['mean_auc_b']:.3f}, "
          f"W={res['statistic']:.2f} p={res['p_value']:.4f} effect={res['effect_size']:.3f} -> {sig}")

wilcoxon_df = pd.DataFrame(wilcoxon_rows)
wilcoxon_df.to_csv('./wilcoxon_loss_comparison.csv', index=False)

# %% [markdown]
# ## Confondimenti noti tra le configurazioni di training (Limitations)

# %%
KNOWN_CONFOUNDS = """
CONFONDIMENTI NOTI TRA LE CONFIGURAZIONI DI TRAINING (BCE/Contrastive/Triplet):

1. embedding_dim: BCE=32, Contrastive=128, Triplet=128
2. frozen_backbone_layers: BCE=0 (full fine-tuning), Contrastive=3, Triplet=3
3. batch_size: BCE=16, Contrastive=32, Triplet=16 (con cardinalita' di
   sample per epoca diversa tra Triplet e BCE/Contrastive)
4. dropout: BCE=0.2, Contrastive=0.4, Triplet=0.2

Le differenze di performance tra loss NON possono essere attribuite alla
sola natura della loss finche' questi iperparametri non vengono uniformati.
Nessuna ripetizione multi-seed: le CI bootstrap sono sugli score di un
singolo run, non sulla varianza di training. mu/sigma non sono
confrontabili in valore assoluto tra loss diverse (probabilita' sigmoid
per BCE vs cosine similarity per contrastive/triplet); solo d_prime,
decidability, AUC, EER, FAR/FRR sono confrontabili cross-loss.
"""
print(KNOWN_CONFOUNDS)
with open('./known_confounds.txt', 'w', encoding='utf-8') as f:
    f.write(KNOWN_CONFOUNDS)

# %% [markdown]
# ## Riepilogo file generati

# %%
print("""
File generati (Results):
  tables/table1_protocol.md         -> Tabella 1 (+1b supplementare)
  tables/table2_auc.md              -> Tabella 2 (AUC per split)
  tables/table3_eer.md              -> Tabella 3 (EER per split)
  tables/table4_full.md             -> Tabella 4 (long format)
  tables/table4_full_export.csv     -> Tabella 4 in CSV
  tables/table5_aggregated.md       -> Tabella 5 (mean+/-std, solo 6/6 backbone)
  tables/missing_configurations.md  -> Configurazioni mancanti (CSV/log)
  tables/training_dynamics.csv      -> Tabella C (epoche/early stopping)
  same_dataset_aggregated.csv       -> Aggregato bootstrap-based (supplementare)
  cross_dataset_aggregated.csv      -> Aggregato bootstrap-based (supplementare)
  wilcoxon_loss_comparison.csv      -> Test di significativita'
  known_confounds.txt               -> Blocco Limitations
  images/*.png                      -> Tutti i plot (distribuzioni, ROC, DET, training, heatmap)
""")