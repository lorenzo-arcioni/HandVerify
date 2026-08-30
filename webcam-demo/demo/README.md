# Demo webcam — verifica della scrittura a mano

Demo live del modello **EfficientNet-B1 triplet** addestrato su IAM: si
inquadrano due scritture con la webcam e il sistema dice se sono della stessa
persona.

## File

| file | cosa fa |
|---|---|
| `webcam_demo.py` | demo live |
| `engine.py` | caricamento modello + cosine similarity + soglie |
| `model_def.py` | definizione stand-alone della rete |
| `preprocess.py` | preprocessing IAM-like adattato alla webcam + diagnostica |

Il file dei pesi `efficientnet_b1_triplet_iam_to_iam_best.pth` (33 MB) sta **un
livello sopra** la cartella `demo/`: è lì che `engine.py` lo cerca. Se lo sposti,
passa il percorso con `--checkpoint`.

```
webcam-demo/
├── efficientnet_b1_triplet_iam_to_iam_best.pth
└── demo/
    ├── webcam_demo.py
    ├── engine.py
    ├── model_def.py
    ├── preprocess.py
    └── README.md
```

## Installazione

Serve Python 3.10+ con quattro pacchetti:

```bash
pip install torch torchvision opencv-python numpy
```

La CPU basta: un confronto richiede circa 100 ms, non serve GPU. Al primo avvio
non viene scaricato nulla — la rete si costruisce vuota e i pesi arrivano dal
file `.pth`.

## Lancio

Dalla cartella `demo/`:

```bash
python webcam_demo.py
```

oppure

```bash
python webcam_demo.py --camera 1 --checkpoint ../../results/contrastive/contrastive_experiments/resnet18_contrastive_rimes_to_iam/resnet18_contrastive_rimes_to_iam_final.pth --threshold far01
```

I tasti funzionano sia maiuscoli sia minuscoli, e la finestra si chiude anche
con la X. Se una scorciatoia non risponde, `python webcam_demo.py --debug-keys`
stampa il codice di ogni tasto premuto.

Se la finestra non si apre o è nera, la webcam è un'altra: `--camera 0`.

## Le due modalità di acquisizione

**Doppia** (default): i due post-it stanno insieme nel frame, uno per riquadro.
Comoda per la demo, ma ogni post-it occupa solo mezza inquadratura, quindi il
tratto arriva alla rete con pochi pixel e spesso sfocato.

**Singola** (tasto `M`): un post-it alla volta che riempie tutto il frame.
Cattura A, cambi post-it, catturi B, poi confronti. Molti più pixel sul tratto e
messa a fuoco migliore. **Se i risultati in doppia non convincono, questa è la
prima cosa da provare.**

## Tasti

| tasto | azione |
|---|---|
| `SPAZIO` | cattura (doppia: entrambi e confronta · singola: primo slot libero) |
| `1` / `2` | cattura nello slot A / B |
| `INVIO` | confronta gli slot A e B |
| `M` | cambia modalità doppia ↔ singola |
| `C` | svuota gli slot |
| `R` | torna al live |
| `B` | preprocessing `scan` ↔ `raw` |
| `D` | auto-crop del post-it on/off |
| `T` | cambia soglia |
| `X` | salva gli stadi di preprocessing (per capire dove sbaglia) |
| `S` | salva il confronto in `captures/` + riga in `log.csv` |
| `Q` | esci |

## Il pannello laterale

A destra del video ci sono tre miniature: **anteprima live** (cosa vedrebbe la
rete se catturassi adesso) e i due **slot** già catturati. Sotto l'anteprima ci
sono due numeri:

- **focus** — varianza del laplaciano. Sotto ~150 l'immagine è sfocata e gli
  score non valgono niente. È il problema più frequente: le webcam non mettono a
  fuoco sotto i 20-30 cm.
- **inchiostro** — percentuale di pixel scuri. Sotto il 2.5% c'è troppo poco
  testo, sopra il 25% l'immagine è troppo scura o c'è dentro dell'ombra.

L'etichetta è verde se va bene, ambra se c'è un problema. **Guarda sempre
l'anteprima prima di premere SPAZIO**: se lì il testo è tagliato, storto o
sbiadito, il verdetto è già compromesso.

## Se non funziona: come diagnosticare

**1. Guarda cosa vede la rete, non cosa vedi tu.** Premi `X` dopo una cattura:
salva in `captures/` un'immagine con i 4 stadi affiancati (grayscale → post-it
ritagliato → Otsu → 448×448 finale). Nel 90% dei casi il problema è visibile lì:

| cosa vedi nel debug | causa | rimedio |
|---|---|---|
| stadio 2 ritaglia la cosa sbagliata | l'auto-crop ha preso un riflesso o la scrivania chiara | `D` per disattivarlo, oppure sfondo scuro sotto il post-it |
| stadio 3 tutto nero o tutto bianco | Otsu fallisce, contrasto insufficiente | più luce, penna più scura, `B` per provare `raw` |
| stadio 3 pieno di puntini | rumore/texture della carta | più luce diffusa, evita il flash diretto |
| stadio 4 con testo tagliato | il testo tocca il bordo del post-it | lascia margine bianco attorno alla scrittura |
| stadio 4 sfocato | webcam troppo vicina | allontana, o usa la modalità singola |

**2. Controlla di non essere in "dice sempre stessa persona".** È il sintomo
tipico del domain gap: tutte le foto da webcam finiscono vicine nello spazio
degli embedding e superano la soglia IAM. Fai un test di controllo: confronta un
tuo post-it con uno scritto da un'altra persona e guarda lo score. Se è ancora
sopra 0.79, non è che il sistema "sbaglia", è che la soglia non è tarata su
questo dominio → vai al punto 3.

**3. Ricalibra la soglia.** Raccogli qualche campione per persona (basta salvare
con `S`, o fare foto col telefono) e organizzali così:

```
samples/
  enrico/    01.jpg 02.jpg 03.jpg 04.jpg
  collega/   01.jpg 02.jpg 03.jpg
```

e chiedi a Enrico lo script `calibrate.py`, che non è incluso in questo
pacchetto: stampa la distribuzione degli score genuini e impostori sul *tuo*
dominio, l'EER locale e la soglia corrispondente.

Nel frattempo la soglia si può forzare a mano:

```bash
python webcam_demo.py --threshold-value 0.93
```

Questo risolve il problema alla radice: il modello va bene, era la soglia a
essere di un altro dominio.

**4. Prova `--mode raw`** (o il tasto `B`). Il preprocessing `scan` rimuove
l'ombreggiatura e sbianca lo sfondo; su carta già bianca e ben illuminata a
volte `raw` dà immagini più fedeli a quelle IAM.

## Consigli di ripresa

Il modello ha visto solo scansioni di moduli IAM: carta bianca, penna scura,
niente ombre, un paragrafo intero di testo. Più ti avvicini a quella condizione,
meglio funziona.

- **4-8 righe di testo** per campione, non una parola sola: nelle immagini IAM il
  crop contiene un paragrafo, quindi lo spessore relativo del tratto che la rete
  si aspetta è quello. Un post-it con due parole grandi è, per la rete, una cosa
  che non ha mai visto.
- **Foglio bianco A5 meglio del post-it colorato**, e penna nera meglio della
  biro azzurra chiara.
- **Luce diffusa**, mai la lampada di lato: le ombre creano gradienti che Otsu
  interpreta come inchiostro.
- **Foglio parallelo alla webcam** e non inclinato: non c'è correzione
  prospettica nella pipeline.
- **30-40 cm di distanza** in modalità singola: abbastanza lontano da mettere a
  fuoco, abbastanza vicino da riempire il frame.
- La mano fuori dall'inquadratura, e niente riflessi sulla plastica.

## Soglie

Le soglie **non sono fisse nel codice**: vengono lette dal CSV delle metriche
dell'esperimento a cui appartiene il checkpoint caricato, cercato accanto al file
`.pth` e dentro `triplet_experiments/`. Cambiando i pesi con `--checkpoint`
cambiano anche le soglie, automaticamente. Se il CSV non si trova, si ricade su
quelle di IAM → IAM elencate qui sotto.

Da `triplet_experiments/efficientnet_b1_triplet_iam_to_iam/..._final_metrics.csv`:

| soglia | valore | significato |
|---|---|---|
| `eer` (default) | 0.7912 | equal error rate: FAR ≈ FRR ≈ 11.5% |
| `far1` | 0.9313 | FAR 1%, ma GAR solo ~43% |
| `custom` | — | quella che passi con `--threshold-value` |

Sulle distribuzioni IAM: cosine media 0.89 per le genuine, 0.27 per gli
impostori.

Nota: quelle soglie sono calcolate sul checkpoint `_final.pth` (epoca 19), non
su `_best.pth` (epoca 14) che è quello caricato di default. Per coerenza totale
si può passare `--checkpoint ..\triplet_experiments\efficientnet_b1_triplet_iam_to_iam\efficientnet_b1_triplet_iam_to_iam_final.pth`.

## Protocollo di test per il report

1. **Coppia genuina**: due campioni scritti da te → atteso "STESSA PERSONA".
2. **Coppia impostore**: uno tuo e uno di un'altra persona, stessa frase →
   atteso "PERSONE DIVERSE".
3. Ripeti 3-4 volte e premi `S` ogni volta: catture e score finiscono in
   `captures/` con un `log.csv` pronto da riportare.

## Limite da dichiarare

Il modello è addestrato su IAM (scansioni, carta bianca, penna nera). Le foto da
webcam sono un dominio diverso. Il preprocessing riduce il gap ma non lo annulla,
quindi la demo vale come dimostrazione qualitativa, non come valutazione: le
performance non sono confrontabili con l'EER dell'11.5% misurato su IAM. Il caso
`iam_to_rimes` degli esperimenti (AUC 0.65) quantifica bene quanto pesa il
cambio di dominio.
