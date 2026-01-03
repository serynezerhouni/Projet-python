# Projet-python
# Stratégie Momentum Long/Short — “Poche A” (US Large Caps)

## Objectif
Ce projet implémente et backteste une stratégie **momentum long/short** sur un univers actions US large caps (liste proche du S&P 100).  
L’objectif est de construire un portefeuille long/short basé sur un **score momentum composite**, puis d’évaluer la performance, le risque et les expositions factorielles (CAPM / Fama–French).

## Paramétrage retenu (version finale)
- **Univers** : actions US large caps (liste approximative type S&P 100)
- **Période** : 2010-01-01 → dernière date disponible
- **Fréquence de rebalancement** : **mensuelle** (fin de mois)
- **Construction du portefeuille** : **70% long / 30% short**
- **Sélection** :
  - **TOP = 0.2** : on prend les **20% des titres** avec les meilleurs scores pour la jambe long
  - **BOTTOM = 0.4** : on prend les **40% des titres** avec les moins bons scores pour la jambe short (jambe short plus diversifiée)
- **Pondération (produit final)** : `rank_inv_vol` (rang du score × inverse de la volatilité)
- **Coûts de transaction** : **TC_BPS = 8** (8 bps appliqués au turnover)
- **Données** : `yfinance` (prix ajustés + volumes)

## Données utilisées
- **Prix et volumes** : téléchargés via `yfinance` (`auto_adjust=True`)
- **Marché (CAPM)** : S&P 500 (`^GSPC` via yfinance)
- **Facteurs Fama–French 3** : via `pandas_datareader` (dataset *F-F_Research_Data_Factors_Daily*)

## Score : signal momentum
Le score combine deux composantes :
1) **Momentum (ROC 6 mois)** : taux de variation sur ~126 jours de bourse  
2) **Tendance** : ratio **MA50 / MA200**

Chaque composante est transformée en **rang percentile cross-section** (sur l’univers à une date donnée), puis combinée :

### Score de base
- `rank_ROC(i)` = rang percentile du `ROC_126(i)`
- `rank_MA(i)`  = rang percentile du `MA50(i)/MA200(i)`
- **score_base(i) = 0.7 × rank_ROC(i) + 0.3 × rank_MA(i)**

> Les poids **0.7 / 0.3** sont un choix de design pour donner plus d’importance au momentum “pur” (ROC) tout en gardant un filtre de tendance (MA).

### Pénalités (multiplicatives)
- **RSI (surachat)** : si `RSI_14(i) > 80`, alors on multiplie le score par **0.5**
- **Volume (liquidité)** : si `avg_volume_60(i) < 500 000`, alors on multiplie le score par **0.5**

### Score final
**score_final(i) = score_base(i) × pénalité_RSI(i) × pénalité_volume(i)**

> Dans la **version finale**, le score **n’est pas divisé par la volatilité** : la gestion du risque/volatilité est faite au niveau des poids.

## Construction du portefeuille (pondération `rank_inv_vol`)
À chaque date de rebalancement :
1) On calcule `score_final` pour chaque titre
2) On sélectionne :
   - **Longs** : TOP 20% des scores
   - **Shorts** : BOTTOM 40% des scores
3) On calcule :
   - `r(i)` = rang percentile du score (entre 0 et 1)
   - `inv_vol(i) = 1 / vol_20(i)` (volatilité sur 20 jours)

### Importance (avant normalisation)
- **Longs** : `importance_long(i) = r(i) × inv_vol(i)`
- **Shorts** : `importance_short(i) = (1 − r(i)) × inv_vol(i)`

### Normalisation des poids
On renormalise ensuite pour respecter les expositions cibles :
- Somme des poids long = **+0.70**
- Somme des poids short = **−0.30**

## Coûts de transaction
On utilise un modèle simple basé sur le **turnover** :
- `turnover_t = Σ |w_t(i) − w_{t−1}(i)|`
- **coût_t = turnover_t × (8 / 10 000)**

Le coût est déduit **le premier jour** après chaque rebalancement.

## Analyses incluses (robustesse / diagnostics)
Le notebook inclut :
- Fréquence à laquelle le filtre volume est “binding”
- Comparaison **70/30 vs 50/50** (performance + beta marché via CAPM)
- Régressions **CAPM** et **Fama–French 3 facteurs** (alpha, beta, loadings, R²)
- Analyse par sous-périodes : 2010–2014, 2015–2019, 2020–2024
- Analyse de sensibilité (TOP/BOTTOM/exposition)
- Ablation study : impact de RSI, volume, et du risk-adjust du score

## Exécution
### Option A — Google Colab
1) Ouvrir le notebook dans `notebooks/`
2) Exécuter toutes les cellules
3) Les sorties (tableaux/graphes + poids) sont sauvegardées dans `signals/` (ou `outputs/` selon la config)

### Option B — Local
Installer les dépendances :
```bash
pip install -r requirements.txt
