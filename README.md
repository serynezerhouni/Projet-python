# Stratégie Momentum Long/Short (Poche A)

## Idée générale
On met en place une stratégie **momentum cross-section** sur un univers d’actions US large caps (type S&P100).
À chaque date de rebalancement, on **achète les meilleurs** titres (long) et on **vend les moins bons** (short), puis on mesure la performance et les expositions (CAPM / Fama-French).

## Paramètres (version finale)
- **Période** : 2010 → aujourd’hui  
- **Rebalancement** : mensuel (fin de mois)  
- **Exposition** : **70% long / 30% short**  
- **Sélection** :  
  - **Top 20%** en long  
  - **Bottom 40%** en short (short plus diversifié)  
- **Coûts** : 8 bps appliqués au turnover  
- **Données** : `yfinance` (prix ajustés + volumes)

## Signal (score momentum)
Le score combine :
- **Momentum** : performance sur ~6 mois (ROC)
- **Tendance** : ratio MA50 / MA200

Les deux composantes sont transformées en **rang percentile** dans l’univers, puis combinées :
- score = 0.7 × rank(ROC) + 0.3 × rank(MA50/MA200)

On applique ensuite deux pénalités simples :
- **RSI** : si RSI > 80 → score × 0.5  
- **Liquidité** : si volume moyen < seuil → score × 0.5  

> Dans la version finale, on ne divise pas le score par la volatilité : la gestion du risque se fait via la pondération.

## Construction du portefeuille (pondération `rank_inv_vol`)
Sur les titres sélectionnés, les poids sont proportionnels à :
- **rang du score × inverse de la volatilité (20 jours)**  
Puis on normalise pour obtenir :
- somme des poids **long = +0.70**
- somme des poids **short = −0.30**

## Coûts de transaction
Turnover = Σ |w_t − w_{t−1}|
Coût = turnover × (8 / 10 000), déduit au premier jour après rebalancement.

## Analyses produites
- comparaison **70/30 vs 50/50** + CAPM (alpha/beta)
- FF3 (loadings + R²)
- sous-périodes (2010–2014 / 2015–2019 / 2020–2024)
- sensibilité hyperparamètres (TOP/BOTTOM/expo)
- ablation (impact RSI / volume)
---
## Exécution (Google Colab)

Dans un notebook Colab, exécuter :

```bash
!git clone https://github.com/serynezerhouni/Projet-python.git
%cd Projet-python
!pip -q install -r requirements.txt
!python main.py
