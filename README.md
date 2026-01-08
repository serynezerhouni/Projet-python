# Stratégie Momentum Long/Short 

Ce projet implémente une stratégie momentum long/short sur un univers de large caps US, rebalancée mensuellement : 
- long sur les meilleurs scores, 
- short sur les plus faibles, 
- avec des pénalités RSI et volume pour limiter les signaux “fragiles”.  

La pondération rank × inverse-volatilité permet de concentrer l’exposition sur les convictions tout en réduisant la contribution des titres les plus instables.

**Configuration retenue :**
- Période : 2010 → aujourd’hui
- Rebalancement : mensuel
- Exposition : 70% long / 30% short
- Sélection : Top 20% (long) / Bottom 40% (short)
- Pondération finale : `rank_inv_vol` (rang du score × inverse-vol 20j)
- Coûts de transaction : 8 bps appliqués au turnover
- Données : `yfinance` (prix ajustés + volumes)

Le notebook présente la littérature, la construction du signal et du portefeuille, le backtest, puis les analyses de robustesse (sensibilité, CAPM/FF3, sous-périodes, ablation, train/test), avant de conclure sur les limites et les principaux enseignements.

> Ce notebook contient déjà l’ensemble des résultats (tables, graphiques) et les commentaires d’analyse.  
> Les commandes ci-dessous servent uniquement à reproduire le backtest à partir du repo GitHub dans un nouveau notebook Colab.

---

## Exécution (Google Colab)

Dans un notebook Colab, exécuter :

```bash
!git clone https://github.com/serynezerhouni/Projet-python.git
%cd Projet-python
!pip -q install -r requirements.txt
!python main.py
```
Où trouver les résultats :
- Tableaux CSV : `outs/tables/`
- Figures (PNG) : `outs/figures/`
- Poids "latest" : `signals/`
  
Télécharger toutes les sorties :
  ```bash
import shutil
from google.colab import files
shutil.make_archive("outs", "zip", "outs")
files.download("outs.zip")
```

