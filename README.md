# Stratégie Momentum Long/Short (Poche A)

Ce projet implémente une stratégie **momentum cross-section** sur un univers d’actions US large caps. Chaque mois, les titres sont classés via un **score momentum** (performance ~6 mois + tendance MA50/MA200), puis on construit un portefeuille long/short : **long sur les meilleurs**, **short sur les moins bons**.

**Configuration retenue :**
- Période : 2010 → aujourd’hui
- Rebalancement : mensuel
- Exposition : 70% long / 30% short
- Sélection : Top 20% (long) / Bottom 40% (short)
- Pondération finale : `rank_inv_vol` (rang du score × inverse-vol 20j)
- Coûts de transaction : 8 bps appliqués au turnover
- Données : `yfinance` (prix ajustés + volumes)

Le notebook détaille la construction du signal, le backtest et les analyses de robustesse (sensibilité, CAPM/FF3, sous-périodes, ablation).


---
## Exécution (Google Colab)

Dans un notebook Colab, exécuter :

```bash
!git clone https://github.com/serynezerhouni/Projet-python.git
%cd Projet-python
!pip -q install -r requirements.txt
!python main.py

Où trouver les résultats :
	•	Tableaux CSV : outs/tables/
	•	Figures (PNG) : outs/figures/
	•	Poids “latest” : signals/

Télécharger toutes les sorties :
import shutil
from google.colab import files

shutil.make_archive("outs", "zip", "outs")
files.download("outs.zip")
