import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def plot_hist_top_k(df, cols, k):
    """Plotting the frequencies of top k unique values on specific columns
       in a dataframe 
         Args: 
           df: DataFrame
           cols: list of columns of the DataFrame to analyse
           k: int, the number of unique values to plot    
    """
    for col in cols:
        # Compter les fréquences relatives (%)
        top_values = (df[col]
                    .value_counts(normalize=True, dropna=False) * 100
                    ).head(k+1)
        
        # Transformer en DataFrame pour seaborn
        plot_df = top_values.reset_index()
        plot_df.columns = [col, 'Fréquence (%)']
        plot_df = plot_df.sort_values(by='Fréquence (%)', ascending=False)
        
        # Tracer le graphique
        plt.figure(figsize=(8, 4))
        sns.barplot(
            data=plot_df,
            x=col,
            hue=col,
            y='Fréquence (%)',
            order= plot_df[col].tolist(),
            palette='viridis'
        )
        plt.title(f"Top {k} valeurs de '{col}' (en %)")
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.show()

def cramers_v(x, y):
  #Corriger
  "Performs cramers test and plot corelation matrix "
  table = pd.crosstab(x, y)
  chi2 = chi2_contingency(table)[0]
  n = table.sum().sum()
  r, k = table.shape
  return np.sqrt(chi2 / (n * (min(r - 1, k - 1))))