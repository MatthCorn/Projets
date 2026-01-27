import numpy as np
import matplotlib.pyplot as plt

def add_boxplot(Q1, Q2, Q3, lower_whisker, upper_whisker, outliers=None, ax=None, i=0, color='black'):
    if ax is None:
        ax = plt.gca()
    # Dessiner la boîte avec un bord coloré
    ax.add_patch(plt.Rectangle((-0.1 + i, Q1), 0.2, Q3 - Q1, fill=False, edgecolor=color, linewidth=1))

    # Dessiner la médiane
    ax.plot([i - 0.1, 0.1 + i], [Q2, Q2], color=color, linewidth=1)

    # Dessiner les moustaches
    ax.plot([i, i], [lower_whisker, Q1], color=color, linewidth=1)  # Moustache basse
    ax.plot([i, i], [Q3, upper_whisker], color=color, linewidth=1)  # Moustache haute

    # Dessiner les capuchons des moustaches
    ax.plot([i - 0.05, 0.05 + i], [lower_whisker, lower_whisker], color=color, linewidth=1)  # Capuchon bas
    ax.plot([i - 0.05, 0.05 + i], [upper_whisker, upper_whisker], color=color, linewidth=1)  # Capuchon haut

    # Dessiner les outliers
    if outliers is not None:
        ax.scatter([i] * len(outliers), outliers, color=color, marker='o', s=100)  # Outliers en rouge


if __name__ == '__main__':
    # Générer des données bidons
    np.random.seed(0)
    data = np.random.randn(100)  # Données normales pour l'exemple

    # Calculer les éléments nécessaires pour un boxplot
    Q1 = np.percentile(data, 25)
    Q2 = np.median(data)  # Médiane (50%)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Calcul des moustaches (limitées aux valeurs observées)
    lower_whisker = max(Q1 - 1.5 * IQR, min(data))  # Moustache basse
    upper_whisker = min(Q3 + 1.5 * IQR, max(data))  # Moustache haute

    # Identification des outliers
    outliers = data[(data < lower_whisker) | (data > upper_whisker)]

    # Créer une figure et des axes
    fig, ax = plt.subplots()

    i = 0

    add_boxplot(Q1, Q2, Q3, lower_whisker, upper_whisker, outliers, ax=ax, i=i)

    # Ajuster les axes et les labels
    ax.set_xlim(i - 0.5, 0.5 + i)
    ax.set_ylim(min(data) - 1, max(data) + 1)  # Ajuster les limites en fonction des données
    ax.set_yticks([Q1, Q2, Q3, lower_whisker, upper_whisker])
    ax.set_yticklabels(['Q1', 'Q2 (médiane)', 'Q3', 'Moustache basse', 'Moustache haute'])

    # Ajouter un titre
    plt.title('Boxplot Personnalisé avec Matplotlib')

    # Afficher
    plt.show()