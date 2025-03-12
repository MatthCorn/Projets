import numpy as np
import matplotlib.pyplot as plt

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

# Dessiner la boîte avec un bord coloré
ax.add_patch(plt.Rectangle((0.4, Q1), 0.2, Q3 - Q1, fill=False, edgecolor='black', linewidth=2))

# Dessiner la médiane
ax.plot([0.4, 0.6], [Q2, Q2], color='black', linewidth=2)

# Dessiner les moustaches
ax.plot([0.5, 0.5], [lower_whisker, Q1], color='black', linewidth=2)  # Moustache basse
ax.plot([0.5, 0.5], [Q3, upper_whisker], color='black', linewidth=2)  # Moustache haute

# Dessiner les capuchons des moustaches
ax.plot([0.45, 0.55], [lower_whisker, lower_whisker], color='black', linewidth=2)  # Capuchon bas
ax.plot([0.45, 0.55], [upper_whisker, upper_whisker], color='black', linewidth=2)  # Capuchon haut

# Dessiner les outliers
ax.scatter([0.5] * len(outliers), outliers, color='red', marker='o', s=100)  # Outliers en rouge

# Ajuster les axes et les labels
ax.set_xlim(0, 1)
ax.set_ylim(min(data) - 1, max(data) + 1)  # Ajuster les limites en fonction des données
ax.set_yticks([Q1, Q2, Q3, lower_whisker, upper_whisker])
ax.set_yticklabels(['Q1', 'Q2 (médiane)', 'Q3', 'Moustache basse', 'Moustache haute'])

# Ajouter un titre
plt.title('Boxplot Personnalisé avec Matplotlib')

# Afficher
plt.show()