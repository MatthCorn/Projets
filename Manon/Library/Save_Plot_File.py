import torch
import os
import datetime
from Tools.XMLTools import saveObjAsXml
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Méthode pour sauvegarder toutes les informations de l'apprentissage :
################################################################################################################################################
def save_file(param, model, TrainingError, ValidationError, DictParamObserver) : 
   
    local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('MANON'))], 'MANON')
    folder = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
    save_path = os.path.join(local, 'Manon', 'Save', folder)
    os.makedirs(save_path, exist_ok=True)
    
    saveObjAsXml(param, os.path.join(save_path, 'param'))
    saveObjAsXml({'TrainingError': TrainingError, 'ValidationError': ValidationError}, os.path.join(save_path, 'error'))
    torch.save(model.state_dict(), os.path.join(save_path, 'model_weights.pt'))
    with open(os.path.join(save_path, 'ParamObs.pkl'), 'wb') as f:
        pickle.dump(DictParamObserver(model), f)
################################################################################################################################################

# Simple Plotting Method :
################################################################################################################################################
def plot_graph(title_graph, graph_path, name_file, TrainingError, ValidationError) : 
    plt.figure(figsize=(10, 6))
    plt.plot(TrainingError, label='Training Loss')
    plt.plot(ValidationError, label='Validation Loss')
    #plt.plot([float(torch.std(ValidationOutput))] * len(ValidationError), 'black')
    plt.title(title_graph)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graph_path, name_file))
    plt.close()
################################################################################################################################################

# Plot Training vs Validation loss, save the figure, and annotate minima directly on the graph.
################################################################################################################################################
def plot_graph_with_min(title_graph, graph_path, name_file, TrainingError, ValidationError, mark_train_min=True, mark_val_min=True):
    train = np.asarray(TrainingError)
    val   = np.asarray(ValidationError)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train, label='Training Loss')
    ax.plot(val,   label='Validation Loss')
    plt.ylim(0, 1)
    ax.set_title(title_graph)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    if mark_train_min:
        i = train.argmin()
        ax.scatter(i, train[i], color='tab:blue', zorder=5)
        ax.annotate(f"min train = {train[i]:.4g}\n@ epoch {i}",
                    xy=(i, train[i]), xytext=(5, 10),
                    textcoords="offset points", color='tab:blue',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:blue", lw=0.8))

    if mark_val_min:
        j = val.argmin()
        ax.scatter(j, val[j], color='tab:orange', zorder=5)
        ax.annotate(f"min val = {val[j]:.4g}\n@ epoch {j}",
                    xy=(j, val[j]), xytext=(5, -30),
                    textcoords="offset points", color='tab:orange',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:orange", lw=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(graph_path, name_file))
    plt.close()
################################################################################################################################################

# Plotting Method to have graph between values and 0 (at the minimum) : 
################################################################################################################################################
def plot_graph_with_limitAxis(title_graph, graph_path, name_file, TrainingError, ValidationError) : 
    plt.figure(figsize=(10, 6))
    plt.plot(TrainingError, label='Training Loss')
    plt.plot(ValidationError, label='Validation Loss')
    plt.ylim(0, 1)
    #y_min = min(0, min(TrainingError),min(ValidationError))
    #y_max = max(max(TrainingError), max(ValidationError)) * 1.05  # small headroom above max
    #plt.ylim(y_min, y_max)
    plt.title(title_graph)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graph_path, name_file))
    plt.close()
################################################################################################################################################

# Plotting Method to have variance line in graph : 
################################################################################################################################################
def plot_graph_with_variance(title, save_path, filename, train_error, val_error, variance=None):
    plt.figure()
    plt.plot(train_error, label='Train Loss')
    plt.plot(val_error, label='Validation Loss')
    plt.axhline(y=variance, color='gray', linestyle='--', label='Variance (baseline)')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename))
    plt.close()
################################################################################################################################################