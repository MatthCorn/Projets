import torch
from torch.nn import functional as F
import math

def TrainingError(source, target, ended, batch_size, batch_indice, network):
    j = batch_indice
    batch_source = source[j * batch_size: (j + 1) * batch_size].detach().to(device=network.device, dtype=torch.float32)
    batch_target = target[j * batch_size: (j + 1) * batch_size].detach().to(device=network.device, dtype=torch.float32)
    batch_ended = ended[j * batch_size: (j + 1) * batch_size].detach().to(device=network.device, dtype=torch.float32)
    # batch_ended.shape = (batch_size, len_target, 1)

    # Pour faire en sorte que l'action "continuer d'écrire" soit autant représentée en True qu'en False, on pondère le masque des actions
    n_words = torch.sum((1 - batch_ended), dim=1).unsqueeze(-1).to(torch.int32)
    # n_words.shape = (batch_size, 1, 1)
    batch_action_mask = F.pad(1 - batch_ended, [0, 0, 0, 1])/torch.sqrt((2*n_words + 1e-5))
    for i in range(len(n_words)):
        if int(n_words[i]) == 0:
            batch_action_mask[(i, int(n_words[i]))] = 1
        else:
            batch_action_mask[(i, int(n_words[i]))] = 1/4

    predicted_target, predicted_action = network.forward(source=batch_source, target=batch_target)

    # Comme la décision d'arrêter d'écrire passe par "action", et pas par un token <end>, on ne s'interesse pas au dernier mot. En revenche, la dernière action,
    # si observée, ne peut être que "attendre la nouvelle salve", sinon cela signifie que le réseau est sous-dimensionné (len_target - n_PDWs_memory trop petit).
    # On ne s'interesse aussi qu'aux prédictions à partir de la position "network.n_PDWs_memory", ce qui correspond aux prédictions devant être
    # faites à l'arrivée de la dernière salve.

    predicted_target = predicted_target[:, network.n_PDWs_memory-1:-1]
    # predicted_target.shape = (batch_size, len_target-PDWsMemory, d_target+num_flags)
    predicted_action = predicted_action[:, network.n_PDWs_memory-1:]
    # predicted_action.shape = (batch_size, len_target-PDWsMemory+1, 1)

    error_trans = torch.norm((batch_target[:, network.n_PDWs_memory:] - predicted_target) * (1 - batch_ended), dim=(0, 1))/float(torch.norm((1 - batch_ended)))
    # error_trans.shape = d_PDW+n_flags
    error_act = torch.norm((predicted_action - F.pad(1 - batch_ended, [0, 0, 0, 1])) * batch_action_mask)/math.sqrt(batch_size)

    return error_trans, error_act


def ErrorAction(source, target, ended, network, hookers, weights, batch_size=50, action='', optimizer=None):
    data_size, _, d_out = target.shape
    n_batch = int(data_size/batch_size)

    error_trans = []
    error_act = 0

    if action == 'Training':
        for j in range(n_batch):
            optimizer.zero_grad(set_to_none=True)

            err_trans, err_act = TrainingError(source, target, ended, batch_size, j, network)

            err = torch.norm(err_trans)*weights['trans'] + err_act*weights['act'] + \
                  (1 - F.tanh(hookers.ReleaseError('variance')/weights['var']['threshold']))*weights['var']['mod'] + \
                  (1 - F.tanh(hookers.ReleaseError('diversity')/weights['div']['threshold']))*weights['div']['mod'] + \
                  hookers.ReleaseError('symetry')*weights['sym'] + hookers.ReleaseError('likeli')*weights['likeli']

            err.backward()
            optimizer.step()

            error_act += float(err_act)**2
            error_trans.append(err_trans**2)


    elif action == 'Validation':
        for j in range(n_batch):
            with torch.no_grad():
                err_trans, err_act = TrainingError(source, target, ended, batch_size, j, network)

            error_act += float(err_act)**2
            error_trans.append(err_trans**2)

    error_temp = torch.sqrt(sum(error_trans) / n_batch)
    error_trans = error_temp.tolist()
    error = float(torch.norm(error_temp) / math.sqrt(network.d_PDW + network.n_flags))
    error_act = math.sqrt(error_act / n_batch)
    return error, error_act, error_trans
