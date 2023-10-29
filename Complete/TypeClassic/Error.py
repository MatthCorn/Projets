import torch
from torch.nn import functional as F
import math

def TrainingError(source, target, ended, batch_size, batch_indice, network):
    j = batch_indice
    batch_source = source[j * batch_size: (j + 1) * batch_size].detach().to(device=network.device, dtype=torch.float32)
    batch_target = target[j * batch_size: (j + 1) * batch_size].detach().to(device=network.device, dtype=torch.float32)
    batch_ended = ended[j * batch_size: (j + 1) * batch_size].detach().to(device=network.device, dtype=torch.float32)
    # batch_ended.shape = (batch_size, len_target, 1)

    predicted_target = network.forward(source=batch_source, target=batch_target)

    # Comme la décision d'arrêter d'écrire passe par "action", et pas par un token <end>, on ne s'interesse pas au dernier mot. En revenche, la dernière action,
    # si observée, ne peut être que "attendre la nouvelle salve", sinon cela signifie que le réseau est sous-dimensionné (len_target - n_PDWs_memory trop petit).
    # On ne s'interesse aussi qu'aux prédictions à partir de la position "network.n_PDWs_memory", ce qui correspond aux prédictions devant être
    # faites à l'arrivée de la dernière salve.

    predicted_target = predicted_target[:, network.n_PDWs_memory-1:-1]
    # predicted_target.shape = (batch_size, len_target-PDWsMemory, d_target+num_flags)

    error_trans = torch.norm((batch_target[:, network.n_PDWs_memory:, :(network.d_PDW + network.n_flags)] - predicted_target) * (1 - batch_ended), dim=(0, 1))/float(torch.norm((1 - batch_ended)))
    # error_trans.shape = d_PDW+n_flags

    return error_trans


def ErrorAction(source, target, ended, network, batch_size=50, action='', optimizer=None):
    data_size, _, d_out = target.shape
    n_batch = int(data_size/batch_size)

    error_trans = []

    if action == 'Training':
        for j in range(n_batch):
            optimizer.zero_grad(set_to_none=True)

            err_trans = TrainingError(source, target, ended, batch_size, j, network)

            err = torch.norm(err_trans)
            err.backward()
            optimizer.step()

            error_trans.append(err_trans**2)


    elif action == 'Validation':
        for j in range(n_batch):
            with torch.no_grad():
                err_trans = TrainingError(source, target, ended, batch_size, j, network)

            error_trans.append(err_trans**2)

    error_temp = torch.sqrt(sum(error_trans) / n_batch)
    error_trans = error_temp.tolist()
    error = float(torch.norm(error_temp) / math.sqrt(network.d_PDW + network.n_flags))
    return error, error_trans
