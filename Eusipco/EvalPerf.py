from Eusipco.DataMaker import MakeTargetedData
from Eusipco.Transformer import Network as Transformer
from Eusipco.CNN import Encoder as CNN
from math import sqrt
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from Tools.XMLTools import loadXmlAsObj
import os

if torch.cuda.is_available():
    torch.cuda.set_device(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ChoseOutput(Pred, Input):
    Diff = Pred.unsqueeze(dim=1) - Input.unsqueeze(dim=2)
    Dist = torch.norm(Diff, dim=-1)
    Arg = torch.argmin(Dist, dim=1)
    return Arg

save_path_Tr = r'C:\Users\Matth\Documents\Projets\Eusipco\Save\2025-03-13__20-45'
save_path_Cnn = r'C:\Users\Matth\Documents\Projets\Eusipco\Save\2025-03-13__20-44'

Tr_param = loadXmlAsObj(os.path.join(save_path_Tr, 'param'))
Tr_network = Transformer(n_encoder=Tr_param["n_encoder"], d_in=Tr_param["d_in"], d_att=Tr_param["d_att"],
            WidthsEmbedding=Tr_param["WidthsEmbedding"], dropout=Tr_param["dropout"])
Tr_network.load_state_dict(torch.load(os.path.join(save_path_Tr, 'Last_network')))
Tr_network.to(device)
Tr_weight = torch.load(os.path.join(save_path_Tr, 'Weight'))


Cnn_param = loadXmlAsObj(os.path.join(save_path_Cnn, 'param'))
Cnn_network = CNN(n_encoder=Cnn_param["n_encoder"], d_in=Cnn_param["d_in"], d_att=Cnn_param["d_att"],
            WidthsEmbedding=Cnn_param["WidthsEmbedding"], dropout=Cnn_param["dropout"])
Cnn_network.load_state_dict(torch.load(os.path.join(save_path_Cnn, 'Last_network')))
Cnn_network.to(device)
Cnn_weight = torch.load(os.path.join(save_path_Cnn, 'Weight'))



NDataV = Tr_param["NDataV"]

p = {"n_points_reg": 10,
     "lbd": {"min": 1e-4, "max": 1},
     "distrib": "log"}

windows = []
if p['distrib'] == 'log':
    f = np.log
    g = np.exp
elif p['distrib'] == 'uniform':
    f = lambda x: x
    g = lambda x: x

lbd = np.flip(g(np.linspace(f(p['lbd']['min']), f(p['lbd']['max']), p['n_points_reg'], endpoint=True)))

Error_Tr = []
Error_Cnn = []
ErrorQ1 = []
ErrorQ2 = []
ErrorQ3 = []
Error_lower_whisker = []
Error_upper_whisker = []
ErrorOutliers = []

Perf_Tr = []
Perf_Cnn = []
PerfQ1 = []
PerfQ2 = []
PerfQ3 = []
Perf_lower_whisker = []
Perf_upper_whisker = []
PerfOutliers = []

DVec = Tr_param["d_in"]
NVec = Tr_param["len_in"]

for i in tqdm(range(p['n_points_reg'])):
    window = deepcopy(Tr_param['training_strategy'][0])

    window['std'][0] *= lbd[i]

    ValidationInput, ValidationOutput, ValidationStd = MakeTargetedData(
        NVec=NVec,
        DVec=DVec,
        mean_min=window["mean"][0],
        mean_max=window["mean"][1],
        std_min=window["std"][0],
        std_max=window["std"][1],
        distrib=Tr_param["distrib"],
        NData=NDataV,
        Weight=Tr_weight,
    )

    with torch.no_grad():
        Input = ValidationInput.to(device)
        Output = ValidationOutput.to(device)
        Std = ValidationStd.to(device)

        Prediction = Tr_network(Input)

        err = torch.norm((Prediction - Output) / Std, p=2, dim=[1, 2]) / sqrt(DVec * NVec)
        Error_Tr.append(float(torch.mean(err)))

        data = err.cpu().numpy()
        ErrorQ1.append(np.percentile(data, 25))
        ErrorQ2.append(np.median(data))
        ErrorQ3.append(np.percentile(data, 75))
        IQR = ErrorQ3[-1] - ErrorQ1[-1]

        Error_lower_whisker.append(max(ErrorQ1[-1] - 1.5 * IQR, min(data)))
        Error_upper_whisker.append(min(ErrorQ3[-1] + 1.5 * IQR, max(data)))

        perf = torch.mean((ChoseOutput(Prediction, Input) == ChoseOutput(Output, Input)).to(float), dim=-1)

        Perf_Tr.append(float(torch.mean(perf)))
        data = perf.cpu().numpy()
        PerfQ1.append(np.percentile(data, 25))
        PerfQ2.append(np.median(data))
        PerfQ3.append(np.percentile(data, 75))
        IQR = PerfQ3[-1] - PerfQ1[-1]

        Perf_lower_whisker.append(max(PerfQ1[-1] - 1.5 * IQR, min(data)))
        Perf_upper_whisker.append(min(PerfQ3[-1] + 1.5 * IQR, max(data)))

    ValidationInput, ValidationOutput, ValidationStd = MakeTargetedData(
        NVec=NVec,
        DVec=DVec,
        mean_min=window["mean"][0],
        mean_max=window["mean"][1],
        std_min=window["std"][0],
        std_max=window["std"][1],
        distrib=Tr_param["distrib"],
        NData=NDataV,
        Weight=Cnn_weight,
    )

    with torch.no_grad():
        Input = ValidationInput.to(device)
        Output = ValidationOutput.to(device)
        Std = ValidationStd.to(device)

        Prediction = Cnn_network(Input)

        err = torch.norm((Prediction - Output) / Std, p=2, dim=[1, 2]) / sqrt(DVec * NVec)
        Error_Cnn.append(float(torch.mean(err)))
        perf = torch.mean((ChoseOutput(Prediction, Input) == ChoseOutput(Output, Input)).to(float), dim=-1)
        Perf_Cnn.append(float(torch.mean(perf)))



import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(Error_Tr, 'red', label='Transformer')
ax1.plot(Error_Cnn, 'b', label='CNN')

ax2.plot(Perf_Tr, 'red', label='Transformer')
ax2.plot(Perf_Cnn, 'b', label='CNN')

error = {"ErrorQ1": ErrorQ1,
         "ErrorQ2": ErrorQ2,
         "ErrorQ3": ErrorQ3,
         "Error_lower_whisker": Error_lower_whisker,
         "Error_upper_whisker": Error_upper_whisker,
         "ErrorOutliers": ErrorOutliers,
         "PerfQ1": PerfQ1,
         "PerfQ2": PerfQ2,
         "PerfQ3": PerfQ3,
         "Perf_lower_whisker": Perf_lower_whisker,
         "Perf_upper_whisker": Perf_upper_whisker,
         "PerfOutliers": PerfOutliers,}

from Eusipco.Boxplot import add_boxplot
for i in range(len(lbd)):
    add_boxplot(
        error['ErrorQ1'][i],
        error['ErrorQ2'][i],
        error['ErrorQ3'][i],
        error['Error_lower_whisker'][i],
        error['Error_upper_whisker'][i],
        ax=ax1,
        i=i
    )

for i in range(len(lbd)):
    add_boxplot(
        error['PerfQ1'][i],
        error['PerfQ2'][i],
        error['PerfQ3'][i],
        error['Perf_lower_whisker'][i],
        error['Perf_upper_whisker'][i],
        ax=ax2,
        i=i
    )

ax1.set_ylim(top=1, bottom=0)
ax1.legend(loc='upper left', framealpha=0.3)
ax1.set_title("Error")
ax1.set_xticks([i for i in range(10) if not i%3])
ax1.set_xticklabels([f"{lbd[i]:.0e}" for i in range(10) if not i%3])
ax1.set_box_aspect(1)

ax2.set_ylim(bottom=0, top=1)
ax2.legend(loc='lower left', framealpha=0.3)
ax2.set_title("Accuracy")
ax2.set_xticks([i for i in range(10) if not i%3])
ax2.set_xticklabels([f"{lbd[i]:.0e}" for i in range(10) if not i%3])
ax2.set_box_aspect(1)

fig.tight_layout(pad=1.0)

plt.show()

plt.show()