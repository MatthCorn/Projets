from Eusipco.DataMaker import MakeTargetedData
from Eusipco.Transformer import Network as Transformer
from Complete.LRScheduler import Scheduler
from math import sqrt
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from Tools.XMLTools import loadXmlAsObj
import os

################################################################################################################################################
def ChoseOutput(Pred, Input):
    Diff = Pred.unsqueeze(dim=1) - Input.unsqueeze(dim=2)
    Dist = torch.norm(Diff, dim=-1)
    Arg = torch.argmin(Dist, dim=1)
    return Arg

save_path = r'C:\Users\Matth\Documents\Projets\Eusipco\Save\2025-03-11__12-43'

param = loadXmlAsObj(os.path.join(save_path, 'param'))

if torch.cuda.is_available():
    torch.cuda.set_device(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NDataV = param["NDataV"]

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

N = Transformer(n_encoder=param["n_encoder"], d_in=param["d_in"], d_att=param["d_att"],
            WidthsEmbedding=param["WidthsEmbedding"], dropout=param["dropout"])
N.load_state_dict()
N.to(device)

Error = []
ErrorQ1 = []
ErrorQ2 = []
ErrorQ3 = []
Error_lower_whisker = []
Error_upper_whisker = []
ErrorOutliers = []

PerfQ1 = []
PerfQ2 = []
PerfQ3 = []
Perf_lower_whisker = []
Perf_upper_whisker = []
PerfOutliers = []

DVec = param["d_in"]
NVec = param["len_in"]

for i in tqdm(range(p['n_points_reg'])):
    window = deepcopy(param['training_strategy'][0])

    window['std'][0] *= lbd[i]

    ValidationInput, ValidationOutput, ValidationStd = MakeTargetedData(
        NVec=NVec,
        DVec=DVec,
        mean_min=window["mean"][0],
        mean_max=window["mean"][1],
        std_min=window["std"][0],
        std_max=window["std"][1],
        distrib=param["distrib"],
        NData=NDataV,
        Weight=Weight,
    )

    with torch.no_grad():
        Input = ValidationInput.to(device)
        Output = ValidationOutput.to(device)
        Std = ValidationStd.to(device)

        if param['error_weighting'] == 'n':
            Std = torch.mean(Std)

        Prediction = N(Input)

        err = torch.norm((Prediction - Output) / Std, p=2, dim=[1, 2]) / sqrt(DVec * NVec)
        Error.append(torch.mean(err))

        data = err.cpu().numpy()
        ErrorQ1.append(np.percentile(data, 25))
        ErrorQ2.append(np.median(data))
        ErrorQ3.append(np.percentile(data, 75))
        IQR = ErrorQ3[-1] - ErrorQ1[-1]

        Error_lower_whisker.append(max(ErrorQ1[-1] - 1.5 * IQR, min(data)))
        Error_upper_whisker.append(min(ErrorQ3[-1] + 1.5 * IQR, max(data)))

        perf = torch.mean((ChoseOutput(Prediction, Input) == ChoseOutput(Output, Input)).to(float), dim=-1)

        data = perf.cpu().numpy()
        PerfQ1.append(np.percentile(data, 25))
        PerfQ2.append(np.median(data))
        PerfQ3.append(np.percentile(data, 75))
        IQR = PerfQ3[-1] - PerfQ1[-1]

        Perf_lower_whisker.append(max(PerfQ1[-1] - 1.5 * IQR, min(data)))
        Perf_upper_whisker.append(min(PerfQ3[-1] + 1.5 * IQR, max(data)))


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

import matplotlib.pyplot as plt
from Eusipco.Boxplot import add_boxplot
for i in range(len(lbd)):
    add_boxplot(
        error['PerfQ1'][i],
        error['PerfQ2'][i],
        error['PerfQ3'][i],
        error['Perf_lower_whisker'][i],
        error['Perf_upper_whisker'][i],
        ax=None,
        i=i
    )
plt.show()