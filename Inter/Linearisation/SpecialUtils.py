import torch
from Inter.Model.DataMaker import GetData as GD

def GetData(d_in, n_pulse_plateau, n_sat, n_mes, len_in, len_out, n_data_training, n_data_validation=1000, sensitivity=0.1,
            weight_f=None, weight_l=None, bias='none', std_min=1., std_max=5., mean_min=-10., mean_max=10.,
            distrib='log', save_path=None, parallel=False, max_inflight=None):

    [(TrainingInput, TrainingOutput, TrainingMasks, TrainingStd),
     (ValidationInput, ValidationOutput, ValidationMasks, ValidationStd)] = GD(
        d_in=d_in,
        n_pulse_plateau=n_pulse_plateau,
        n_sat=n_sat,
        n_mes=n_mes,
        len_in=len_in,
        len_out=len_out,
        n_data_training=n_data_training,
        n_data_validation=n_data_validation,
        sensitivity=sensitivity,
        bias=bias,
        mean_min=mean_min,
        mean_max=mean_max,
        std_min=std_min,
        std_max=std_max,
        distrib=distrib,
        weight_f=weight_f,
        weight_l=weight_l,
        type='complete',
        save_path=save_path,
        parallel=parallel,
        max_inflight=max_inflight
    )

    (TrainingProcessedInput1, TrainingProcessedInput2, TrainingProcessedOutput, TrainingNextMaskInput, TrainingNextMaskOutput,
     TrainingNoSequenceMask) = PostProcess(TrainingInput, TrainingOutput, TrainingMasks, len_in, len_out, n_data_training)

    (ValidationProcessedInput1, ValidationProcessedInput2, ValidationProcessedOutput, ValidationNextMaskInput, ValidationNextMaskOutput,
     ValidationNoSequenceMask) = PostProcess(ValidationInput, ValidationOutput, ValidationMasks, len_in, len_out, n_data_validation)

    return [(TrainingProcessedInput1, TrainingProcessedInput2, TrainingProcessedOutput, TrainingStd,
             TrainingNextMaskInput, TrainingNextMaskOutput, TrainingNoSequenceMask),
            (ValidationProcessedInput1, ValidationProcessedInput2, ValidationProcessedOutput, ValidationStd,
             ValidationNextMaskInput, ValidationNextMaskOutput, ValidationNoSequenceMask)]

def PostProcess(Input, Output, Masks, len_in, len_out, n_data):
    AddMask = Masks[0][:, :, 0].clone()
    Input = Input.clone()
    Output = Output.clone()

    # on décode l'information du temps d'arrivée
    Output[..., -1] = torch.arange(0, len_out, 1) - Output[..., -1]

    tps_maintien = 1
    Mask = torch.arange(0, len_out, 1).unsqueeze(0) >= AddMask.argmax(dim=-1).unsqueeze(1)  # détermine les positions avec padding
    TOE_Out = Output[:, :, -1] + Output[:, :, -2]  # calcul des temps de fin des impulsions cibles (TOA + LI)
    TOE_Out[Mask] = torch.inf  # les impulsions de padding sont associées à un padding de +inf
    TOA_In = torch.arange(0, len_in, 1).expand(n_data, -1)  # tensor des temps d'arrivée des impulsions sources
    TOE_In = TOA_In + Input[..., -1]  # calcul des temps de fin des impulsions sources (TOA + LI)

    # calcul de la différence de temps d'arrivée des impulsions sources, puis ajout de l'info pour chaque impulsion
    DTOA_In = torch.cat([TOA_In[:, 1:] - TOA_In[:, :-1],
                         (TOE_In.max(dim=1).values - TOA_In[:, -1] + tps_maintien).unsqueeze(-1)], dim=-1)
    Input = torch.cat([Input, DTOA_In.unsqueeze(-1)], dim=-1)

    # liste des temps associés à chaque évènement pour les impulsions cibles (temps de publication) concaténé
    # à la liste des temps associés à chaque évènement pour les impulsions sources (temps d'arrivée)
    TimeEvent = torch.cat((TOE_Out, TOA_In), dim=-1)

    # on récupère les indices des évènements pour pouvoir les réorganisés par ordre chronologique
    ArgSorted_TimeEvent = torch.argsort(TimeEvent, dim=-1)

    # crée un masque indiquant la position des PDW cibles dans les évènements triés chronologiquement, en ne considérant pas les PDW cibles padding
    IsOutput_position = torch.gather(torch.cat((1 - Mask.to(torch.float), torch.zeros_like(TOA_In)), dim=-1), dim=-1, index=ArgSorted_TimeEvent)

    # crée un masque indiquant la position des PDW sources dans les évènements triés chronologiquement
    IsInput = torch.cat((torch.zeros_like(TOE_Out), torch.ones_like(TOA_In)), dim=-1)
    IsInput_position = torch.gather(IsInput, dim=-1, index=ArgSorted_TimeEvent)

    # crée la liste des indices des PDW sources dans le flux sérialisé (liée à Input 1 dans le manuscrit)
    # la taille de cette liste est la taille du flux, soit len(Input)+len(Output), les indices dans la liste varient entre 0 et len(Input)-1
    Input_position = (torch.cumsum(IsInput_position, dim=-1) - 1).to(torch.int64)

    # crée la liste des indices des PDW cibles dans le flux sérialisé (liée à Input 2 dans le manuscrit)
    # la taille de cette liste est la taille du flux, soit len(Input)+len(Output), les indices dans la liste varient entre 0 et len(Output)-1
    Output_position = ((torch.cumsum(IsOutput_position, dim=-1) - 1) * IsOutput_position).to(torch.int64)

    # crée le masque indiquant si le PDW cible doit être le jeton NEXT (liée à Input2 dans le manuscrit)
    NextMaskInput = 1 - IsOutput_position.unsqueeze(-1)
    # crée le masque indiquant si le shifted-PDW cible doit être le jeton NEXT (liée à Output dans le manuscrit)
    NextMaskOutput = torch.roll(NextMaskInput, shifts=-1, dims=1)

    # on recrée les séquences de PDW (2 Input et 1 Output, cf manuscrit) avec la liste des PDW sources et des
    # PDW cibles, et les listes des indices de ces PDW dans le flux sérialisé (ce qui correspond cette fois à créer Input 1, Input 2
    # et Output, comme dans le manuscrit)
    ProcessedInput1 = torch.gather(Input, dim=1, index=Input_position.unsqueeze(-1).expand(*Input_position.shape, Input.shape[-1]))
    ProcessedInput2 = torch.gather(Output, dim=1, index=Output_position.unsqueeze(-1).expand(*Output_position.shape, Output.shape[-1]))
    ProcessedOutput = torch.roll(ProcessedInput2, shifts=-1, dims=1)

    # on modifie l'encodage temporel des temps d'arrivées pour les PDW cibles (Input 2 et Output, cf manuscrit)
    TOAInput1 = torch.gather(TOA_In, dim=1, index=Input_position)
    ProcessedInput2[..., -1] -= TOAInput1
    ProcessedOutput[..., -1] -= TOAInput1

    # masque qui détermine dans le flux sérialisé s'il s'agit de PDW padding
    OnSequenceMask = (IsOutput_position + IsInput_position).unsqueeze(-1)

    return (ProcessedInput1, ProcessedInput2, ProcessedOutput, NextMaskInput, NextMaskOutput, OnSequenceMask)


if __name__ == '__main__':
    GetData(4, 6, 5, 5, 10, 15, 10000, n_data_validation=100, parallel=True)