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

    return 0

def PostProcess(Input, Output, Masks, len_in, len_out, n_data):
    Output[..., -1] = torch.arange(0, len_out, 1) - Output[..., -1]
    AddMask = Masks[0][:, :, 0]

    tps_maintien = 1
    Mask = torch.arange(0, len_out, 1).unsqueeze(0) >= AddMask.argmax(dim=-1).unsqueeze(1)
    TOE_Out = Output[:, :, -1] + Output[:, :, -2]
    TOE_Out[Mask] = torch.inf
    TOA_In = torch.arange(0, len_in, 1).expand(n_data, -1)
    TOE_In = TOA_In + Input[..., -1]
    DTOA_In = torch.cat([TOA_In[:, 1:] - TOA_In[:, :-1],
                         (TOE_In.max(dim=1).values - TOA_In[:, -1] + tps_maintien).unsqueeze(-1)], dim=-1)
    Input = torch.cat([Input, DTOA_In.unsqueeze(-1)], dim=-1)

    TimeEvent = torch.cat((TOE_Out, TOA_In), dim=-1)
    ArgSorted_TimeEvent = torch.argsort(TimeEvent, dim=-1)
    IsInput = torch.cat((torch.zeros_like(TOE_Out), torch.ones_like(TOA_In)), dim=-1)

    IsOutput_position = torch.gather(torch.cat((1 - Mask.to(torch.float), torch.zeros_like(TOA_In)), dim=-1), dim=-1, index=ArgSorted_TimeEvent)

    IsInput_position = torch.gather(IsInput, dim=-1, index=ArgSorted_TimeEvent)
    Input_position = (torch.cumsum(IsInput_position, dim=-1) - 1).to(torch.int64)
    Output_position = ((torch.cumsum(IsOutput_position, dim=-1) - 1) * IsOutput_position).to(torch.int64)

    NextMaskInput = 1 - IsOutput_position
    NextMaskOutput = torch.roll(NextMaskInput, shifts=-1, dims=-1)

    ProcessedInput1 = torch.gather(Input, dim=1, index=Input_position.unsqueeze(-1).expand(*Input_position.shape, Input.shape[-1]))
    ProcessedInput2 = torch.gather(Output, dim=1, index=Output_position.unsqueeze(-1).expand(*Output_position.shape, Output.shape[-1]))
    TOAInput1 = torch.gather(TOA_In, dim=1, index=Input_position)
    ProcessedInput2[..., -1] -= TOAInput1
    ProcessedOutput = torch.roll(ProcessedInput2, shifts=-1, dims=1)

    NoSequenceMask = 1 - (IsOutput_position + IsInput_position)

    return (ProcessedInput1, ProcessedInput2, ProcessedOutput, NextMaskInput, NextMaskOutput, NoSequenceMask)


if __name__ == '__main__':
    GetData(4, 6, 5, 5, 10, 15, 100, n_data_validation=1)