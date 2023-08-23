import os
import numpy as np


# Cette fonction ne prend pas en compte le chargement du mode évaluation pour l'instant
def FDTDataLoader(ListTypeData=[], local='', variables_dict={}):

    NewArg = []
    NewValue = []
    for TypeData in ['Training', 'Validation']:
        if TypeData in ListTypeData:
            Source, Translation = ParallelLoading(TypeData, local)

            NewArg.append(TypeData + 'Source')
            NewValue.append(Source)
            NewArg.append(TypeData + 'Translation')
            NewValue.append(Translation)

    for i in range(len(NewArg)):
        variables_dict.__setitem__(NewArg[i], NewValue[i])

def load_data(filename, result_queue):
    data = np.load(filename)  # Load data from filename
    result_queue.put(data)
def ParallelLoading(TypeData, local):
    import multiprocessing

    PDWsFileNames = []
    PulsesFileNames = []
    for filename in os.listdir(os.path.join(local, 'FakeDigitalTwinTranslator', 'Bursts', 'BurstsData', 'Data', TypeData)):
        if 'PDWsDCI' in filename:
            PDWsFileNames.append(filename)
        if 'PulsesAnt' in filename:
            PulsesFileNames.append(filename)

    result_queue = multiprocessing.Queue()

    processes = []
    for filename in PulsesFileNames:
        process = multiprocessing.Process(target=load_data, args=(filename, result_queue))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    Source_list = [result_queue.get() for _ in PulsesFileNames]
    Source = np.concatenate(Source_list, axis=0)

    processes = []
    for filename in PDWsFileNames:
        process = multiprocessing.Process(target=load_data, args=(filename, result_queue))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    Translation_list = [result_queue.get() for _ in PDWsFileNames]
    Translation = np.concatenate(Translation_list, axis=0)

    # from concurrent.futures import ProcessPoolExecutor
    #
    # with ProcessPoolExecutor() as executor:
    #     Source_list = list(executor.map(np.load, PulsesFileNames))
    # Source = np.concatenate(Source_list, axis=0)
    #
    # with ProcessPoolExecutor() as executor:
    #     Translation_list = list(executor.map(np.load, PDWsFileNames))
    # Translation = np.concatenate(Translation_list, axis=0)

    return Source, Translation

def LoadParam(dict, variables_dict):
    for key in dict.keys():
        variables_dict.__setitem__(key, dict[key])


