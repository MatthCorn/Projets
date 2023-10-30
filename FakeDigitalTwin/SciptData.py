from FakeDigitalTwin.Experience import MakeData

def MakeSets(density, seed=None):
    if seed is not None:
        s1, s2, s3 = seed, seed + 1, seed + 2
    else:
        s1, s2, s3 = seed, seed, seed
    MakeData(Batch_size=6000, seed=s1, density=density, name='Training')
    MakeData(Batch_size=100, seed=s2, density=density, name='Validation')
    MakeData(Batch_size=50, seed=s3, density=density, name='Evaluation')

if __name__ == '__main__':
    MakeSets(density=0.5)
