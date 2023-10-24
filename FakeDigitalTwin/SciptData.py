from FakeDigitalTwin.Experience import MakeData

def MakeSets(density):
    MakeData(Batch_size=3000, seed=1, density=density, name='Training')
    MakeData(Batch_size=100, seed=2, density=density, name='Validation')
    MakeData(Batch_size=50, seed=3, density=density, name='Evaluation')

if __name__ == '__main__':
    MakeSets(density=0.5)
