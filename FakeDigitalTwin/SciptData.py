from FakeDigitalTwin.Experience import MakeData

MakeData(Batch_size=3000, seed=1, name='Training')
MakeData(Batch_size=100, seed=2, name='Validation')
MakeData(Batch_size=50, seed=3, name='Evaluation')

