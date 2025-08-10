class Sensor:
    def __init__(self):
        self.name = 'sensor'

    def identify(self):
        print(self.name)

class Simu:
    def __init__(self):
        self.name = 'simu'
        self.sensor = Sensor()

    def use_sensor(self):
        print(self.name)
        self.sensor.identify()

class FreqSimu(Simu):
    def __init__(self):
        super().__init__()

    def use_sensor(self):
        print('freq ')
        super().use_sensor()

FreqSimu().use_sensor()

print('==================')

class LocalSimu(Simu):
    def __init__(self):
        super().__init__()
        self.sensor = Sensor()

    def use_sensor(self):
        print(self.name)
        print('osef du sensor ici')

LocalSimu().use_sensor()

class LocalFreqSimu(LocalSimu):
    pass

for name, value in vars(FreqSimu).items():
    if name != '__dict__' and name != '__weakref__':
        setattr(LocalFreqSimu, name, value)

LocalFreqSimu().use_sensor()
'''
je m'attends à avoir : 

"
freq
simu
osef du sensor ici
"

'''

