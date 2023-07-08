from FakeDigitalTwin.XMLTools import loadXmlAsObj
from FakeDigitalTwinTranslator.Network import TransformerTranslator
import os

# local = r'C:\\Users\\matth\\OneDrive\\Documents\\Python\\Projets'
local = r'C:\Users\Matthieu\Documents\Python\Projets'
PulsesAnt = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', 'PulsesAnt.xml'))
PDWsDCI = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', 'PDWsDCI.xml'))

PulsesAntList = [[Pulse['TOA'], Pulse['LI'], Pulse['Level'], Pulse['FreqStart'], Pulse['FreqEnd']] for Pulse in PulsesAnt]
PDWsDCIList = [[PDW['TOA'], PDW['LI'], PDW['Level'], PDW['FreqMin'], PDW['FreqMax'],
                int('CW' in PDW['flags']), int('TroncAv' in PDW['flags']), int(len(PDW['flags']) == 0)] for PDW in PDWsDCI]

d_source = 5
d_target = 8
d_input_Enc = 32
d_input_Dec = 32
d_att = 32
num_heads = 4

Translator = TransformerTranslator(d_source=d_source, d_target=d_target, d_input_Enc=d_input_Enc, d_input_Dec=d_input_Dec, num_heads=num_heads)

