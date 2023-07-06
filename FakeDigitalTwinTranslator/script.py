from FakeDigitalTwin.XMLTools import loadXmlAsObj
import os

local = r'C:\\Users\\matth\\OneDrive\\Documents\\Python\\Projets'
PulsesAnt = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', 'PulsesAnt.xml'))
PDWsDCI = loadXmlAsObj(os.path.join(local, 'FakeDigitalTwin', 'Data', 'PDWsDCI.xml'))