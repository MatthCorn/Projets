from FakeDigitalTwin.Meter import Meter
from FakeDigitalTwin.Pulse import Pulse
from FakeDigitalTwin.Plateau import Plateau
from FakeDigitalTwin.Processor import Processor
from time import time

class DigitalTwin():
    def __init__(self, Param):
        self.Param = Param
        # Making meters to follow the pulses through the plateaus, all initially free
        self.Meters = []
        for i in range(Param['N_mesureurs_max']):
            self.Meters.append(Meter(Id=str(i), Parent=self))

        # We use only one object to represente the plateaus, this object is updated to represent a change of plateau
        self.Plateau = Plateau(Parent=self)

        self.Processor = Processor(Parent=self)
        self.PDWs = []

        self.TrackingCounter = 0
        self.PlateauCounter = 0
        self.PlateauUpdateCounter = 0
        self.NADCounter = 0
        self.NbPulseInPlateau = 0

    def forward(self, AntPulses=None):
        if AntPulses is not None:
            self.AntPulses = AntPulses.copy()

        self.Running = True

        # This loop is in charge of making the operation on each Plateau
        while self.Running:
            self.PlateauCounter += 1
            self.PlateauProcessing()

        self.NbPulseInPlateau = self.NbPulseInPlateau/self.PlateauCounter
        self.PlateauUpdateCounter = self.PlateauUpdateCounter/self.PlateauCounter
        self.TrackingCounter = self.TrackingCounter/self.PlateauCounter
        self.NADCounter = self.NADCounter/self.PlateauCounter

        if self.Param['PDW_tries']:
            self.Tplist = [x[1] for x in self.PDWs]
            self.PDWs = [x[0] for x in self.PDWs]



    def PlateauProcessing(self):

        # We load the pulses of the upcoming plateau
        t = time()
        self.Plateau.Next()
        self.PlateauUpdateCounter += time() - t

        self.NbPulseInPlateau += len(self.Plateau.Pulses)

        # print('\n')
        # print('starting time :', self.Plateau.StartingTime)
        # print('\n')
        # print('current pulses :', self.Plateau.Pulses)
        # print('\n')
        # print('ending time :', self.Plateau.EndingTime)
        # print('\n')

        # If the plateau is empty
        if self.Plateau.IsEmpty():
            # we release all opened meters
            t = time()
            for meter in self.Meters:
                if meter.IsOpen:
                    meter.Release(self.Plateau.StartingTime)
            self.TrackingCounter += time() - t

        else:
            # We compute the NAD on the current plateau
            t = time()
            NAD = self.Processor.FreqAmbRemoval(self.Plateau)
            self.NADCounter += time() - t

            # We link the NAD to the meters and start updating them
            t = time()
            self.Processor.LinkMeterNAD(NAD, self.Meters, self.Plateau)

            # We end updating the meters by making the maintenance
            for meter in self.Meters:
                if meter.InMaintenance:
                    meter.Maintenance()
            self.TrackingCounter += time() - t

        # print('meters:', [el for el in self.Meters if el.IsOpen])
        # print('\n')
        # print('meters Ids:', [el.Id for el in self.Meters if el.IsOpen])
        # print('\n')
        # print('PDWs:', self.PDWs)
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        # print('\n')
        # print('\n')
        # print('\n')

    def AddPDW(self, PDW, Tp=0):
        if self.Param['PDW_tries']:
            TpList = [x[1] for x in self.PDWs]

            # We're looking for the place of the new pulse in self.PDWs emitted at the time Tp
            # It is more likely to be added at the end of self.PDWs
            if self.PDWs == []:
                self.PDWs.append([PDW, Tp])

            else:
                for i in reversed(range(len(self.PDWs))):
                    if TpList[i] < Tp:
                        self.PDWs.insert(i+1, [PDW, Tp])
                        break

        else:
            self.PDWs.append(PDW)


if __name__ == '__main__':
    import numpy as np
    TOA = 0
    Dt = 0
    AntP = []
    for k in range(30):
        DF = 0.1 * np.random.random() - 0.05
        F = 10 + 0.5 * np.random.random()
        AntP.append(Pulse(TOA=TOA, LI=0.5 + 2 * np.random.random(), FreqStart=F, FreqEnd=F + DF, Level=5.5 * np.random.random()))
        TOA += np.random.random()

    Param = {
        'Fe_List': [5.1, 5, 4.9, 4.8],
        'Duree_max_impulsion': 4,
        'Seuil_mono': 10,
        'Seuil_harmo': 8,
        'Seuil_IM': 8,
        'Seuil_sensi_traitement': 6,
        'Seuil_sensi': 1,
        'Contraste_geneur': 0.2,
        'Nint': 500,
        'Contraste_geneur_2': 1,
        'M1_aveugle': 2,
        'M2_aveugle': 2,
        'M_local': 5,
        'N_DetEl': 12,
        'Seuil_ecart_freq': 5e-3,
        'Duree_maintien_max': 0.2,
        'N_mesureurs_max': 8,
        'PDW_tries': True,
    }

    DT = DigitalTwin(Param)
    DT.forward(AntPulses=AntP)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import random

    def value_to_rgb(value, min_val=0, max_val=5.5, colormap='plasma'):
        # Normalize the value between 0 and 1
        normalized_value = (value - min_val) / (max_val - min_val)

        # Clip the normalized value to ensure it stays within [0, 1]
        normalized_value = np.clip(normalized_value, 0, 1)

        # Get the colormap
        cmap = plt.get_cmap(colormap)

        # Map the normalized value to an RGB color
        rgb = cmap(normalized_value)  # Returns an RGBA tuple, we need the RGB part

        return rgb

    fig, ax = plt.subplots(1)

    L = AntP

    for pulse in L:
        T1 = pulse.TOA
        T2 = T1 + pulse.LI
        N = pulse.Level
        r, g, b = random.random(), random.random(), random.random()
        Rectangle = Polygon(((T1, 0), (T1, N), (T2, N), (T2, 0)), fc=(r, g, b, 0.1), ec=(0, 0, 0, 1), lw=2)
        ax.add_artist(Rectangle)

    plt.show()

    df = Param['M_local'] * Param['Fe_List'][1] / Param['Nint']

    from matplotlib import colors
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

    L = AntP
    for pulse in L:
        T1 = pulse.TOA
        T2 = T1 + pulse.LI
        F1 = pulse.FreqStart
        F2 = pulse.FreqEnd
        DF = abs(pulse.FreqStart - pulse.FreqEnd) + df
        N = pulse.Level
        sommets = [
            [T1, F1, N],
            [T2, F2, N],
            [T1, F1-df, 0],
            [T1, F1+df, 0],
            [T2, F2+df, 0],
            [T2, F2-df, 0]
        ]

        surf = np.array([[sommets[0], sommets[0], sommets[0], sommets[0]],
                         [sommets[0], sommets[2], sommets[3], sommets[0]],
                         [sommets[1], sommets[5], sommets[4], sommets[1]],
                         [sommets[1], sommets[1], sommets[1], sommets[1]]])

        # Plot the surface
        r, g, b, a = value_to_rgb(N)
        ax1.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))



    R = DT.PDWs
    for pulse in R:
        T1 = pulse['TOA']
        T2 = T1 + pulse['LI']
        F = (pulse['FreqMin'] + pulse['FreqMax']) / 2
        DF = abs(pulse['FreqMin'] - pulse['FreqMax']) + df
        N = pulse['Level']

        sommets = [
            [T1, F, N],
            [T2, F, N],
            [T1, F-df, 0],
            [T1, F+df, 0],
            [T2, F+df, 0],
            [T2, F-df, 0]
        ]

        surf = np.array([[sommets[0], sommets[0], sommets[0], sommets[0]],
                         [sommets[0], sommets[2], sommets[3], sommets[0]],
                         [sommets[1], sommets[5], sommets[4], sommets[1]],
                         [sommets[1], sommets[1], sommets[1], sommets[1]]])

        # Plot the surface
        r, g, b, a = value_to_rgb(N)
        ax2.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], color=(r, g, b, a))


    # Define the colormap and normalization (vmin, vmax)
    cmap = plt.get_cmap('plasma')  # You can choose 'plasma', 'coolwarm', etc.
    norm = colors.Normalize(vmin=0, vmax=5.5)  # Define the range for the colorbar

    # Create a ScalarMappable to be used for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Optional, required in some cases to avoid warnings

    # Add the colorbar to the figure without any plot
    fig.colorbar(sm, ax=ax2, shrink=0.4)
    fig.colorbar(sm, ax=ax1, shrink=0.4)

    ax1.set_xlim(-0.1, 11)
    ax1.set_ylim(9.9, 10.6)
    ax1.set_zlim(0, 5.5)
    ax1.set_box_aspect([2, 2, 0.5])
    ax1.zaxis.set_ticks([])  # Remove z-axis ticks
    ax1.zaxis.set_ticklabels([])  # Remove z-axis tick labels
    ax1.zaxis.label.set_visible(False)  # Hide z-axis label
    ax1.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax1.set_xlabel('temps')
    ax1.set_ylabel('fréquence')
    ax1.set_title("Impulsions d'entrée")
    ax1.set_proj_type('ortho')
    ax1.view_init(elev=90, azim=-90)
    ax2.set_xlim(-0.1, 11)
    ax2.set_ylim(9.9, 10.6)
    ax2.set_zlim(0, 5.5)
    ax2.set_box_aspect([2, 2, 0.5])
    ax2.zaxis.set_ticks([])  # Remove z-axis ticks
    ax2.zaxis.set_ticklabels([])  # Remove z-axis tick labels
    ax2.zaxis.label.set_visible(False)  # Hide z-axis label
    ax2.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax2.set_xlabel('temps')
    ax2.set_ylabel('fréquence')
    ax2.set_title("Impulsions de sortie")
    ax2.set_proj_type('ortho')
    ax2.view_init(elev=90, azim=-90)
    plt.show()

    print('end')