from manim import *
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

def value_to_rgb(value, min_val=0, max_val=2, colormap='plasma'):
    # Normalize the value between 0 and 1
    normalized_value = (value - min_val) / (max_val - min_val)

    # Clip the normalized value to ensure it stays within [0, 1]
    normalized_value = np.clip(normalized_value, 0, 1)

    # Get the colormap
    cmap = plt.get_cmap(colormap)

    # Map the normalized value to an RGB color
    rgb = cmap(normalized_value)  # Returns an RGBA tuple, we need the RGB part

    return rgb

config.background_color = None  # Fond transparent
config.frame_width = 16.

class TransformToImage(Scene):
    def construct(self):
        num_input_squares = 10
        square_size = 7 / (num_input_squares // 2)
        scan_speed = 3  # unités par seconde
        wait_between = 1 / scan_speed
        drop_duration = 0.5

        # Création des carrés bleus d'entrée
        input_squares = VGroup(*[
            Square(side_length=square_size, fill_color=None, stroke_color=BLUE, stroke_width=8, fill_opacity=0.5).move_to(
                LEFT * 7 + RIGHT * ((i + 1 / 2) * square_size) +
                (4 - square_size) * UP
            )
            for i in range(num_input_squares)
        ])
        self.add(input_squares)

        # Barre rouge qui scanne
        bar = Line(UP , DOWN, color=RED).scale(square_size)
        bar_start = LEFT * 7 + (4 - square_size) * UP
        bar.move_to(bar_start)
        self.play(FadeIn(bar))

        # Génération aléatoire des rectangles cible (on pourra modifier ici)
        target_rectangles = []

        centers_freq = np.random.normal(0, 1, num_input_squares)
        dt = (0.5 * abs(np.random.normal(0, 1, num_input_squares - 1)))
        starting_time = np.cumsum([0] + dt.tolist())
        duration = (2 * abs(np.random.normal(0, 1, num_input_squares)))
        sensibility = 0.2
        level = np.random.normal(0, 1, num_input_squares)

        target_rect_scale = 12 / (starting_time[-1] + duration[-1])

        for i in range(num_input_squares):
            width = target_rect_scale * duration[i]
            height = sensibility
            N = 0.5 * np.tanh(level[i]) + 1
            r, g, b, a = value_to_rgb(N)
            r, g, b, a = int(r * 256), int(g * 256), int(b * 256), int(a * 256)
            pos = (LEFT * 7 +
                   RIGHT * target_rect_scale * (starting_time[i] + duration[i] / 2) +
                   centers_freq[i] * UP * target_rect_scale * 0.25)


            c = rgb_to_color((r, g, b))
            rect = Rectangle(width=width, height=height, fill_color=c, fill_opacity=1., stroke_color=BLACK).move_to(pos)
            target_rectangles.append(rect)

        self.rect_puls = []
        # Animation principale : duplication + transformation
        for i, source_square in enumerate(input_squares):
            # Déplacement de la barre vers le carré i
            target_x = source_square.get_center()[0]
            target_y = source_square.get_center()[1]
            self.play(bar.animate.move_to([target_x, target_y, 0]), run_time=wait_between, rate_func=linear)

            # Création d'une copie
            clone = source_square.copy()

            # Ajout sur scène
            self.add(clone)

            # Transformation en rectangle image
            self.play(
                Transform(clone, target_rectangles[i], run_time=drop_duration, rate_func=smooth)
            )

            self.rect_puls.append(clone)

        self.play(FadeOut(bar))

        time_checkpoint = (starting_time * target_rect_scale).tolist() + ((starting_time + duration) * target_rect_scale).tolist()
        time_checkpoint.sort()
        print(time_checkpoint)

        window_width = time_checkpoint[0]
        window = Rectangle(
            width=window_width, height=3, fill_color=None, stroke_color=RED, stroke_width=3, fill_opacity=0.).move_to(
            7 * LEFT + RIGHT * window_width / 2).set_z_index(1)
        self.play(FadeIn(window))
        bar = Line(UP, DOWN, color=RED).scale(window.height / 2)
        bar.move_to(7 * LEFT)
        self.play(FadeIn(bar))

        palier = []
        for i in range(len(time_checkpoint) - 1):
            li = []
            for j in range(len(starting_time)):
                if ((starting_time[j] * target_rect_scale <= time_checkpoint[i]) and
                        ((starting_time[j] + duration[j]) * target_rect_scale > time_checkpoint[i])):
                    li.append([centers_freq[j], np.random.rand(), 0, 0])

            if li == []:
                palier.append(torch.tensor(li))

                continue

            # We make a mask, BM, that is 0 only if the pulse is selected
            Frequencies = torch.matmul(torch.tensor(li).to(float), torch.tensor([1., 0, 0, 0]).to(float))
            Levels = torch.matmul(torch.tensor(li).to(float), torch.tensor([0., 1, 0, 0]).to(float))

            BF = torch.abs(Frequencies.unsqueeze(-1) - Frequencies.unsqueeze(-2)) < sensibility
            BN = (Levels.unsqueeze(-1) > Levels.unsqueeze(-2))
            BM = torch.sum(BF * BN, dim=-2) == 0

            # The selected pulses are then sorted by decreasing level
            Selected = torch.tensor(li)[BM]
            Levels = torch.matmul(Selected, torch.tensor([0., 1, 0, 0]).to(float))
            Orders = Levels.argsort(dim=-1, descending=True)[:5]

            palier.append(Selected[Orders].to(float))

        print(palier)
        local = []
        mesureur = []
        mesureur_shape = []
        TI = []
        TM = []
        e = 0
        output_square = [
            Square(side_length=square_size, fill_color=None, stroke_color=BLUE, stroke_width=8, fill_opacity=0.5).move_to(
                LEFT * 7 + RIGHT * ((i + 1 / 2) * square_size) +
                (4 - square_size) * DOWN
            )
            for i in range(num_input_squares)
        ]
        for i in range(len(time_checkpoint) - 1):
            self.play(bar.animate.move_to(7 * LEFT + RIGHT * time_checkpoint[i]), run_time=wait_between, rate_func=linear)

            window_width = time_checkpoint[i + 1] - time_checkpoint[i]
            target_window = Rectangle(
                width=window_width, height=window.height, fill_color=None, stroke_color=RED, stroke_width=3, fill_opacity=0.).move_to(
                7 * LEFT + RIGHT * (time_checkpoint[i] + window_width / 2))

            for rect in local:
                self.remove(rect)

            self.play(Transform(window, target_window),
                      rate_func=smooth,
                      run_time=0.5)

            local = []
            for p in palier[i]:
                freq = p[0]
                new_rect = Rectangle(
                    width=1, height=sensibility, fill_color=GREEN, fill_opacity=1.0).move_to(
                    window.get_left() + RIGHT * 1 / 2 + UP * float(freq) * 0.25 * target_rect_scale)
                local.append(new_rect)
                self.add(new_rect)

            if palier[i].shape[0] == 0:

                j = 0
                while j < len(TM):
                    # le vecteur de suivi n° j n'est plus mise-à-jour : l'information interceptée est complète
                    if time_checkpoint[i + 1] - TM[j] > 0.3:
                        exit_TI, exit_TM, exit_P, triangle = TI.pop(j), TM.pop(j), mesureur.pop(j), mesureur_shape.pop(j)
                        self.play(
                            Transform(triangle, output_square[e], rate_func=smooth, run_time=0.5)
                        )
                        e += 1
                    else:
                        j += 1
                continue

            if mesureur == []:
                for j in range(len(palier[i])):
                    mesureur.append(palier[i][j].tolist())
                    TI.append(time_checkpoint[i + 1])
                    TM.append(time_checkpoint[i])
                    freq = float(palier[i][j][0])
                    triangle = Triangle(fill_color=BLACK, stroke_color=BLACK, fill_opacity=1.).rotate(- PI / 2).scale(0.3).move_to(
                        window.get_left() + LEFT * 1 / 2 + UP * freq * 0.25 * target_rect_scale
                    )
                    self.add(triangle)
                    mesureur_shape.append(triangle)
                continue

            fV = torch.matmul(palier[i], torch.tensor([1., 0, 0, 0]).to(float))
            fP = torch.matmul(torch.tensor(mesureur).to(float), torch.tensor([1., 0, 0, 0]).to(float))
            lP = torch.matmul(torch.tensor(mesureur).to(float), torch.tensor([0., 1, 0, 0]).to(float))
            correlation = torch.abs(fV.unsqueeze(-1) - fP.unsqueeze(-2)) < sensibility
            m = len(mesureur)
            for k in range(len(palier[i])):
                selected_instance = []
                for j in range(m):
                    if correlation[k, j] and (TM[j] < time_checkpoint[i + 1]):
                        selected_instance.append(j)
                # traitement 1 : pas de corrélation entre V[i] et P : création d'un nouveau vecteur de suivi
                if selected_instance == []:
                    mesureur.append(palier[i][k].tolist())
                    TM.append(time_checkpoint[i + 1])
                    TI.append(time_checkpoint[i])
                    freq = float(palier[i][k][0])
                    triangle = Triangle(fill_color=BLACK, stroke_color=BLACK, fill_opacity=1.).rotate(- PI / 2).scale(0.3).move_to(
                        window.get_left() + LEFT * 1 / 2 + UP * freq * 0.25 * target_rect_scale
                    )
                    mesureur_shape.append(triangle)
                    self.add(triangle)
                # traitement 2 : corrélation entre V[i] et plusieurs vecteurs de suivi P[j1], P[j2] ...
                else:
                    # on détermine le vecteur de suivi avec corrélation de niveau le plus haut
                    Levels = lP[selected_instance]
                    o = selected_instance[torch.argmax(Levels)]
                    # on met à jour le vecteur de suivi P[k] correspondant
                    TM[o] = time_checkpoint[i + 1]
                    mesureur[o] = (torch.tensor(mesureur)[o] + (fV[k] - fP[o]) * torch.tensor([1., 0, 0, 0])).tolist()
                    freq = float(fV[k])
                    self.play(mesureur_shape[o].animate.move_to(
                        window.get_left() + LEFT * 1 / 2 + UP * freq * 0.25 * target_rect_scale
                    ), run_time=wait_between, rate_func=linear)

            j = 0
            while j < len(TM):
                # le vecteur de suivi n° j n'est plus mise-à-jour : l'information interceptée est complète
                if time_checkpoint[i + 1] - TM[j] > 0.3:
                    exit_TI, exit_TM, exit_P, triangle = TI.pop(j), TM.pop(j), mesureur.pop(j), mesureur_shape.pop(j)
                    self.play(
                        Transform(triangle, output_square[e], rate_func=smooth, run_time=0.5)
                    )
                    e += 1
                else:
                    j += 1

        for rect in local:
            self.remove(rect)
        self.play(FadeOut(window), FadeOut(bar))

        while mesureur != []:
            exit_P, triangle = mesureur.pop(), mesureur_shape.pop()
            self.play(
                Transform(triangle, output_square[e], rate_func=smooth, run_time=0.5)
            )
            e += 1

        self.wait(1)