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
        n_focus_in = 4
        n_tamp_in = 3
        n_focus_out = 5
        n_tamp_out = 3
        num_input_squares = 17
        square_size_in = 7 / ((num_input_squares + n_focus_in + n_tamp_in) // 2)
        square_size_out = 7 / ((num_input_squares + n_focus_out + n_tamp_out) // 2)
        scan_speed = 3  # unités par seconde
        wait_between = 1 / scan_speed
        drop_duration = 0.5
        e = 0

        # Création des carrés bleus d'entrée
        input_squares = VGroup(*[
            Square(side_length=square_size_in, fill_color=None, stroke_color=BLUE, stroke_width=8, fill_opacity=0.5).move_to(
                LEFT * 7 + RIGHT * ((i + 1 / 2 + n_tamp_in) * square_size_in) +
                (4 - square_size_in) * UP
            )
            for i in range(num_input_squares)
        ])
        self.add(input_squares)

        output_squares = [
            Square(side_length=square_size_out, fill_color=None, stroke_color=BLUE, stroke_width=8, fill_opacity=0.5).move_to(
                LEFT * 7 + RIGHT * ((i + 1 / 2 + n_tamp_out) * square_size_out) +
                (4 - square_size_out) * DOWN
            )
            for i in range(num_input_squares)
        ]

        # Barre rouge qui scanne
        bar = Line(UP , DOWN, color=RED).scale(0.7 * square_size_in)
        target_x = input_squares[0].get_center()[0]
        target_y = input_squares[0].get_center()[1]
        bar.move_to([target_x, target_y, 0])
        self.play(FadeIn(bar))

        # Génération aléatoire des rectangles cible (on pourra modifier ici)
        target_rectangles = []

        centers_freq = 1.5 * np.random.uniform(-1, 1, num_input_squares)
        dt = 0.5 * np.random.uniform(0.1, 1, num_input_squares - 1)
        starting_time = np.cumsum([0] + dt.tolist())
        duration = 0.5 * np.random.uniform(0.3, 2, num_input_squares)
        sensibility = 0.4
        level = np.random.normal(0, 1, num_input_squares)

        target_rect_scale = 13 / (starting_time[-1] + duration[-1])

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

        window_width = 0
        window = Rectangle(
            width=window_width, height=3, fill_color=None, stroke_color=RED, stroke_width=3, fill_opacity=0.).move_to(
            7 * LEFT + RIGHT * window_width / 2).set_z_index(1)
        self.play(FadeIn(window))

        local = []
        mesureur = []
        mesureur_shape = []
        TI = []
        TM = []

        i_checkpoint = 0
        time_checkpoint = (starting_time * target_rect_scale).tolist() + (
                    (starting_time + duration) * target_rect_scale).tolist()
        time_checkpoint.sort()
        tc_i = tc_o = 0
        self.rect_puls = []

        pos = 100 * LEFT
        focus_in = Rectangle(width=n_focus_in * square_size_in, height=square_size_in, fill_color=None, stroke_width=10, fill_opacity=0., stroke_color=RED).move_to(pos)
        tamp_in = Rectangle(width=n_tamp_in * square_size_in, height=square_size_in, fill_color=None, stroke_width=10, fill_opacity=0., stroke_color=GREEN).move_to(pos)
        focus_out = Rectangle(width=n_focus_out * square_size_out, height=square_size_out, fill_color=None, stroke_width=10, fill_opacity=0., stroke_color=RED).move_to(pos)
        tamp_out = Rectangle(width=n_tamp_out * square_size_out, height=square_size_out, fill_color=None, stroke_width=10, fill_opacity=0., stroke_color=GREEN).move_to(pos)
        foc_in_li = [
            Rectangle(width=square_size_in, height=square_size_in, fill_color=None, stroke_width=5,
                      fill_opacity=0., stroke_color=BLACK).move_to(pos) for _ in range(n_focus_in)
        ]
        tamp_in_li = [
            Rectangle(width=square_size_in, height=square_size_in, fill_color=None, stroke_width=5,
                      fill_opacity=0., stroke_color=BLACK).move_to(pos) for _ in range(n_tamp_in)
        ]
        foc_ou_li = [
            Rectangle(width=square_size_out, height=square_size_out, fill_color=None, stroke_width=5,
                      fill_opacity=0., stroke_color=BLACK).move_to(pos) for _ in range(n_focus_out)
        ]
        tamp_ou_li = [
            Rectangle(width=square_size_out, height=square_size_out, fill_color=None, stroke_width=5,
                      fill_opacity=0., stroke_color=BLACK).move_to(pos) for _ in range(n_tamp_out)
        ]

        tamp_out.set_z_index(2)
        focus_out.set_z_index(2)

        for x in foc_in_li + tamp_in_li + foc_ou_li + tamp_ou_li:
            x.set_z_index(3)
            self.add(x)



        self.add(focus_in, focus_out, tamp_out, tamp_in)


        # Animation principale : duplication + transformation
        for i, source_square in enumerate(input_squares):
            # Déplacement de la barre vers le carré i
            target_x = source_square.get_center()[0]
            target_y = source_square.get_center()[1]
            self.play(bar.animate.move_to([target_x, target_y, 0]), run_time=wait_between, rate_func=linear)

            if not i % n_focus_in:

                if i != 0:
                    self.play(
                        focus_in.animate.set_stroke(width=20),
                        tamp_in.animate.set_stroke(width=20),
                        focus_out.animate.set_stroke(width=20),
                        tamp_out.animate.set_stroke(width=20),
                        run_time=0.5
                    )

                    # Animation retour à l'état initial
                    self.play(
                        focus_in.animate.set_stroke(width=10),
                        tamp_in.animate.set_stroke(width=10),
                        focus_out.animate.set_stroke(width=10),
                        tamp_out.animate.set_stroke(width=10),
                        run_time=0.5
                    )

                focus_in.align_to(input_squares[i], LEFT)
                focus_in.align_to(input_squares[i], UP)
                tamp_in.next_to(focus_in, direction=LEFT, buff=0)
                focus_out.align_to(output_squares[e], LEFT)
                focus_out.align_to(output_squares[e], UP)
                tamp_out.next_to(focus_out, direction=LEFT, buff=0)
                couple = [(focus_in, foc_in_li), (focus_out, foc_ou_li), (tamp_in, tamp_in_li), (tamp_out, tamp_ou_li)]
                for x, li in couple:
                    li[0].align_to(x, LEFT)
                    li[0].align_to(x, UP)
                    for indice in range(len(li) - 1):
                        li[indice + 1].next_to(li[indice], RIGHT, buff=0)
                        li[indice + 1].align_to(li[indice], UP)

            # Création d'une copie
            clone = source_square.copy()

            # Ajout sur scène
            self.add(clone)

            # Transformation en rectangle image
            self.play(
                Transform(clone, target_rectangles[i], run_time=drop_duration, rate_func=smooth)
            )

            self.rect_puls.append(clone)

            print(target_rectangles[i].get_left()[0] + 7)
            print(starting_time[i] * target_rect_scale)
            while (
                    ((i != len(input_squares) - 1) and ((tc_o != starting_time[i] * target_rect_scale) and (i_checkpoint < len(time_checkpoint)))) or
                    ((i == len(input_squares) - 1) and (i_checkpoint < len(time_checkpoint)))
            ):
                tc_o = time_checkpoint[i_checkpoint]
                i_checkpoint += 1
                li = []
                for j in range(len(starting_time)):
                    if ((starting_time[j] * target_rect_scale <= tc_i) and
                            ((starting_time[j] + duration[j]) * target_rect_scale > tc_i)):
                        li.append([centers_freq[j], np.random.rand(), 0, 0])

                if li == []:
                    palier = torch.tensor(li)

                else:
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

                    palier = Selected[Orders].to(float)

                print('tc_i :', tc_i)
                print('tc_o :', tc_o)

                window_width = tc_o - tc_i
                target_window = Rectangle(
                    width=window_width, height=window.height, fill_color=None, stroke_color=RED, stroke_width=3,
                    fill_opacity=0.).move_to(
                    7 * LEFT + RIGHT * (tc_i + window_width / 2))

                for rect in local:
                    self.remove(rect)

                self.play(Transform(window, target_window),
                          rate_func=smooth,
                          run_time=0.5)

                local = []
                for p in palier:
                    freq = p[0]
                    new_rect = Rectangle(
                        width=1, height=sensibility, fill_color=GREEN, fill_opacity=1.0).move_to(
                        window.get_left() + RIGHT * 1 / 2 + UP * float(freq) * 0.25 * target_rect_scale)
                    local.append(new_rect)
                    self.add(new_rect)

                if palier.shape[0] == 0:

                    j = 0
                    while j < len(TM):
                        # le vecteur de suivi n° j n'est plus mise-à-jour : l'information interceptée est complète
                        if tc_o - TM[j] > 0.3:
                            exit_TI, exit_TM, exit_P, triangle = TI.pop(j), TM.pop(j), mesureur.pop(
                                j), mesureur_shape.pop(j)
                            self.play(
                                Transform(triangle, output_squares[e], rate_func=smooth, run_time=0.5)
                            )
                            e += 1
                        else:
                            j += 1
                    tc_i = tc_o
                    window_width = tc_o - tc_i
                    target_window = Rectangle(
                        width=window_width, height=window.height, fill_color=None, stroke_color=RED, stroke_width=3,
                        fill_opacity=0.).move_to(
                        7 * LEFT + RIGHT * (tc_i + window_width / 2))

                    for rect in local:
                        self.remove(rect)

                    self.play(Transform(window, target_window),
                              rate_func=smooth,
                              run_time=0.5)
                    continue

                if mesureur == []:
                    for j in range(len(palier)):
                        mesureur.append(palier[j].tolist())
                        TI.append(tc_o)
                        TM.append(tc_i)
                        freq = float(palier[j][0])
                        triangle = Triangle(fill_color=BLACK, stroke_color=BLACK, fill_opacity=1.).rotate(
                            - PI / 2).scale(0.3).move_to(
                            window.get_left() + LEFT * 1 / 2 + UP * freq * 0.25 * target_rect_scale
                        )
                        triangle.set_z_index(1)

                        self.add(triangle)
                        mesureur_shape.append(triangle)
                    tc_i = tc_o

                    window_width = tc_o - tc_i
                    target_window = Rectangle(
                        width=window_width, height=window.height, fill_color=None, stroke_color=RED, stroke_width=3,
                        fill_opacity=0.).move_to(
                        7 * LEFT + RIGHT * (tc_i + window_width / 2))

                    for rect in local:
                        self.remove(rect)

                    self.play(Transform(window, target_window),
                              rate_func=smooth,
                              run_time=0.5)
                    continue

                fV = torch.matmul(palier, torch.tensor([1., 0, 0, 0]).to(float))
                fP = torch.matmul(torch.tensor(mesureur).to(float), torch.tensor([1., 0, 0, 0]).to(float))
                lP = torch.matmul(torch.tensor(mesureur).to(float), torch.tensor([0., 1, 0, 0]).to(float))
                correlation = torch.abs(fV.unsqueeze(-1) - fP.unsqueeze(-2)) < sensibility
                m = len(mesureur)
                for k in range(len(palier)):
                    selected_instance = []
                    for j in range(m):
                        if correlation[k, j] and (TM[j] < tc_o):
                            selected_instance.append(j)

                    if selected_instance == []:
                        mesureur.append(palier[k].tolist())
                        TM.append(tc_o)
                        TI.append(tc_i)
                        freq = float(palier[k][0])
                        triangle = Triangle(fill_color=BLACK, stroke_color=BLACK, fill_opacity=1.).rotate(
                            - PI / 2).scale(0.3).move_to(
                            window.get_left() + LEFT * 1 / 2 + UP * freq * 0.25 * target_rect_scale
                        )
                        triangle.set_z_index(1)

                        mesureur_shape.append(triangle)
                        self.add(triangle)

                    else:
                        # on détermine le vecteur de suivi avec corrélation de niveau le plus haut
                        Levels = lP[selected_instance]
                        o = selected_instance[torch.argmax(Levels)]
                        # on met à jour le vecteur de suivi P[k] correspondant
                        TM[o] = tc_o
                        mesureur[o] = (
                                    torch.tensor(mesureur)[o] + (fV[k] - fP[o]) * torch.tensor([1., 0, 0, 0])).tolist()
                        freq = float(fV[k])
                        self.play(mesureur_shape[o].animate.move_to(
                            window.get_left() + LEFT * 1 / 2 + UP * freq * 0.25 * target_rect_scale
                        ), run_time=wait_between, rate_func=linear)

                j = 0
                while j < len(TM):

                    if tc_o - TM[j] > 0.3:
                        exit_TI, exit_TM, exit_P, triangle = TI.pop(j), TM.pop(j), mesureur.pop(j), mesureur_shape.pop(
                            j)
                        self.play(
                            Transform(triangle, output_squares[e], rate_func=smooth, run_time=0.5)
                        )
                        e += 1
                    else:
                        j += 1

                tc_i = tc_o

                window_width = tc_o - tc_i
                target_window = Rectangle(
                    width=window_width, height=window.height, fill_color=None, stroke_color=RED, stroke_width=3,
                    fill_opacity=0.).move_to(
                    7 * LEFT + RIGHT * (tc_i + window_width / 2))

                for rect in local:
                    self.remove(rect)

                self.play(Transform(window, target_window),
                          rate_func=smooth,
                          run_time=0.5)

        if not (i+1) % n_focus_in:

            self.play(
                focus_in.animate.set_stroke(width=20),
                tamp_in.animate.set_stroke(width=20),
                focus_out.animate.set_stroke(width=20),
                tamp_out.animate.set_stroke(width=20),
                run_time=0.5
            )

            # Animation retour à l'état initial
            self.play(
                focus_in.animate.set_stroke(width=10),
                tamp_in.animate.set_stroke(width=10),
                focus_out.animate.set_stroke(width=10),
                tamp_out.animate.set_stroke(width=10),
                run_time=0.5
            )

            focus_in.next_to(input_squares[i], RIGHT, buff=0)
            focus_in.align_to(input_squares[i], UP)
            tamp_in.next_to(focus_in, direction=LEFT, buff=0)
            focus_out.align_to(output_squares[e], LEFT)
            focus_out.align_to(output_squares[e], UP)
            tamp_out.next_to(focus_out, direction=LEFT, buff=0)
            couple = [(focus_in, foc_in_li), (focus_out, foc_ou_li), (tamp_in, tamp_in_li), (tamp_out, tamp_ou_li)]
            for x, li in couple:
                li[0].align_to(x, LEFT)
                li[0].align_to(x, UP)
                for indice in range(len(li) - 1):
                    li[indice + 1].next_to(li[indice], RIGHT, buff=0)
                    li[indice + 1].align_to(li[indice], UP)

        for rect in local:
            self.remove(rect)
        self.play(FadeOut(window, bar))

        while mesureur != []:
            exit_P, triangle = mesureur.pop(), mesureur_shape.pop()
            self.play(
                Transform(triangle, output_squares[e], rate_func=smooth, run_time=0.5)
            )
            e += 1

        self.play(
            focus_in.animate.set_stroke(width=20),
            tamp_in.animate.set_stroke(width=20),
            focus_out.animate.set_stroke(width=20),
            tamp_out.animate.set_stroke(width=20),
            run_time=0.5
        )

        # Animation retour à l'état initial
        self.play(
            focus_in.animate.set_stroke(width=10),
            tamp_in.animate.set_stroke(width=10),
            focus_out.animate.set_stroke(width=10),
            tamp_out.animate.set_stroke(width=10),
            run_time=0.5
        )

        self.wait(1)