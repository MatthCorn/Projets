################################################################################################################################
# De base, nous avons une classe B qui hérite d'une classe A
class A:
    def Oi(self):
        return 'A'

class B(A):
    def Oi(self):
        return 'B' + super().Oi()

print(B().Oi()) # BA
################################################################################################################################

################################################################################################################################
class D(A):
    def Oi(self):
        return 'D' + super().Oi()

# On voudrait remplacer que B hérite d'une autre sous-classe de A à la place : D
# mais on ne veut pas réécrire B comme ici

class B(D):
    def Oi(self):
        return 'B' + super().Oi()

print(B().Oi()) # CBA
################################################################################################################################


################################################################################################################################
# à la place, on écrit la sous-classe B comme n'héritant de personne
# et on l'utilisera comme template pour la variante B(A) et B(D)

class B:
    def Oi(self):
        return 'B' + super().Oi()

class BA(B, A):
    pass

class BD(B, D):
    pass

print(BA().Oi()) # BA
print(BD().Oi()) # BA
