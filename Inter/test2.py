class A:
    def Oi(self):
        return 'A'

class D:
    def Oi(self):
        return 'D'

class B(D):
    def Oi(self):
        return 'B' + super().Oi()

class C(B, A):
    def Oi(self):
        return 'C' + super().Oi()

print(C().Oi()) # CBD

class B(D):
    def Oi(self):
        return 'B' + super().Oi()

class C(B, A):
    def Oi(self):
        return 'C' + super().Oi()

print(C().Oi()) # CBA

class D:
    def Oi(self):
        return 'D' + super().Oi()

class B(D):
    def Oi(self):
        return 'B' + super().Oi()

class C(B, A):
    def Oi(self):
        return 'C' + super().Oi()

print(C().Oi()) # CBDA