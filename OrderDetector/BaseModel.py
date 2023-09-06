import numpy as np

# Ce script sert à donner la classe de chacun des vecteurs d'une liste de NVec vecteurs. La classe correpond à leur position suivant un ordre définit ici-bas.
# si V1 et V2 sont deux vecteurs, V1>V2 ssi V1[0] + V1[1] > V2[0] + V2[1]
def GetOrders(L):
    NVec = len(L)
    Values = [(L[i][0] + L[i][1], i) for i in range(len(L))]
    Values.sort()
    Orders = np.zeros((NVec, NVec))
    for i in range(NVec):
        j = Values[i][1]
        Orders[j, i] = 1
    return Orders

if __name__ == '__main__':
    L = [[5, 0], [2, 0], [1, 0], [4, 0], [3, 0]]
    Orders = GetOrders(L)
    print(Orders)