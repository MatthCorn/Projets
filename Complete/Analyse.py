from Tools.XMLTools import loadXmlAsObj
import os
import torch
import matplotlib.pyplot as plt

# Ce script sert à analyser les paramètres des IA entrainées

local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'OneDrive', 'Documents', 'Python', 'Projets')
# local = os.path.join(os.path.abspath(os.sep), 'Users', 'matth', 'Documents', 'Python', 'Projets')

type1 = 'TypeClassic'
type2 = 'TypeTrackerInspired'
save_path = os.path.join(local, 'Complete', type1, 'Save')
d_att = 256

mat = torch.zeros(5, 0)
for dir in os.listdir(save_path):
    print(dir)

    try:
        D = os.listdir(os.path.join(save_path, dir))[-1]
        param = loadXmlAsObj(os.path.join(save_path, dir, D, 'param'))
        dico = torch.load(os.path.join(save_path, dir, D, 'Translator'))
        print(param['d_att'])
        if param['d_att'] == d_att:
            if D == 'D_0.2':
                print('y')
        mat = torch.cat((mat, dico['source_embedding.linears.0.weight'].cpu().t()), dim=1)

    except:
        None

mat = torch.abs(mat)
mat_mean = torch.mean(mat, dim=1)
mat = mat/mat_mean.unsqueeze(-1)

plt.imshow(torch.cov(mat).numpy())

plt.colorbar()
plt.show()
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    plt.close()

    p1 = np.array([1, 1])
    p2 = np.array([1, -1])
    p3 = np.array([-1, -1])
    p4 = np.array([-1, 1])
    ord = 20
    radius = 0
    m1 = []
    while len(m1) < 1000:
        X = np.random.uniform(-1, 1, size=2)
        if np.linalg.norm(p1 - X, ord=ord) > radius and \
                np.linalg.norm(p2 - X, ord=ord) > radius and \
                np.linalg.norm(p3 - X, ord=ord) > radius and \
                np.linalg.norm(p4 - X, ord=ord) > radius:
            m1.append(X)

    m1 = np.array(m1)

    rot = np.array([[np.sqrt(2)/2, np.sqrt(2)/2], [-np.sqrt(2)/2, np.sqrt(2)/2]])
    m2 = np.matmul(m1, rot)

    m1, m2 = np.abs(m1), np.abs(m2)

    plt.scatter(m1[:, 0], m1[:, 1])
    plt.scatter(m2[:, 0], m2[:, 1])
    plt.show()

    print(np.cov(m1.transpose()))
    print(np.cov(m2.transpose()))

