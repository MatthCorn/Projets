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

# d_att = 256
#
# mat = torch.zeros(5, 0)
# for dir in os.listdir(save_path):
#     print(dir)
#
#     try:
#         D = os.listdir(os.path.join(save_path, dir))[-1]
#         param = loadXmlAsObj(os.path.join(save_path, dir, D, 'param'))
#         dico = torch.load(os.path.join(save_path, dir, D, 'Translator'))
#         print(param['d_att'])
#         if param['d_att'] == d_att:
#             if D == 'D_0.2':
#                 print('y')
#         mat = torch.cat((mat, dico['source_embedding.linears.0.weight'].cpu().t()), dim=1)
#
#     except:
#         None


dico = torch.load(os.path.join(save_path, '2023-12-04__13-15', 'D_3', 'Translator'))



mat1 = dico['source_embedding.linears.0.weight'].cpu().t()
info = torch.cov(mat1)
# info = torch.matmul(torch.abs(mat1)/(torch.norm(mat1, p=1, dim=-1).unsqueeze(-1)), (torch.abs(mat1)/(torch.norm(mat1, p=1, dim=-1).unsqueeze(-1))).t())
# info = torch.cov(torch.abs(mat1))


plt.imshow(info.numpy())
plt.colorbar()
plt.show()


mat2 = dico['target_embedding.linears.0.weight'].cpu().t()
info = torch.cov(mat2)
# info = torch.matmul(torch.abs(mat2)/(torch.norm(mat2, p=1, dim=-1).unsqueeze(-1)), (torch.abs(mat2)/(torch.norm(mat2, p=1, dim=-1).unsqueeze(-1))).t())
# info = torch.cov(torch.abs(mat2))
plt.imshow(info.numpy())
plt.colorbar()
plt.show()

# dico = torch.load(os.path.join(save_path, '2023-12-04__09-56', 'D_3', 'Translator'))
# mat1 = torch.abs(dico['source_embedding.linears.0.weight'].cpu().t())
# mat2 = torch.abs(dico['target_embedding.linears.0.weight'].cpu().t())
#
#
# plt.imshow(torch.cov(mat1).numpy())
# plt.colorbar()
# plt.show()
#
# plt.imshow(torch.cov(mat2).numpy())
# plt.colorbar()
# plt.show()

if False: #__name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    plt.close()

    p1 = np.array([1, 1])
    p2 = np.array([1, -1])
    p3 = np.array([-1, -1])
    p4 = np.array([-1, 1])
    ord = 20
    radius = 0.95
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

    print(np.cov(m1.transpose()))
    print(np.cov(m2.transpose()))

    plt.scatter(m1[:, 0], m1[:, 1])
    plt.scatter(m2[:, 0], m2[:, 1])
    plt.show()

    m1, m2 = np.abs(m1), np.abs(m2)

    plt.scatter(m1[:, 0], m1[:, 1])
    plt.scatter(m2[:, 0], m2[:, 1])
    plt.show()

    print(np.cov(m1.transpose()))
    print(np.cov(m2.transpose()))

