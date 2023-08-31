from git import Repo
import os

local = r'C:\Users\matth\Documents\Python\Projets'
# local = r'C:\Users\matth\OneDrive\Documents\Python\Projets'

COMMIT_MESSAGE = 'Rien à dire'

def git_push(local, file, CommitMsg=COMMIT_MESSAGE):
    try:
        repo = Repo(os.path.join(local, r'.git'))
        repo.git.add([os.path.join(local, file)])
        repo.index.commit(CommitMsg)
        origin = repo.remote(name='main')
        origin.push()
    except:
        print('Some error occured while pushing the code')

if __name__ == '__main__':
    git_push(local=local, file=r'GitPush.py')