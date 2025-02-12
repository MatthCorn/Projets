from git import Repo
import os

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')

COMMIT_MESSAGE = 'Rien à dire'

def git_push(local, save_path, CommitMsg=COMMIT_MESSAGE):
    try:
        repo = Repo(os.path.join(local, r'.git'))
        repo.git.add([save_path])
        repo.index.commit(CommitMsg)
        origin = repo.remote(name='origin')
        origin.pull(rebase=True)
        origin.push()
    except Exception as e:
        print(f"Une erreur est survenue: {e}")


if __name__ == '__main__':
    git_push(local=local, file=r'Tools.GitPush.py')