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

        # Si des fichiers sont modifiés, on stash avant de récupérer les mises à jour
        try:
            repo.git.stash('push', '--keep-index')
        except Exception as e:
            print(f"Une erreur est survenue lors du stash push: {e}")

        # Récupérer les mises à jour sans modifier la copie de travail
        origin.fetch()

        try:
            repo.git.rebase('origin/main')  # Rebase sécurisé
        except Exception as e:
            print(f"Une erreur est survenue lors du rebase: {e}")
            repo.git.reset('--hard', 'origin/main')

        # Tenter de récupérer le stash (si existant)
        try:
            repo.git.stash('pop')
        except Exception as e:
            print(f"Une erreur est survenue lors du stash pop: {e}")

        try:
            origin.push()
        except Exception as e:
            print(f"Une erreur est survenue lors du push: {e}")
            origin.push(force=True)

    except Exception as e:
        print(f"Une erreur est survenue: {e}")



if __name__ == '__main__':
    git_push(local=local, file=r'Tools.GitPush.py')