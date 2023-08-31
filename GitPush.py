from git import Repo

PATH_OF_GIT_REPO = r'C:\Users\matth\Documents\Python\Projets\.git'  # make sure .git folder is properly configured
COMMIT_MESSAGE = 'test push script'

def git_push(file):
    try:
        repo = Repo(PATH_OF_GIT_REPO)
        repo.git.add([file], update=True)
        repo.index.commit(COMMIT_MESSAGE)
        origin = repo.remote(name='origin')
        origin.push()
    except:
        print('Some error occured while pushing the code')

git_push(r'C:\Users\matth\Documents\Python\Projets\GitPush.py')