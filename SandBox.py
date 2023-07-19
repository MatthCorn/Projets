from SandBoxAPI import test

NameList = ["fifi", "riri", "loulou"]
for i in range(len(NameList)):
    exec(NameList[i] + "= i")
print(type(fifi))

vars().__setitem__('salut', 13)

def load(fun):
    name, value = fun()
    print(name)
    return name

print('name :', load(test))
load(test)

foo = 4
print(f'{foo}')
