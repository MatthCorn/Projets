class Hooker():
    def __init__(self, model):
        self.model = model

        self.hooker = None

        self.tensors = None

        self.InitHooker()

    def InitHooker(self):
        self.hooker = self.model.HookMatmul.register_forward_hook(self.GetAllPDWs)

    def RemoveHooker(self, reset=True):
        self.hooker.remove()
        if reset:
            self.hooker = None
            self.tensors = None

    def GetAllPDWs(self, module, input, output):
        # on veut enregistrer tous les PDWs avant sélection par le decider
        self.tensors = output[:, :, :-1]
