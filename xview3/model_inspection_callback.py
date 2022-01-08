from catalyst.core import Callback, CallbackOrder


class InspectorCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.optimizer + 1)

    def on_batch_end(self, runner: "IRunner"):
        pass
