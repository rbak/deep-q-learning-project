from unityagents import UnityEnvironment


class Environment():
    """Learning Environment."""

    def __init__(self, file_name="environments/Banana.app", no_graphics=True):
        """Initialize parameters and build model.
        Params
        ======
            file_name (string): unity environment file
            no_graphics (boolean): Start environment with graphics
        """
        self.env = UnityEnvironment(file_name=file_name, no_graphics=no_graphics)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.action_space_size = self.brain.vector_action_space_size

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.env.close()

    def reset(self, train_mode=False):
        return self.env.reset(train_mode=train_mode)[self.brain_name]

    def step(self, action):
        return self.env.step(action)[self.brain_name]
