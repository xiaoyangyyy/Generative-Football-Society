class PluginError(Exception):
    pass


class DuplicatePluginError(PluginError):
    pass


class MissingDependencyError(PluginError):
    pass


class DependencyCycleError(PluginError):
    pass


class CapabilityError(PluginError):
    pass


class PluginStateError(PluginError):
    pass


class PluginActivationError(PluginError):
    def __init__(self, plugin_id: str):
        self.plugin_id = plugin_id
        super().__init__(f"plugin activation failed: {plugin_id}")
