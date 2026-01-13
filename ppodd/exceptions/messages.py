EM_CANNOT_INIT_MODULE = (
    "Cannot initialise module {module_name} as one or more of the constants "
    "({constants}) are not defined on the controlling Dataset. These are "
    "required at initialization time. This operation cannot be deferred. "
    "ensure these constants are set if this module is required."
)

EM_CANNOT_RUN_MODULE_CONSTANTS = (
    "Cannot run module {module_name} as one or more of the constants "
    "({constants}) are not defined on the controlling Dataset. These "
    "are required at run time. ensure these constants are set if this "
    "module is required."
)
