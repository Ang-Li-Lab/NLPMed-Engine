from typing import Any


def get_effective_param(
    instance_value: Any,  # noqa: ANN401
    provided_value: Any,  # noqa: ANN401
    *,
    required: bool = True,
) -> Any:  # noqa: ANN401
    if provided_value is not None:
        return provided_value

    if instance_value is not None:
        return instance_value

    if required:
        message = "A value must be provided."
        raise ValueError(message)

    return None
