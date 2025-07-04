"""Types used throughout the project."""

from typing import Dict, List, Union,TypeAlias
import typing as t
from numpy.typing import NDArray
from transformers.trainer_utils import EvalPrediction

if t.TYPE_CHECKING:
    from .data_models import GenerativeModelOutput


# --- Basic metric types ---
RawMetric: TypeAlias = Dict[str, float]
RawList: TypeAlias = List[RawMetric]
TotalMetric: TypeAlias = Dict[str, float]
# --- Result Entry Type (for one learning rate/config) ---
ResultEntry: TypeAlias = Dict[str, Union[RawList, TotalMetric]]  # expects 'raw' and 'total' keys
# --- Multi-config format: learning_rate -> ResultEntry ---
MultiResultFormat: TypeAlias = Dict[str, ResultEntry]
# --- Single-config format with metadata ---
SingleResultWithMeta: TypeAlias = Dict[str, Union[
    RawList,
    TotalMetric,
    int,
    bool,
    str,
    None
]]
ScoreDict: TypeAlias = Union[MultiResultFormat, SingleResultWithMeta]
class ComputeMetricsFunction(t.Protocol):
    """A function used to compute the metrics."""

    def __call__(
        self,
        model_outputs_and_labels: EvalPrediction
        | tuple[
            NDArray | list[str] | list[list[str]], NDArray | list[str] | list[list[str]]
        ],
    ) -> dict[str, float]:
        """Compute the metrics.

        Args:
            model_outputs_and_labels:
                The model outputs and labels.

        Returns:
            The computed metrics.
        """
        ...


class ExtractLabelsFunction(t.Protocol):
    """A function used to extract the labels from the generated output."""

    def __call__(
        self, input_batch: dict[str, list], model_output: "GenerativeModelOutput"
    ) -> list[str]:
        """Extract the labels from the generated output.

        Args:
            input_batch:
                The input batch.
            model_output:
                The model output.

        Returns:
            The extracted labels.
        """
        ...


def is_list_of_int(x: object) -> t.TypeGuard[list[int]]:
    """Check if an object is a list of integers.

    Args:
        x:
            The object to check.

    Returns:
        Whether the object is a list of integers.
    """
    return isinstance(x, list) and all(isinstance(i, int) for i in x)


def is_list_of_list_of_int(x: object) -> t.TypeGuard[list[list[int]]]:
    """Check if an object is a list of list of integers.

    Args:
        x:
            The object to check.

    Returns:
        Whether the object is a list of list of integers.
    """
    return (
        isinstance(x, list)
        and all(isinstance(i, list) for i in x)
        and all(isinstance(j, int) for i in x for j in i)
    )


def is_list_of_str(x: object) -> t.TypeGuard[list[str]]:
    """Check if an object is a list of integers.

    Args:
        x:
            The object to check.

    Returns:
        Whether the object is a list of strings.
    """
    return isinstance(x, list) and all(isinstance(i, str) for i in x)
