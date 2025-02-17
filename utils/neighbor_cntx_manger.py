from colorama import Fore, Style
from torch import Tensor


class NeighborContextManager:

    def __init__(self, source: Tensor, target: Tensor, log: bool = True):
        self.source: Tensor = source
        self.target: Tensor = target
        self.log: bool = log

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None:
            if self.log:
                print(
                    "Source node " + Fore.RED + f"{self.source}" + Style.RESET_ALL +
                    " does not have any path connecting with node " + Fore.GREEN +
                    f"{self.target}" + Style.RESET_ALL + ", it will be left to infinity"
                )

            return True
