import csv
import re
from dataclasses import dataclass
from io import StringIO
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class CsvRepairDiagnostic:
    """Summary of in-memory CSV text repair applied before pandas parsing."""

    source: str
    mode: str
    needs_repair: bool
    replacements: int = 0
    malformed_after_repair: bool = False
    row_width_errors: Tuple[Tuple[int, int, int], ...] = ()

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "mode": self.mode,
            "needs_repair": self.needs_repair,
            "replacements": self.replacements,
            "malformed_after_repair": self.malformed_after_repair,
            "row_width_errors": self.row_width_errors,
        }


class CsvRepairError(ValueError):
    """Raised when a CSV repair policy detects malformed input in strict modes."""

    def __init__(self, message: str, diagnostic: CsvRepairDiagnostic) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


class UserLabelCsvRepair:
    """
    In-memory repair policy for legacy CSV files where UserLabel values contain
    commas but were written without quotes.

    Modes:
        off: current behavior, no text repair.
        check: detect known unquoted values and raise without repairing.
        repair: quote known values in memory before pandas reads the CSV.
        strict: repair, validate row widths, and raise if malformed data remains.
    """

    DEFAULT_VALUES = (
        "продувка, цикл 1",
        "продувка, цикл 2",
        "аналит, цикл 1",
        "аналит, цикл 2",
        "воздух, цикл 1",
        "воздух, цикл 2",
    )
    MODES = {"off", "check", "repair", "strict"}

    def __init__(self, mode: str = "repair", values: Optional[Iterable[str]] = None) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown CSV repair mode {mode!r}. Expected one of {sorted(self.MODES)}")

        self.mode = mode
        self.values = tuple(values) if values is not None else self.DEFAULT_VALUES
        self._patterns = tuple(
            re.compile(rf'(?<!"){re.escape(value)}(?!")') for value in self.values
        )

    def prepare(self, text: str, source: str = "") -> Tuple[str, CsvRepairDiagnostic]:
        if self.mode == "off":
            diagnostic = CsvRepairDiagnostic(source=source, mode=self.mode, needs_repair=False)
            return text, diagnostic

        replacements = 0
        repaired_text = text

        for pattern, value in zip(self._patterns, self.values):
            repaired_text, count = pattern.subn(f'"{value}"', repaired_text)
            replacements += count

        needs_repair = replacements > 0
        diagnostic = CsvRepairDiagnostic(
            source=source,
            mode=self.mode,
            needs_repair=needs_repair,
            replacements=replacements if self.mode in {"repair", "strict"} else 0,
        )

        if self.mode == "check" and needs_repair:
            raise CsvRepairError(f"{source} contains unquoted UserLabel values", diagnostic)

        if self.mode == "check":
            return text, diagnostic

        if self.mode == "strict":
            row_width_errors = self._row_width_errors(repaired_text)
            if row_width_errors:
                diagnostic = CsvRepairDiagnostic(
                    source=source,
                    mode=self.mode,
                    needs_repair=needs_repair,
                    replacements=replacements,
                    malformed_after_repair=True,
                    row_width_errors=tuple(row_width_errors),
                )
                raise CsvRepairError(f"{source} is still malformed after UserLabel repair", diagnostic)

        return repaired_text, diagnostic

    @staticmethod
    def _row_width_errors(text: str) -> List[Tuple[int, int, int]]:
        reader = csv.reader(StringIO(text))
        try:
            header = next(reader)
        except StopIteration:
            return []

        expected_width = len(header)
        errors = []
        for row_number, row in enumerate(reader, start=2):
            if len(row) == 0:
                continue
            if len(row) != expected_width:
                errors.append((row_number, expected_width, len(row)))
        return errors
