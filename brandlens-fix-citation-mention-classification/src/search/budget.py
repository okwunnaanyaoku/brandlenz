"""Budget management utilities for Tavily searches."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetLimits:
    max_searches: int
    max_cost_usd: float


@dataclass
class BudgetState:
    total_searches: int = 0
    total_cost_usd: float = 0.0

    def record(self, cost_usd: float) -> None:
        self.total_searches += 1
        self.total_cost_usd += cost_usd

    @property
    def average_cost(self) -> float:
        if self.total_searches == 0:
            return 0.0
        return self.total_cost_usd / self.total_searches


class BudgetManager:
    """Track search usage against configured limits."""

    def __init__(self, limits: BudgetLimits) -> None:
        self._limits = limits
        self._state = BudgetState()

    @property
    def state(self) -> BudgetState:
        return self._state

    @property
    def limits(self) -> BudgetLimits:
        return self._limits

    def record_call(self, cost_usd: float) -> None:
        self._state.record(cost_usd)

    def remaining_searches(self) -> int:
        return max(self._limits.max_searches - self._state.total_searches, 0)

    def remaining_budget(self) -> float:
        return max(self._limits.max_cost_usd - self._state.total_cost_usd, 0.0)

    def should_stop(self) -> bool:
        if self._state.total_searches >= self._limits.max_searches:
            return True
        if self._state.total_cost_usd >= self._limits.max_cost_usd:
            return True
        return False
