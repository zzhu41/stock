"""Only an ideal-close diagnostic; never a candidate-selection clock."""
from v10_round2.timing import run_same_close
from .data import START, END
from .policy import SearchPolicy


def simulate_same_close(config, bank, fear, end=END):
    policy = SearchPolicy(config, bank, fear=fear, trace=True, same_close=True)
    indices = {d: i for i, d in enumerate(bank.calendar)}

    def adapter(date, observed, state):
        holding = state["holding"]
        entry = state["holding_since"].get(holding)
        instruction = policy(indices[date], holding, indices[entry] if entry else None,
                             state["execution_deferred"])
        if instruction is None:
            return None
        return {instruction: 1.0} if instruction else {}

    result = run_same_close(bank.histories, bank.calendar, adapter, START, end, fee=.0001, slippage=0)
    result["policy_metadata"] = policy.metadata
    return result
