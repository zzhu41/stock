"""Original parameter encoding followed by exactly one V12 boolean switch."""
from v10_deep.schema import (PARAMETERS as FROZEN_PARAMETERS, F,
                            ASSET_ORDER, CASH, GOLD, BENCHMARK,
                            encode as frozen_encode)

PARAMETERS = FROZEN_PARAMETERS + ("locked_panic_exit",)
P = {name: i for i, name in enumerate(PARAMETERS)}


def validate_locked_panic_exit(config):
    value = config.get("locked_panic_exit", 0)
    if value not in (0, 1):
        raise ValueError("locked_panic_exit must be 0 or 1")
    return int(value)


def encode(config, meta):
    flag = validate_locked_panic_exit(config)
    return frozen_encode(config, meta) + [float(flag)]


def native_header():
    lines = ["enum Feature {" + ", ".join("F_"+name.upper()+"="+str(i) for name,i in F.items()) + "};",
             "enum Parameter {" + ", ".join("P_"+name.upper()+"="+str(i) for name,i in P.items()) + "};",
             "constexpr int NF=%d, NP=%d, CASH=%d, GOLD=%d, BENCH=%d;" % (
                 len(F),len(P),ASSET_ORDER.index(CASH),ASSET_ORDER.index(GOLD),ASSET_ORDER.index(BENCHMARK))]
    return "\n".join(lines)+"\n"
