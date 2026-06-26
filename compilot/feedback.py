"""Format a backend Result into the LLM feedback message (paper's Fig. 6).

Five categories: invalid, illegal (incl. cannot-parallelize), compiler/runtime
error, unsupported, and successful execution (with the measured speedup).
"""

_CONTINUE = ("\nIf a better schedule may exist, propose it. Otherwise output "
             "<schedule>no_further_transformations</schedule>.")


def format_feedback(result, best_so_far=None):
    s = result.status
    if s == "success":
        sp = result.speedup
        verdict = "a speedup" if sp >= 1.0 else "a SLOWDOWN"
        msg = (f"Legal schedule. It yielded {verdict} of {sp:.2f}x ({result.detail}). "
               f"Best so far: {max(sp, best_so_far or sp):.2f}x.")
    elif s == "illegal":
        msg = ("Illegal schedule: it violates data dependencies and was rejected by "
               "the legality checker. " + result.detail)
    elif s == "parallel_illegal":
        msg = "Illegal schedule: " + result.detail + " Parallelize a different loop."
    elif s == "invalid":
        msg = "Invalid schedule (could not be parsed/applied): " + result.detail
    elif s in ("compile_error", "runtime_error"):
        msg = f"{s.replace('_', ' ').title()}: {result.detail}"
    elif s == "unsupported":
        msg = ("Schedule is legal but could not be measured (codegen gap): "
               + result.detail + " Try transformations expressible in C.")
    elif s == "incorrect":
        msg = "Internal error: result mismatch. Try a different schedule."
    else:
        msg = f"Unknown outcome: {s}."
    return msg + _CONTINUE
