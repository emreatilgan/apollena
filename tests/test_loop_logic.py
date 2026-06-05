"""Unit tests for the orchestrator's outer-loop decision logic.

These cover the PURE functions only — no SDK calls, no network, no agent spend.
Run: ./venv/bin/python -m pytest tests/test_loop_logic.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator import (  # noqa: E402
    should_stop,
    next_action,
    update_best,
    update_stall,
    build_attempt_history,
    hypothesis_is_distinct,
)


def make_config(**overrides) -> dict:
    cfg = {
        "max_iterations": 3,
        "target_auc": 0.98,
        "patience": 2,
        "rescout_after": 1,
        "max_budget_usd": None,
    }
    cfg.update(overrides)
    return cfg


def make_sessions(**overrides) -> dict:
    s = {
        "run_id": "test1234",
        "domain": "statistical_physics",
        "total_cost_usd": 0.0,
        "scout": {"status": "completed", "runs": 1},
        "iterations": [],
        "best": None,
        "stall": 0,
        "rescout_next": False,
        "rescouted_this_stall": False,
        "stop_reason": None,
    }
    s.update(overrides)
    return s


def _completed_iter(index, auc, model_name="ModelA"):
    return {
        "index": index,
        "rescouted": False,
        "stages": {
            "theorist": {"status": "completed", "output": f"runs/r/iter_{index:03d}/hypothesis.json"},
            "engineer": {"status": "completed", "output": {"model_file": "m.py"}},
            "benchmarker": {"status": "completed", "output": f"runs/r/iter_{index:03d}/benchmark_result.json"},
        },
        "result": {"model_name": model_name, "auc": auc, "logloss": 0.2, "dataset": "Frappe_x1", "status": "completed"},
        "attempt": {"iteration": index, "model_name": model_name, "auc": auc, "status": "completed"},
    }


# --- should_stop ---

def test_should_stop_none_when_under_limits():
    assert should_stop(make_sessions(), make_config()) is None


def test_should_stop_target_auc():
    s = make_sessions(best={"auc": 0.985})
    assert should_stop(s, make_config()) == "target_auc"


def test_should_stop_max_iterations():
    s = make_sessions(iterations=[_completed_iter(0, 0.9), _completed_iter(1, 0.91), _completed_iter(2, 0.92)])
    assert should_stop(s, make_config(max_iterations=3)) == "max_iterations"


def test_should_stop_patience():
    s = make_sessions(stall=2)
    assert should_stop(s, make_config(patience=2)) == "patience"


def test_should_stop_budget():
    s = make_sessions(total_cost_usd=21.0)
    assert should_stop(s, make_config(max_budget_usd=20.0)) == "budget"


def test_should_stop_target_beats_max_iterations():
    # success short-circuits even at the iteration cap
    s = make_sessions(
        best={"auc": 0.99},
        iterations=[_completed_iter(0, 0.99), _completed_iter(1, 0.9), _completed_iter(2, 0.9)],
    )
    assert should_stop(s, make_config(max_iterations=3)) == "target_auc"


# --- next_action ---

def test_next_action_runs_scout_first():
    s = make_sessions(scout={"status": "pending", "runs": 0})
    assert next_action(s, make_config()) == ("scout", None)


def test_next_action_starts_first_iteration():
    assert next_action(make_sessions(), make_config()) == ("start_iter", 0)


def test_next_action_resumes_in_progress_iteration():
    in_progress = _completed_iter(0, 0.9)
    in_progress["result"] = None  # interrupted before finalize
    s = make_sessions(iterations=[in_progress])
    assert next_action(s, make_config()) == ("resume_iter", 0)


def test_next_action_starts_next_index_after_finished():
    s = make_sessions(iterations=[_completed_iter(0, 0.9)])
    assert next_action(s, make_config()) == ("start_iter", 1)


def test_next_action_stops_when_should_stop():
    s = make_sessions(stall=2)
    assert next_action(s, make_config(patience=2)) == ("stop", "patience")


# --- update_best ---

def test_update_best_sets_first_result():
    s = make_sessions()
    improved = update_best(s, _completed_iter(0, 0.95))
    assert improved is True
    assert s["best"]["auc"] == 0.95
    assert s["best"]["iteration"] == 0


def test_update_best_improves():
    s = make_sessions(best={"auc": 0.95})
    assert update_best(s, _completed_iter(1, 0.97)) is True
    assert s["best"]["auc"] == 0.97


def test_update_best_no_regression():
    s = make_sessions(best={"auc": 0.97, "iteration": 0})
    assert update_best(s, _completed_iter(1, 0.90)) is False
    assert s["best"]["auc"] == 0.97


def test_update_best_ignores_failed_iteration():
    s = make_sessions(best={"auc": 0.97})
    failed = {"index": 1, "result": {"auc": None, "status": "failed"}, "stages": {}}
    assert update_best(s, failed) is False
    assert s["best"]["auc"] == 0.97


# --- update_stall ---

def test_update_stall_resets_on_improvement():
    s = make_sessions(stall=1, rescouted_this_stall=True)
    update_stall(s, make_config(), improved=True)
    assert s["stall"] == 0
    assert s["rescouted_this_stall"] is False


def test_update_stall_arms_rescout_when_stuck():
    s = make_sessions(stall=0)
    update_stall(s, make_config(rescout_after=1), improved=False)
    assert s["stall"] == 1
    assert s["rescout_next"] is True
    assert s["rescouted_this_stall"] is True


def test_update_stall_does_not_rearm_rescout():
    s = make_sessions(stall=1, rescouted_this_stall=True, rescout_next=False)
    update_stall(s, make_config(rescout_after=1), improved=False)
    assert s["stall"] == 2
    assert s["rescout_next"] is False  # already re-scouted this stall


# --- build_attempt_history ---

def test_build_attempt_history_skips_missing():
    s = make_sessions(iterations=[
        _completed_iter(0, 0.95),
        {"index": 1, "attempt": None},
        _completed_iter(2, 0.96),
    ])
    history = build_attempt_history(s)
    assert [a["iteration"] for a in history] == [0, 2]


# --- hypothesis_is_distinct ---

def test_hypothesis_distinct_true_for_new_name():
    prior = [{"model_name": "TAPNet"}]
    assert hypothesis_is_distinct({"model_name": "SpinGlassNet"}, prior) is True


def test_hypothesis_distinct_false_for_repeat():
    prior = [{"model_name": "TAPNet"}]
    # parenthetical + case difference must still be caught
    assert hypothesis_is_distinct({"model_name": "tapnet (Thouless-Anderson-Palmer)"}, prior) is False


def test_hypothesis_distinct_true_for_empty_name():
    # empty name is deferred to validate_theorist_output, not blocked here
    assert hypothesis_is_distinct({"model_name": ""}, [{"model_name": "TAPNet"}]) is True


# --- Loop convergence simulation ---
# Drives the REAL decision functions (next_action / should_stop / update_best /
# update_stall) through a scripted sequence of iteration AUCs, mirroring the
# bookkeeping run_pipeline does around the (here stubbed-out) agent stages. This
# verifies the assembled loop terminates and reports the right stop_reason — the
# single most important behavioral property of the outer loop — with zero spend.

def _simulate(aucs, config):
    """Return (stop_reason, num_iterations, rescout_indices) for scripted AUCs.

    `aucs[i]` is the AUC iteration i yields, or None for a failed iteration.
    """
    s = make_sessions(scout={"status": "pending", "runs": 0})
    rescout_indices = []
    guard = 0
    while True:
        guard += 1
        assert guard < 100, "loop failed to converge"

        action, arg = next_action(s, config)
        if action == "stop":
            return arg, len(s["iterations"]), rescout_indices
        if action == "scout":
            s["scout"]["status"] = "completed"
            s["scout"]["runs"] = s["scout"].get("runs", 0) + 1
            continue

        idx = arg
        if action == "start_iter":
            did_rescout = bool(s.get("rescout_next"))
            if did_rescout:
                s["rescout_next"] = False
                rescout_indices.append(idx)
            s["iterations"].append({"index": idx, "rescouted": did_rescout,
                                    "stages": {"theorist": {"output": None},
                                               "benchmarker": {"output": None}},
                                    "result": None, "attempt": None})
        it = s["iterations"][idx]

        auc = aucs[idx] if idx < len(aucs) else None
        completed = auc is not None
        it["result"] = {"model_name": f"M{idx}", "auc": auc, "logloss": 0.2,
                        "dataset": "Frappe_x1", "status": "completed" if completed else "failed"}
        it["attempt"] = {"iteration": idx, "model_name": f"M{idx}", "auc": auc,
                         "status": "completed" if completed else "failed"}

        improved = update_best(s, it) if completed else False
        update_stall(s, config, improved)


def test_sim_stops_at_max_iterations():
    reason, n, rescouts = _simulate([0.90, 0.92, 0.94], make_config())
    assert reason == "max_iterations"
    assert n == 3
    assert rescouts == []  # steady improvement never stalls


def test_sim_stops_early_on_target():
    reason, n, rescouts = _simulate([0.90, 0.99, 0.99], make_config())
    assert reason == "target_auc"
    assert n == 2  # stops as soon as the target is hit


def test_sim_stall_triggers_rescout_then_patience():
    reason, n, rescouts = _simulate([0.90, 0.90, 0.90], make_config(max_iterations=5, patience=2, rescout_after=1))
    assert reason == "patience"
    assert n == 3
    assert rescouts == [2]  # re-scout fired once to break the stall


def test_sim_failed_iterations_converge():
    reason, n, rescouts = _simulate([None, None, None], make_config(max_iterations=5, patience=2, rescout_after=1))
    assert reason == "patience"
    assert n == 2  # two failures with no AUC → patience stop
    assert rescouts == [1]
