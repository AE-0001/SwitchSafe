from switchsafe.policy import Decision, RiskSignals, decide, policy_metrics


def test_uncertain_critical_entity_requires_confirmation() -> None:
    decision, _ = decide(RiskSignals(-0.2, 0.01, 0.4, critical_action=True))
    assert decision == Decision.CONFIRM


def test_low_asr_confidence_retries() -> None:
    decision, _ = decide(RiskSignals(-1.1, 0.01))
    assert decision == Decision.RETRY


def test_reliable_transcript_is_accepted() -> None:
    decision, _ = decide(RiskSignals(-0.2, 0.01))
    assert decision == Decision.ACCEPT


def test_silence_escalates() -> None:
    decision, _ = decide(RiskSignals(-2.0, 0.9))
    assert decision == Decision.ESCALATE


def test_policy_metrics_measure_unsafe_acceptance_and_coverage() -> None:
    metrics = policy_metrics([
        {"decision": Decision.ACCEPT, "is_safe": True},
        {"decision": Decision.CONFIRM, "is_safe": False},
        {"decision": Decision.RETRY, "is_safe": False},
    ])
    assert metrics["automated_processing_coverage"] == 1 / 3
    assert metrics["unsafe_transcript_acceptance_rate"] == 0.0
