import json
from pathlib import Path
from typing import Any

from app.environment.graders import grade_task
from app.environment.scenarios.accident import generate_accident_case
from app.environment.scenarios.fire import generate_fire_case
from app.environment.scenarios.medical import generate_medical_case
from app.environment.validation import DifficultyModifier, HospitalValidator
from app.models.action import Action
from app.models.observation import ArrivalOutcomeObservation, HospitalObservation, Observation
from app.models.reward import RewardBreakdown, RewardModel, StepInfo
from app.models.state import ArrivalOutcome, EnvState, HospitalState, HospitalValidationDetails, LearningEntry
from app.utils.calculations import compute_speed_kmh, compute_travel_time_minutes
from app.utils.randomizer import SeededRandomizer


TASKS = {
    "acde_easy": {
        "difficulty": "easy",
        "objective": "Stabilize quickly while information is mostly reliable.",
    },
    "acde_medium": {
        "difficulty": "medium",
        "objective": "Balance speed, uncertainty, and specialization constraints.",
    },
    "acde_hard": {
        "difficulty": "hard",
        "objective": "Make least-bad decisions when every hospital has trade-offs.",
    },
}

MIN_REWARD = 0.001
MAX_REWARD = 0.999

OUTCOME_SCORE = {"accepted": 3, "partial": 2, "rejected": 1}
CONDITION_ORDER = ["stable", "serious", "unstable", "critical"]


class EmergencyEnv:
    """Stateful local RL environment for emergency routing under uncertainty."""

    def __init__(self, memory_file: str):
        self.memory_path = Path(memory_file)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists():
            self.memory_path.write_text("{}", encoding="utf-8")

        self.episode_counter = 0
        self._rng = SeededRandomizer(seed=42)
        self.state_data: EnvState | None = None
        self.validator = HospitalValidator(self._rng)
        self.trajectory: list[dict[str, Any]] = []
        self.last_info: StepInfo | None = None
        self.last_outcome_status: str | None = None
        self.base_speed_kmh = 60.0

    def reset(self, seed: int | None = None, task_id: str | None = None) -> Observation:
        if seed is None:
            seed = self._rng.randint(1, 10**9)

        resolved_task_id = task_id if task_id in TASKS else self._rng.choice(list(TASKS.keys()))
        difficulty = TASKS[resolved_task_id]["difficulty"]

        self._rng = SeededRandomizer(seed)
        self.validator = HospitalValidator(self._rng)
        self.episode_counter += 1
        self.trajectory = []
        self.last_outcome_status = None

        scenario, scenario_type = self._sample_scenario_for_difficulty(difficulty)
        hospitals = self._build_hospital_states(scenario)
        hospitals = self._augment_hospital_options(
            hospitals,
            difficulty,
            required_specialization=scenario["required_specialization"],
        )
        hospitals = self._inject_no_perfect_option(hospitals, difficulty)

        max_steps = {"easy": 3, "medium": 4, "hard": 4}.get(difficulty, 4)

        self.state_data = EnvState(
            episode_id=self.episode_counter,
            seed=seed,
            task_id=resolved_task_id,
            task_objective=TASKS[resolved_task_id]["objective"],
            scenario_type=scenario_type,
            scenario_name=scenario["scenario_name"],
            scenario_difficulty=difficulty,
            patient_condition=scenario["patient_condition"],
            required_specialization=scenario["required_specialization"],
            initial_critical_time_limit_minutes=scenario["critical_time_limit_minutes"],
            critical_time_limit_minutes=scenario["critical_time_limit_minutes"],
            step=1,
            max_steps=max_steps,
            hospitals=hospitals,
            selected_hospital_id=None,
            done=False,
            final_outcome=None,
            reward=MIN_REWARD,
            final_score=MIN_REWARD,
            ambulance_status="en_route",
            current_location_context="incident_site",
            visited_hospitals=[],
            failed_hospitals=[],
            recent_failed_hospitals=[],
            failed_reasons={},
            total_time_spent_minutes=0.0,
            rerouting_reason=None,
            last_arrival_outcome=None,
            accepted_hospital_id=None,
            explanation=[
                "Episode initialized with seeded uncertainty.",
                f"Difficulty: {difficulty}. Hidden hospital state can change during transit.",
                f"Patient condition: {scenario['patient_condition']}.",
                f"Required specialization: {scenario['required_specialization']}.",
                "Primary objective: admit patient successfully under uncertainty.",
            ],
            memory=self._load_memory(),
        )

        self.last_info = StepInfo(
            task_id=resolved_task_id,
            difficulty=difficulty,
            objective=TASKS[resolved_task_id]["objective"],
            progress_score=MIN_REWARD,
            reward_model=RewardModel(
                value=MIN_REWARD,
                breakdown=RewardBreakdown(
                    survival_component=MIN_REWARD,
                    time_efficiency_component=MIN_REWARD,
                    specialization_component=MIN_REWARD,
                    delay_penalty=MIN_REWARD,
                ),
            ),
            grader=None,
            last_action_error=None,
            outcome=None,
        )

        return self._build_observation()

    def state(self) -> EnvState:
        if self.state_data is None:
            self.reset(seed=42, task_id="acde_medium")
        assert self.state_data is not None
        return self.state_data

    def step(self, action: Action | str | dict[str, Any]) -> dict[str, Any]:
        if self.state_data is None:
            self.reset(seed=42, task_id="acde_medium")
        assert self.state_data is not None

        if self.state_data.done:
            info = self.last_info.model_dump() if self.last_info else {}
            return {
                "observation": self._build_observation(),
                "reward": MIN_REWARD,
                "done": True,
                "info": info,
            }

        normalized_action = self._normalize_action(action)
        if normalized_action.step != self.state_data.step:
            raise ValueError(
                f"Action step {normalized_action.step} does not match environment step {self.state_data.step}."
            )

        selected = self._find_hospital(normalized_action.hospital_id)
        if selected is None:
            raise ValueError(f"Unknown hospital id: {normalized_action.hospital_id}")

        was_visited_before = selected.hospital_id in self.state_data.visited_hospitals
        was_failed_before = selected.hospital_id in self.state_data.failed_hospitals

        original_traffic = selected.traffic
        selected.traffic = self._traffic_shift(selected.traffic, self.state_data.scenario_difficulty)

        speed = compute_speed_kmh(self.base_speed_kmh, selected.traffic)
        travel_time = compute_travel_time_minutes(selected.distance_km, speed)

        delay_probability = {
            "easy": 0.10,
            "medium": 0.25,
            "hard": 0.45,
        }.get(self.state_data.scenario_difficulty, 0.25)
        dynamic_delay = self._rng.uniform(0.5, 2.5) if self._rng.random() < delay_probability else 0.0
        travel_time += dynamic_delay

        selected, travel_time, enroute_note = self._apply_enroute_diversion(selected, travel_time)

        self.state_data.total_time_spent_minutes += travel_time

        if selected.hospital_id not in self.state_data.visited_hospitals:
            self.state_data.visited_hospitals.append(selected.hospital_id)

        self.state_data.ambulance_status = "arrived"
        self.state_data.current_location_context = f"arrived_at_{selected.hospital_id}"

        arrival_outcome = self.validator.validate_arrival(
            hospital=selected,
            difficulty=self.state_data.scenario_difficulty,
            patient_condition=self.state_data.patient_condition,
            required_specialization=self.state_data.required_specialization,
            total_time_spent=self.state_data.total_time_spent_minutes,
            critical_time_limit=self.state_data.critical_time_limit_minutes,
            step_number=self.state_data.step,
        )

        # Hidden-case guess: selecting uncertain ICU may lead to wrong guess at arrival.
        arrival_outcome, hidden_case_penalty, hidden_case_note = self._apply_hidden_guess_case(
            selected,
            arrival_outcome,
        )

        # Late-arrival shocks: on arrival, resources may suddenly become unavailable.
        arrival_outcome, shock_penalty, shock_note = self._apply_arrival_hidden_shock(
            arrival_outcome,
            difficulty=self.state_data.scenario_difficulty,
        )

        # Fix 1: cap partial chains so they resolve after repeated delays.
        arrival_outcome, partial_cap_note = self._apply_partial_chain_cap(arrival_outcome)

        # Critical polish: early hard rejections can degrade to partial to preserve recoverability.
        arrival_outcome, early_reject_note = self._apply_early_reject_protection(arrival_outcome)

        # Critical polish: partial outcomes after step 2 can recover into acceptance.
        arrival_outcome, late_partial_note = self._apply_late_partial_recovery(arrival_outcome)

        # Fix 3: final-attempt pressure can produce emergency stabilization.
        arrival_outcome, last_chance_note = self._apply_last_chance_outcome(arrival_outcome)

        reward, breakdown = self._calculate_reward(
            selected=selected,
            arrival_outcome=arrival_outcome,
            travel_time=travel_time,
            was_visited_before=was_visited_before,
            was_failed_before=was_failed_before,
            hidden_case_penalty=hidden_case_penalty + shock_penalty,
        )

        success = arrival_outcome.status in {"accepted", "partial"}
        self._update_learning_memory(selected.hospital_id, success, reward)
        self.state_data.memory = self._load_memory()

        self._record_trajectory(
            selected=selected,
            arrival_outcome=arrival_outcome,
            reward=reward,
            travel_time=travel_time,
            dynamic_delay=dynamic_delay,
            original_traffic=original_traffic,
        )

        self.state_data.selected_hospital_id = selected.hospital_id
        self.state_data.reward = reward
        self.state_data.last_arrival_outcome = arrival_outcome

        self._advance_patient_state(arrival_outcome.status, travel_time, dynamic_delay)

        self._resolve_transition(selected, arrival_outcome)

        self._build_last_info(reward, breakdown, arrival_outcome)

        if not self.state_data.done:
            self._evolve_hospital_uncertainty()

        self._set_explanation(
            selected,
            arrival_outcome,
            travel_time,
            dynamic_delay,
            original_traffic,
            [
                note
                for note in [
                    enroute_note,
                    hidden_case_note,
                    shock_note,
                    partial_cap_note,
                    early_reject_note,
                    late_partial_note,
                    last_chance_note,
                ]
                if note
            ],
        )

        info = self.last_info.model_dump() if self.last_info else {}
        return {
            "observation": self._build_observation(),
            "reward": reward,
            "done": self.state_data.done,
            "info": info,
        }

    def _normalize_action(self, action: Action | str | dict[str, Any]) -> Action:
        if isinstance(action, Action):
            return action
        if isinstance(action, str):
            assert self.state_data is not None
            return Action(step=self.state_data.step, hospital_id=action, rationale="policy selection")
        if isinstance(action, dict):
            assert self.state_data is not None
            payload = {
                "step": action.get("step", self.state_data.step),
                "hospital_id": action.get("hospital_id"),
                "rationale": action.get("rationale"),
            }
            return Action(**payload)
        raise ValueError("Action must be Action, hospital_id string, or action dict.")

    def _build_hospital_states(self, scenario: dict[str, Any]) -> list[HospitalState]:
        hospitals: list[HospitalState] = []
        for template in scenario["hospitals"]:
            distance = round(
                self._rng.uniform(template["distance_range"][0], template["distance_range"][1]),
                1,
            )
            traffic = self._rng.choice(template["traffic_options"])
            icu_actual = self._rng.random() < template["icu_true_probability"]

            if icu_actual:
                icu_display = "available" if self._rng.random() < 0.85 else "unknown"
            else:
                icu_display = "available" if self._rng.random() < 0.2 else "unknown"

            hospitals.append(
                HospitalState(
                    hospital_id=template["hospital_id"],
                    distance_km=distance,
                    icu_display=icu_display,
                    icu_actual=icu_actual,
                    specialization=template["specialization"],
                    traffic=traffic,
                )
            )
        return hospitals

    def _inject_no_perfect_option(self, hospitals: list[HospitalState], difficulty: str) -> list[HospitalState]:
        trigger = {"easy": 0.06, "medium": 0.30, "hard": 0.42}.get(difficulty, 0.30)
        if self._rng.random() >= trigger:
            return hospitals

        if len(hospitals) < 3:
            return hospitals

        hospitals[0].traffic = "high"
        hospitals[1].icu_display = "unknown"
        hospitals[2].specialization = "general" if hospitals[2].specialization != "general" else "trauma"
        hospitals[2].icu_display = "unknown"
        return hospitals

    def _augment_hospital_options(
        self,
        hospitals: list[HospitalState],
        difficulty: str,
        required_specialization: str,
    ) -> list[HospitalState]:
        """Add extra decoy/alternative hospitals to increase decision ambiguity."""
        target_extra = {"easy": 1, "medium": 1, "hard": 2}.get(difficulty, 1)
        extra_count = 0
        while extra_count < target_extra:
            new_id = f"H{len(hospitals) + 1}"
            # Keep options plausible but uncertain: mixed specialization and variable traffic.
            spec_roll = self._rng.random()
            if spec_roll < 0.45:
                specialization = required_specialization
            elif spec_roll < 0.75:
                specialization = "general"
            else:
                specialization = "trauma" if required_specialization != "trauma" else "cardiac"

            distance = round(self._rng.uniform(4.0, 13.5), 1)
            traffic = self._rng.choice(["low", "medium", "high"])

            icu_prob = {"easy": 0.62, "medium": 0.52, "hard": 0.42}.get(difficulty, 0.52)
            icu_actual = self._rng.random() < icu_prob
            if icu_actual:
                icu_display = "available" if self._rng.random() < 0.74 else "unknown"
            else:
                icu_display = "available" if self._rng.random() < 0.18 else "unknown"

            hospitals.append(
                HospitalState(
                    hospital_id=new_id,
                    distance_km=distance,
                    icu_display=icu_display,
                    icu_actual=icu_actual,
                    specialization=specialization,
                    traffic=traffic,
                )
            )
            extra_count += 1
        return hospitals

    def _calculate_reward(
        self,
        selected: HospitalState,
        arrival_outcome: ArrivalOutcome,
        travel_time: float,
        was_visited_before: bool,
        was_failed_before: bool,
        hidden_case_penalty: float,
    ) -> tuple[float, RewardBreakdown]:
        assert self.state_data is not None

        base_status_reward = {
            "accepted": 0.92,
            "partial": 0.55,
            "rejected": 0.08,
        }[arrival_outcome.status]

        if arrival_outcome.status == "rejected":
            status_reward = base_status_reward
        else:
            outcome_modifier = max(0.5, min(1.2, float(arrival_outcome.reward_modifier)))
            status_reward = base_status_reward * outcome_modifier

        critical_patient = self.state_data.patient_condition in {"critical", "unstable"}
        unknown_critical_penalty = (
            0.12
            if critical_patient and selected.icu_display == "unknown"
            else 0.0
        )
        repeat_penalty = 0.15 if was_visited_before else 0.0
        failed_repeat_penalty = 0.20 if was_failed_before else 0.0
        traffic_penalty = 0.10 if critical_patient and selected.traffic == "high" else 0.04 if critical_patient and selected.traffic == "medium" else 0.0

        time_bonus = 0.06 if travel_time <= 8.0 else (0.03 if travel_time <= 14.0 else 0.0)

        improvement_bonus = self._improvement_bonus(arrival_outcome.status)

        reward = (
            status_reward
            + time_bonus
            + improvement_bonus
            - unknown_critical_penalty
            - repeat_penalty
            - failed_repeat_penalty
            - traffic_penalty
            - hidden_case_penalty
        )
        reward = max(MIN_REWARD, min(MAX_REWARD, reward))

        breakdown = RewardBreakdown(
            survival_component=max(MIN_REWARD, min(MAX_REWARD, (status_reward + 0.5) / 1.5)),
            time_efficiency_component=max(MIN_REWARD, min(MAX_REWARD, 1.0 - (travel_time / 25.0))),
            specialization_component=(MAX_REWARD if self._specialization_match(selected) else 0.4),
            delay_penalty=min(
                MAX_REWARD,
                unknown_critical_penalty
                + repeat_penalty
                + failed_repeat_penalty
                + traffic_penalty
                + hidden_case_penalty,
            ),
        )

        return reward, breakdown

    def _improvement_bonus(self, status: str) -> float:
        if self.last_outcome_status is None:
            self.last_outcome_status = status
            return MIN_REWARD

        delta = OUTCOME_SCORE[status] - OUTCOME_SCORE[self.last_outcome_status]
        self.last_outcome_status = status
        if delta > 0:
            return 0.04
        return MIN_REWARD

    def _specialization_match(self, hospital: HospitalState) -> bool:
        assert self.state_data is not None
        return (
            hospital.specialization == self.state_data.required_specialization
            or hospital.specialization == "general"
            or self.state_data.required_specialization == "general"
        )

    def _advance_patient_state(self, outcome_status: str, travel_time: float, dynamic_delay: float) -> None:
        assert self.state_data is not None

        condition = self.state_data.patient_condition
        idx = CONDITION_ORDER.index(condition) if condition in CONDITION_ORDER else 2

        deterioration_risk = 0.0
        if travel_time > 12.0:
            deterioration_risk += 0.20
        if dynamic_delay > 0:
            deterioration_risk += 0.15
        if outcome_status == "rejected":
            deterioration_risk += 0.20

        if self._rng.random() < min(0.95, deterioration_risk):
            idx = min(len(CONDITION_ORDER) - 1, idx + 1)

        if outcome_status == "partial":
            stabilize_prob = {"easy": 0.35, "medium": 0.22, "hard": 0.12}.get(
                self.state_data.scenario_difficulty,
                0.22,
            )
            if self._rng.random() < stabilize_prob:
                idx = max(0, idx - 1)

        self.state_data.patient_condition = CONDITION_ORDER[idx]

    def _resolve_transition(self, selected: HospitalState, arrival_outcome: ArrivalOutcome) -> None:
        assert self.state_data is not None

        if arrival_outcome.status == "accepted":
            self.state_data.accepted_hospital_id = selected.hospital_id
            self.state_data.ambulance_status = "admitted"
            self.state_data.current_location_context = selected.hospital_id
            self.state_data.done = True
            self.state_data.final_outcome = "SUCCESS"
            self.state_data.final_score = self._success_score()
            return

        if arrival_outcome.status == "rejected":
            if selected.hospital_id not in self.state_data.failed_hospitals:
                self.state_data.failed_hospitals.append(selected.hospital_id)

            # Cooldown memory: block immediate retries, but allow later reconsideration.
            self.state_data.recent_failed_hospitals.append(selected.hospital_id)
            if len(self.state_data.recent_failed_hospitals) > 3:
                self.state_data.recent_failed_hospitals.pop(0)

            self.state_data.failed_reasons[selected.hospital_id] = arrival_outcome.reason

            if arrival_outcome.terminal:
                self.state_data.done = True
                self.state_data.final_outcome = "FAILURE"
                self.state_data.final_score = self._failure_score()
                self.state_data.rerouting_reason = arrival_outcome.reason
                self.state_data.ambulance_status = "arrived"
                self.state_data.current_location_context = f"terminal_failure_at_{selected.hospital_id}"
                return

            self.state_data.rerouting_reason = arrival_outcome.reason
            self.state_data.ambulance_status = "rerouting"
            self.state_data.current_location_context = f"rejected_at_{selected.hospital_id}"
        else:
            self.state_data.ambulance_status = "in_transit"
            self.state_data.current_location_context = "post_partial_treatment"

        if self._critical_failure():
            self.state_data.done = True
            self.state_data.final_outcome = "FAILURE"
            self.state_data.final_score = self._failure_score()
            return

        if self.state_data.step >= self.state_data.max_steps:
            self.state_data.done = True
            self.state_data.final_outcome = "FAILURE"
            self.state_data.final_score = self._failure_score()
            return

        self.state_data.step += 1
        self.state_data.done = False
        self.state_data.final_outcome = None

    def _critical_failure(self) -> bool:
        # Time-window based failure is disabled. Episodes end by acceptance or max steps.
        return False

    def _set_explanation(
        self,
        selected: HospitalState,
        arrival_outcome: ArrivalOutcome,
        travel_time: float,
        dynamic_delay: float,
        original_traffic: str,
        hidden_case_notes: list[str],
    ) -> None:
        assert self.state_data is not None
        v = arrival_outcome.validation_details
        assert v is not None
        self.state_data.explanation = [
            f"Step {self.state_data.step}: selected {selected.hospital_id}.",
            f"Traffic changed {original_traffic} -> {selected.traffic} before arrival.",
            f"Travel time: {travel_time:.2f} min (delay {dynamic_delay:.2f} min).",
            f"Validation checks: ICU={v.icu_available}, doctor={v.doctor_available}, equipment={v.equipment_functional}, overload={v.overload_status}",
            f"Patient suitability score = {v.patient_suitability:.2f}",
            f"Arrival outcome = {arrival_outcome.status.upper()}",
            f"Arrival reason = {arrival_outcome.reason}",
            f"Patient condition now = {self.state_data.patient_condition}",
            f"Total time spent = {self.state_data.total_time_spent_minutes:.2f} min",
        ]
        for note in hidden_case_notes:
            self.state_data.explanation.append(note)

    def _apply_enroute_diversion(
        self,
        selected: HospitalState,
        travel_time: float,
    ) -> tuple[HospitalState, float, str | None]:
        """Sometimes traffic collapses mid-route and ambulance diverts before arrival."""
        assert self.state_data is not None

        base_diversion_prob = {
            "easy": 0.04,
            "medium": 0.12,
            "hard": 0.18,
        }.get(self.state_data.scenario_difficulty, 0.20)

        if selected.traffic == "high":
            base_diversion_prob += 0.08
        elif selected.traffic == "medium":
            base_diversion_prob += 0.04

        if self._rng.random() >= min(0.85, base_diversion_prob):
            return selected, travel_time, None

        alternatives = [
            h
            for h in self.state_data.hospitals
            if h.hospital_id != selected.hospital_id and h.hospital_id not in self.state_data.failed_hospitals
        ]
        if not alternatives:
            return selected, travel_time, None

        def _rank(h: HospitalState) -> tuple[int, float]:
            traffic_rank = {"low": 0, "medium": 1, "high": 2}.get(h.traffic, 1)
            return (traffic_rank, h.distance_km)

        diverted = sorted(alternatives, key=_rank)[0]
        diverted_speed = compute_speed_kmh(self.base_speed_kmh, diverted.traffic)
        diverted_time = compute_travel_time_minutes(diverted.distance_km, diverted_speed)
        diversion_overhead = {
            "easy": self._rng.uniform(0.4, 1.1),
            "medium": self._rng.uniform(0.8, 1.8),
            "hard": self._rng.uniform(1.2, 2.6),
        }.get(self.state_data.scenario_difficulty, self._rng.uniform(1.0, 2.2))

        note = (
            f"Hidden case: severe traffic lock en-route to {selected.hospital_id}; "
            f"ambulance diverted to {diverted.hospital_id}."
        )
        return diverted, diverted_time + diversion_overhead, note

    def _apply_hidden_guess_case(
        self,
        selected: HospitalState,
        arrival_outcome: ArrivalOutcome,
    ) -> tuple[ArrivalOutcome, float, str | None]:
        """Resolve hidden guess cases for uncertain hospitals.

        If ICU is shown as unknown, the agent is effectively guessing.
        Wrong guess triggers stronger penalty and forced reroute.
        """
        assert self.state_data is not None

        if selected.icu_display != "unknown":
            return arrival_outcome, MIN_REWARD, None

        difficulty = self.state_data.scenario_difficulty
        guess_success_prob = {
            "easy": 0.82,
            "medium": 0.72,
            "hard": 0.58,
        }.get(difficulty, 0.52)
        guess_correct = self._rng.random() < guess_success_prob

        if guess_correct:
            return (
                arrival_outcome,
                MIN_REWARD,
                "Hidden case: risky ICU-unknown guess was correct this time.",
            )

        # Wrong hidden guess: downgrade to rejected and enforce rerouting signal.
        forced_reject = ArrivalOutcome(
            status="rejected",
            reason="Hidden mismatch at arrival (wrong risky guess). Rerouting required.",
            validation_details=arrival_outcome.validation_details,
            reward_modifier=0.0,
        )
        return (
            forced_reject,
            0.14,
            "Hidden case: risky ICU-unknown guess failed; penalty applied.",
        )

    def _apply_arrival_hidden_shock(
        self,
        arrival_outcome: ArrivalOutcome,
        difficulty: str,
    ) -> tuple[ArrivalOutcome, float, str | None]:
        """Late-arrival operational shocks: ICU/doctor/bed/equipment can fail at handover."""
        if arrival_outcome.status == "rejected":
            return arrival_outcome, MIN_REWARD, None

        shock_prob = {
            "easy": 0.03,
            "medium": 0.05,
            "hard": 0.10,
        }.get(difficulty, 0.14)
        if self._rng.random() >= shock_prob:
            return arrival_outcome, MIN_REWARD, None

        v = arrival_outcome.validation_details
        if v is None:
            return arrival_outcome, MIN_REWARD, None

        shock = self._rng.choice([
            "doctor_unavailable",
            "icu_full",
            "beds_full",
            "machine_failed",
        ])

        if shock == "doctor_unavailable":
            reason = "Doctor was reassigned to another emergency at arrival"
            new_validation = HospitalValidationDetails(
                icu_available=v.icu_available,
                doctor_available=False,
                equipment_functional=v.equipment_functional,
                overload_status=v.overload_status,
                patient_suitability=v.patient_suitability,
            )
        elif shock == "icu_full":
            reason = "ICU got full moments before handover"
            new_validation = HospitalValidationDetails(
                icu_available=False,
                doctor_available=v.doctor_available,
                equipment_functional=v.equipment_functional,
                overload_status=v.overload_status,
                patient_suitability=v.patient_suitability,
            )
        elif shock == "beds_full":
            reason = "Emergency beds became unavailable during arrival"
            new_validation = HospitalValidationDetails(
                icu_available=v.icu_available,
                doctor_available=v.doctor_available,
                equipment_functional=v.equipment_functional,
                overload_status="severe",
                patient_suitability=v.patient_suitability,
            )
        else:
            reason = "Critical treatment machine failed at admission"
            new_validation = HospitalValidationDetails(
                icu_available=v.icu_available,
                doctor_available=v.doctor_available,
                equipment_functional=False,
                overload_status=v.overload_status,
                patient_suitability=v.patient_suitability,
            )

        return (
            ArrivalOutcome(
                status="rejected",
                reason=reason,
                validation_details=new_validation,
                reward_modifier=0.0,
            ),
            0.12,
            f"Hidden case: {reason}. Rerouting required.",
        )

    def _apply_partial_chain_cap(self, arrival_outcome: ArrivalOutcome) -> tuple[ArrivalOutcome, str | None]:
        """Fix 1: after repeated partials, force resolution to accepted or rejected."""
        assert self.state_data is not None
        if arrival_outcome.status != "partial":
            return arrival_outcome, None

        prior_partials = sum(1 for t in self.trajectory if t.get("outcome_status") == "partial")
        partial_count = prior_partials + 1
        if partial_count < 2:
            return arrival_outcome, None

        stabilize_chance = {
            "easy": 0.45,
            "medium": 0.28,
            "hard": 0.05,
        }.get(self.state_data.scenario_difficulty, 0.28)

        if self._rng.random() < stabilize_chance:
            return (
                ArrivalOutcome(
                    status="accepted",
                    reason="Patient stabilized after critical delay",
                    validation_details=arrival_outcome.validation_details,
                    reward_modifier=0.78 if self.state_data.scenario_difficulty == "easy" else 0.68,
                ),
                "Partial chain cap: resolved as emergency stabilization.",
            )

        carry_partial_chance = 0.3 if self.state_data.scenario_difficulty != "hard" else 0.15
        if self._rng.random() < carry_partial_chance:
            return (
                ArrivalOutcome(
                    status="partial",
                    reason="Condition worsened but remains temporarily transferable",
                    validation_details=arrival_outcome.validation_details,
                    reward_modifier=0.44,
                ),
                "Partial chain cap: temporary recovery preserved rerouting chance.",
            )

        return (
            ArrivalOutcome(
                status="rejected",
                reason="Condition became irreversible after delays",
                validation_details=arrival_outcome.validation_details,
                reward_modifier=0.0,
            ),
            "Partial chain cap: condition became irreversible.",
        )

    def _apply_last_chance_outcome(self, arrival_outcome: ArrivalOutcome) -> tuple[ArrivalOutcome, str | None]:
        """Fix 3: near final attempt, allow emergency stabilization chance."""
        assert self.state_data is not None
        if arrival_outcome.status == "accepted":
            return arrival_outcome, None
        # Apply only on the literal final step, not one step earlier.
        if self.state_data.step != self.state_data.max_steps:
            return arrival_outcome, None

        chance = {
            "easy": 0.35,
            "medium": 0.18,
            "hard": 0.02,
        }.get(self.state_data.scenario_difficulty, 0.18)

        reward_modifier = {
            "easy": 0.82,
            "medium": 0.70,
            "hard": 0.58,
        }.get(self.state_data.scenario_difficulty, 0.70)

        if self._rng.random() < chance:
            return (
                ArrivalOutcome(
                    status="accepted",
                    reason="Emergency stabilization at last attempt",
                    validation_details=arrival_outcome.validation_details,
                    reward_modifier=reward_modifier,
                ),
                "Last-chance rule: emergency stabilization triggered.",
            )
        return arrival_outcome, None

    def _apply_early_reject_protection(self, arrival_outcome: ArrivalOutcome) -> tuple[ArrivalOutcome, str | None]:
        """Avoid excessive instant dead-ends by softening some step-1 rejections."""
        assert self.state_data is not None
        if arrival_outcome.status != "rejected":
            return arrival_outcome, None
        if self.state_data.step >= 2:
            return arrival_outcome, None
        soften_reject_chance = 0.3 if self.state_data.scenario_difficulty != "hard" else 0.05
        if self._rng.random() >= soften_reject_chance:
            return arrival_outcome, None

        return (
            ArrivalOutcome(
                status="partial",
                reason="Early rejection mitigated by emergency field stabilization",
                validation_details=arrival_outcome.validation_details,
                reward_modifier=0.50,
                terminal=False,
            ),
            "Recovery guard: early rejection softened to partial.",
        )

    def _apply_late_partial_recovery(self, arrival_outcome: ArrivalOutcome) -> tuple[ArrivalOutcome, str | None]:
        """Allow realistic comeback from partial outcomes after initial stabilization attempts."""
        assert self.state_data is not None
        if arrival_outcome.status != "partial":
            return arrival_outcome, None
        if self.state_data.step < 2:
            return arrival_outcome, None
        recovery_trigger = 0.5 if self.state_data.scenario_difficulty != "hard" else 0.25
        if self._rng.random() >= recovery_trigger:
            return arrival_outcome, None

        reject_from_partial = 0.5 if self.state_data.scenario_difficulty != "hard" else 0.8
        if self._rng.random() < reject_from_partial:
            return (
                ArrivalOutcome(
                    status="rejected",
                    reason="Condition relapsed after temporary stabilization",
                    validation_details=arrival_outcome.validation_details,
                    reward_modifier=0.0,
                    terminal=False,
                ),
                "Recovery guard: partial relapsed to rejected.",
            )

        return (
            ArrivalOutcome(
                status="accepted",
                reason="Condition stabilized after progressive treatment",
                validation_details=arrival_outcome.validation_details,
                reward_modifier=max(0.7, float(arrival_outcome.reward_modifier)),
                terminal=False,
            ),
            "Recovery guard: partial upgraded to accepted after continued care.",
        )

    def _build_last_info(
        self,
        reward: float,
        breakdown: RewardBreakdown,
        arrival_outcome: ArrivalOutcome,
    ) -> None:
        assert self.state_data is not None

        grader_result = None
        if self.state_data.done:
            grader_result = grade_task(
                task_id=self.state_data.task_id,
                difficulty=self.state_data.scenario_difficulty,
                objective=self.state_data.task_objective,
                trajectory=self.trajectory,
            )

        self.last_info = StepInfo(
            task_id=self.state_data.task_id,
            difficulty=self.state_data.scenario_difficulty,
            objective=self.state_data.task_objective,
            progress_score=self._progress_score(),
            reward_model=RewardModel(value=reward, breakdown=breakdown),
            grader=grader_result,
            last_action_error=None,
            outcome={
                "status": arrival_outcome.status,
                "reason": arrival_outcome.reason,
            },
        )

    def _record_trajectory(
        self,
        selected: HospitalState,
        arrival_outcome: ArrivalOutcome,
        reward: float,
        travel_time: float,
        dynamic_delay: float,
        original_traffic: str,
    ) -> None:
        assert self.state_data is not None
        self.trajectory.append(
            {
                "step": self.state_data.step,
                "state": {
                    "patient_condition": self.state_data.patient_condition,
                    "remaining_time_minutes": self.state_data.critical_time_limit_minutes,
                    "visited_hospitals": list(self.state_data.visited_hospitals),
                    "failed_hospitals": list(self.state_data.failed_hospitals),
                },
                "action": {
                    "hospital_id": selected.hospital_id,
                    "traffic_before": original_traffic,
                    "traffic_at_arrival": selected.traffic,
                },
                "outcome_status": arrival_outcome.status,
                "outcome_reason": arrival_outcome.reason,
                "reward": reward,
                "travel_time": travel_time,
                "dynamic_delay": dynamic_delay,
                "critical_limit": self.state_data.critical_time_limit_minutes,
                "specialization_match": self._specialization_match(selected),
                "suitability_score": arrival_outcome.validation_details.patient_suitability if arrival_outcome.validation_details else 0.5,
            }
        )

    def _build_observation(self) -> Observation:
        assert self.state_data is not None

        last_outcome_obs = None
        if self.state_data.last_arrival_outcome and self.state_data.last_arrival_outcome.validation_details:
            last_outcome_obs = ArrivalOutcomeObservation(
                status=self.state_data.last_arrival_outcome.status,
                reason=self.state_data.last_arrival_outcome.reason,
                suitability_score=self.state_data.last_arrival_outcome.validation_details.patient_suitability,
            )

        return Observation(
            episode_id=self.state_data.episode_id,
            seed=self.state_data.seed,
            task_id=self.state_data.task_id,
            task_objective=self.state_data.task_objective,
            scenario_type=self.state_data.scenario_type,
            scenario_name=self.state_data.scenario_name,
            scenario_difficulty=self.state_data.scenario_difficulty,
            patient_condition=self.state_data.patient_condition,
            required_specialization=self.state_data.required_specialization,
            initial_critical_time_limit_minutes=self.state_data.initial_critical_time_limit_minutes,
            critical_time_limit_minutes=self.state_data.critical_time_limit_minutes,
            remaining_time_minutes=self.state_data.critical_time_limit_minutes,
            step=self.state_data.step,
            max_steps=self.state_data.max_steps,
            hospitals=[
                HospitalObservation(
                    hospital_id=h.hospital_id,
                    distance_km=h.distance_km,
                    icu=h.icu_display,
                    specialization=h.specialization,
                    traffic=h.traffic,
                )
                for h in self.state_data.hospitals
            ],
            previous_action=self.state_data.selected_hospital_id,
            ambulance_status=self.state_data.ambulance_status,
            current_location_context=self.state_data.current_location_context,
            visited_hospitals=list(self.state_data.visited_hospitals),
            failed_hospitals=list(self.state_data.failed_hospitals),
            recent_failed_hospitals=list(self.state_data.recent_failed_hospitals),
            failed_reasons=dict(self.state_data.failed_reasons),
            total_time_spent_minutes=self.state_data.total_time_spent_minutes,
            rerouting_reason=self.state_data.rerouting_reason,
            last_arrival_outcome=last_outcome_obs,
            explanation=list(self.state_data.explanation),
            memory_snapshot={k: v.model_dump() for k, v in self.state_data.memory.items()},
        )

    def _evolve_hospital_uncertainty(self) -> None:
        assert self.state_data is not None
        for hospital in self.state_data.hospitals:
            if self._rng.random() < 0.40:
                hospital.traffic = self._traffic_shift(hospital.traffic, self.state_data.scenario_difficulty)

            if self._rng.random() < DifficultyModifier.get_icu_mismatch_probability(self.state_data.scenario_difficulty):
                hospital.icu_actual = not hospital.icu_actual

            if hospital.icu_actual:
                hospital.icu_display = "available" if self._rng.random() < 0.80 else "unknown"
            else:
                hospital.icu_display = "available" if self._rng.random() < 0.2 else "unknown"

    def _traffic_shift(self, current: str, difficulty: str) -> str:
        worsening_prob = {"easy": 0.12, "medium": 0.25, "hard": 0.38}.get(difficulty, 0.25)
        improving_prob = {"easy": 0.18, "medium": 0.10, "hard": 0.06}.get(difficulty, 0.10)

        if current == "low":
            if self._rng.random() < worsening_prob:
                return "medium"
            return "low"

        if current == "medium":
            roll = self._rng.random()
            if roll < worsening_prob:
                return "high"
            if roll < worsening_prob + improving_prob:
                return "low"
            return "medium"

        if self._rng.random() < improving_prob:
            return "medium"
        return "high"

    def _sample_scenario_for_difficulty(self, difficulty: str) -> tuple[dict[str, Any], str]:
        generators = [
            (generate_medical_case, "medical"),
            (generate_accident_case, "accident"),
            (generate_fire_case, "fire"),
        ]
        for _ in range(64):
            generator, scenario_type = self._rng.choice(generators)
            scenario = generator(self._rng)
            if scenario["difficulty"] == difficulty:
                return scenario, scenario_type

        for generator, scenario_type in generators:
            scenario = generator(self._rng)
            if scenario["difficulty"] == difficulty:
                return scenario, scenario_type
        return scenario, scenario_type

    def _find_hospital(self, hospital_id: str) -> HospitalState | None:
        assert self.state_data is not None
        for hospital in self.state_data.hospitals:
            if hospital.hospital_id == hospital_id:
                return hospital
        return None

    def _load_memory(self) -> dict[str, LearningEntry]:
        text = self.memory_path.read_text(encoding="utf-8-sig").strip()
        raw = json.loads(text) if text else {}
        return {k: LearningEntry(**v) for k, v in raw.items()}

    def _update_learning_memory(self, hospital_id: str, success: bool, reward: float) -> None:
        memory = self._load_memory()
        entry = memory.get(hospital_id, LearningEntry())

        if success:
            entry.success += 1
            entry.accepted += 1
        else:
            entry.fail += 1
            entry.rejected += 1

        total = entry.success + entry.fail
        if total == 1:
            entry.avg = max(0.0, min(1.0, reward))
        else:
            normalized_reward = max(0.0, min(1.0, reward))
            entry.avg = ((entry.avg * (total - 1)) + normalized_reward) / total

        memory[hospital_id] = entry
        serialized = {k: v.model_dump() for k, v in memory.items()}
        self.memory_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    def _progress_score(self) -> float:
        if not self.trajectory:
            return MIN_REWARD
        raw = sum(float(t["reward"]) for t in self.trajectory) / len(self.trajectory)
        return max(MIN_REWARD, min(MAX_REWARD, raw))

    def _failure_score(self) -> float:
        assert self.state_data is not None
        progress_component = self._progress_score()
        reward_component = max(MIN_REWARD, min(MAX_REWARD, self.state_data.reward))
        score = 0.15 + (0.35 * reward_component) + (0.25 * progress_component)
        return max(0.1, min(0.85, score))

    def _success_score(self) -> float:
        assert self.state_data is not None
        progress_component = self._progress_score()
        reward_component = max(MIN_REWARD, min(MAX_REWARD, self.state_data.reward))
        total_steps = max(1, len(self.trajectory))
        rejected_steps = sum(1 for item in self.trajectory if item.get("outcome_status") == "rejected")
        route_quality = max(0.0, 1.0 - (rejected_steps / total_steps))
        score = (0.45 * reward_component) + (0.40 * progress_component) + (0.15 * route_quality)
        return max(0.25, min(0.99, score))


ACDEEnvironment = EmergencyEnv
