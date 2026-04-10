"""Hospital validation engine for realistic arrival outcomes.

Simulates hidden validation checks performed when an ambulance arrives at a hospital.
Outcomes are based on difficulty level, hospital capacity, patient suitability, and randomness.
"""

from typing import cast, Literal

from app.models.state import ArrivalOutcome, HospitalValidationDetails, HospitalState
from app.utils.randomizer import SeededRandomizer


class HospitalValidator:
    """Performs hidden validation checks on arrival and returns outcome."""

    def __init__(self, rng: SeededRandomizer):
        self.rng = rng

    def validate_arrival(
        self,
        hospital: HospitalState,
        difficulty: str,
        patient_condition: str,
        required_specialization: str,
        total_time_spent: float,
        critical_time_limit: float,
        step_number: int = 1,
    ) -> ArrivalOutcome:
        """
        Perform hidden validation check when ambulance arrives at hospital.
        
        Returns outcome with status: accepted, partial, or rejected.
        Difficulty affects likelihood and uncertainty of failures.
        """
        
        # 1. ICU Availability (seeded + difficulty-driven)
        icu_available = self._check_icu_availability(hospital, difficulty)
        
        # 2. Doctor/Specialist Availability
        specialization_match = (
            hospital.specialization == required_specialization
            or hospital.specialization == "general"
            or required_specialization == "general"
        )
        doctor_available = self._check_doctor_availability(
            hospital, specialization_match, difficulty
        )
        
        # 3. Equipment Functionality
        equipment_functional = self._check_equipment_functional(difficulty)
        
        # 4. Hospital Overload
        overload_status = self._check_hospital_overload(difficulty)
        
        # 5. Patient Suitability Score
        patient_suitability = self._compute_patient_suitability(
            hospital,
            patient_condition,
            required_specialization,
            overload_status,
            difficulty,
        )
        
        # Determine outcome based on checks
        validation_details = HospitalValidationDetails(
            icu_available=icu_available,
            doctor_available=doctor_available,
            equipment_functional=equipment_functional,
            overload_status=cast(Literal["clear", "moderate", "severe"], overload_status),
            patient_suitability=patient_suitability,
        )
        
        status, reason, reward_modifier, terminal = self._determine_outcome(
            validation_details,
            total_time_spent,
            critical_time_limit,
            patient_condition,
            specialization_match,
            difficulty,
            step_number,
        )
        
        return ArrivalOutcome(
            status=cast(Literal["accepted", "partial", "rejected"], status),
            reason=reason,
            validation_details=validation_details,
            reward_modifier=reward_modifier,
            terminal=terminal,
        )

    def _check_icu_availability(self, hospital: HospitalState, difficulty: str) -> bool:
        """Generate ICU actual availability from seeded difficulty priors with display influence."""
        base_true_prob = {
            "easy": 0.90,
            "medium": 0.78,
            "hard": 0.66,
        }.get(difficulty, 0.70)

        # Displayed status influences belief but does not fully determine truth.
        display_adjust = 0.0
        if hospital.icu_display == "available":
            display_adjust = 0.06 if difficulty == "easy" else (0.04 if difficulty == "medium" else 0.02)
        else:  # unknown
            display_adjust = -0.03 if difficulty == "easy" else (-0.02 if difficulty == "medium" else 0.0)

        p = max(0.05, min(0.97, base_true_prob + display_adjust))
        return self.rng.random() < p

    def _check_doctor_availability(
        self,
        hospital: HospitalState,
        specialization_match: bool,
        difficulty: str,
    ) -> bool:
        """Check if required specialist/doctor is available."""
        base_prob = {
            "easy": 0.92,
            "medium": 0.85,
            "hard": 0.72,
        }.get(difficulty, 0.80)
        # Mismatch should materially reduce specialist availability.
        if not specialization_match:
            base_prob -= 0.30 if difficulty != "hard" else 0.25
        return self.rng.random() < max(0.05, min(0.98, base_prob))

    def _check_equipment_functional(self, difficulty: str) -> bool:
        """Check if required equipment is functional."""
        equipment_working_prob = {
            "easy": 0.95,
            "medium": 0.90,
            "hard": 0.86,
        }.get(difficulty, 0.90)
        return self.rng.random() < equipment_working_prob

    def _check_hospital_overload(self, difficulty: str) -> str:
        """Determine hospital overload status: clear, moderate, or severe."""
        overload_prob = {
            "easy": 0.10,
            "medium": 0.18,
            "hard": 0.24,
        }.get(difficulty, 0.25)
        if difficulty == "hard":
            overload_prob += 0.10
        overloaded = self.rng.random() < overload_prob
        if not overloaded:
            return "clear"

        # Split overloaded state into moderate vs severe (critical overload).
        severe_given_overload = 0.20 if difficulty == "easy" else (0.35 if difficulty == "medium" else 0.50)
        return "severe" if self.rng.random() < severe_given_overload else "moderate"

    def _compute_patient_suitability(
        self,
        hospital: HospitalState,
        patient_condition: str,
        required_specialization: str,
        overload_status: str,
        difficulty: str,
    ) -> float:
        """
        Compute how suitable this hospital is for the patient (0.0 to 1.0).
        Based on specialization match, condition severity, and overload.
        """
        # Specialization match basis
        spec_match = (
            hospital.specialization == required_specialization
            or hospital.specialization == "general"
            or required_specialization == "general"
        )
        spec_score = 0.85 if spec_match else 0.4
        
        # Patient severity
        severity_map = {
            "critical": 0.3,
            "unstable": 0.6,
            "serious": 0.65,
            "stable": 0.8,
        }
        severity_score = severity_map.get(patient_condition.lower(), 0.5)
        
        # Hospital overload impact
        overload_impact = {
            "clear": 1.0,
            "moderate": 0.7,
            "severe": 0.4,
        }
        overload_factor = overload_impact.get(overload_status, 0.7)
        
        # Combine
        suitability = (spec_score * 0.4) + (severity_score * 0.35) + (overload_factor * 0.25)
        
        # Add difficulty-based noise
        if difficulty == "hard":
            noise = self.rng.uniform(-0.15, 0.15)
            suitability = suitability + noise
        
        # Clamp to strict (0, 1) — validator rejects exact 0.0 and 1.0
        return max(0.001, min(0.999, suitability))

    def _determine_outcome(
        self,
        validation: HospitalValidationDetails,
        total_time_spent: float,
        critical_time_limit: float,
        patient_condition: str,
        specialization_match: bool,
        difficulty: str,
        step_number: int,
    ) -> tuple[str, str, float, bool]:
        """
        Determine final outcome (accepted, partial, or rejected) based on validation.
        
        Returns: (status, reason, reward_modifier)
        """
        
        # Rejection criteria (strict rule set)
        rejection_reasons = []
        overload_critical = validation.overload_status == "severe"
        
        if not validation.icu_available:
            rejection_reasons.append("ICU unavailable")
        
        if not validation.doctor_available:
            rejection_reasons.append("No specialist available")
        
        equipment_issue = not validation.equipment_functional
        
        if overload_critical:
            rejection_reasons.append("Hospital overloaded")

        if not specialization_match:
            rejection_reasons.append("Wrong hospital specialization")
        
        # Rejected if strict checks fail, but some single-failure cases can still lead to risky partial admission.
        if rejection_reasons:
            rescue_chance = {
                "easy": 0.48,
                "medium": 0.28,
                "hard": 0.10,
            }.get(difficulty, 0.28)

            # Allow partial stabilization on specialization mismatch instead of strict rejection.
            if not specialization_match and self.rng.random() < 0.3:
                return (
                    "partial",
                    "Temporary stabilization despite specialization mismatch",
                    0.55,
                    False,
                )

            if len(rejection_reasons) == 1 and not overload_critical and self.rng.random() < rescue_chance:
                return (
                    "partial",
                    f"Admitted with significant risk: {rejection_reasons[0]}",
                    0.6,
                    False,
                )

            # Fix 1: hard mode keeps a real but limited chance of successful intervention.
            hard_success_prob = 0.06
            if difficulty == "hard" and step_number == 1:
                hard_success_prob *= 0.2

            if difficulty == "hard" and self.rng.random() < hard_success_prob:
                return (
                    "accepted",
                    "Successful critical intervention under extreme conditions",
                    0.9,
                    False,
                )

            return (
                "rejected",
                f"Hospital cannot admit: {', '.join(rejection_reasons[:2])}",
                0.001,
                False,
            )
        
        # Partial admission checks: no hard check failed, but response is delayed/risky.
        partial_factors = []

        delay_factor = {
            "easy": 0.05,
            "medium": 0.12,
            "hard": 0.2,
        }.get(difficulty, 0.12)
        doctor_delayed = self.rng.random() < delay_factor
        patient_worsened = (
            patient_condition.lower() in {"critical", "unstable"}
            and self.rng.random() < (0.08 if difficulty == "easy" else 0.18 if difficulty == "medium" else 0.3)
        )
        # No hard deadline window: use prolonged transfer strain instead.
        strain_threshold = {
            "easy": 18.0,
            "medium": 15.0,
            "hard": 12.0,
        }.get(difficulty, 15.0)
        time_pressure = total_time_spent > strain_threshold

        if time_pressure:
            partial_factors.append("prolonged transfer strain")

        if doctor_delayed:
            partial_factors.append("doctor delayed")

        if patient_worsened:
            partial_factors.append("patient worsened during transfer")

        if equipment_issue:
            partial_factors.append("equipment issue")

        if validation.overload_status == "moderate":
            partial_factors.append("hospital busy but manageable")
        
        # Partial admission
        if partial_factors:
            reward_modifier = 0.65 if len(partial_factors) >= 2 else 0.8

            # Fix 1: stabilization probability reduced and conditioned by difficulty and severity.
            stabilization_prob = {
                "easy": 0.5,
                "medium": 0.18,
                "hard": 0.03,
            }.get(difficulty, 0.25)
            if patient_condition.lower() in {"critical", "unstable"}:
                stabilization_prob *= 0.55
            if step_number == 1 and difficulty in {"medium", "hard"}:
                stabilization_prob *= 0.45

            if self.rng.random() < stabilization_prob:
                return (
                    "accepted",
                    "Patient stabilized after delayed admission",
                    0.9,
                    False,
                )

            # Fix 2: partial outcomes can deteriorate into rejection.
            if self.rng.random() < 0.3:
                return (
                    "partial",
                    "Critical deterioration managed temporarily; reroute still needed",
                    0.45,
                    False,
                )

            if self.rng.random() < 0.3:
                return (
                    "rejected",
                    "Condition became non-transferable during delay; immediate critical care failed",
                    0.001,
                    True,
                )

            return (
                "partial",
                f"Admitted with delays: {', '.join(partial_factors[:2])}",
                reward_modifier,
                    False,
            )
        
        # Full acceptance
        confidence_bonus = 0.999
        if validation.patient_suitability >= 0.8:
            confidence_bonus = 1.1
        elif validation.patient_suitability >= 0.7:
            confidence_bonus = 1.05

        # Arrival uncertainty by difficulty.
        reject_prob = 0.0
        if difficulty == "medium":
            reject_prob = 0.2
        elif difficulty == "hard":
            reject_prob = 0.12
            reject_prob += 0.10
            reject_prob += 0.08

        if reject_prob > 0.0 and self.rng.random() < reject_prob:
            return (
                "rejected",
                "Unexpected complication at arrival",
                0.001,
                False,
            )

        if difficulty == "medium" and self.rng.random() < 0.05:
            return (
                "accepted",
                "successful admission under uncertainty",
                0.999,
                False,
            )
        
        if step_number == 1 and difficulty in {"medium", "hard"}:
            direct_accept_prob = {"medium": 0.48, "hard": 0.20}.get(difficulty, 0.48)
            if patient_condition.lower() in {"critical", "unstable"}:
                direct_accept_prob *= 0.85
            if self.rng.random() > direct_accept_prob:
                return (
                    "partial",
                    "Initial triage completed; transfer monitoring still required",
                    0.62 if difficulty == "medium" else 0.55,
                    False,
                )

        accepted_prob = 1.0
        if difficulty == "hard":
            accepted_prob *= 0.65
        if self.rng.random() > accepted_prob:
            return (
                "partial",
                "Initial treatment started but full admission remains uncertain",
                0.58,
                False,
            )

        return (
            "accepted",
            "Patient admitted and treatment began",
            confidence_bonus,
            False,
        )


class DifficultyModifier:
    """Manages difficulty-specific modifiers across the system."""

    @staticmethod
    def get_icu_mismatch_probability(difficulty: str) -> float:
        """Probability of hidden ICU mismatch (shown vs actual)."""
        return {"easy": 0.0, "medium": 0.15, "hard": 0.35}.get(difficulty, 0.15)

    @staticmethod
    def get_unexpected_event_probability(difficulty: str) -> float:
        """Probability of unexpected events (delays, recovery)."""
        return {"easy": 0.05, "medium": 0.18, "hard": 0.30}.get(difficulty, 0.18)

    @staticmethod
    def get_minimum_survival_probability(difficulty: str) -> float:
        """Floor below which patient won't survive regardless."""
        return {"easy": 0.05, "medium": 0.02, "hard": 0.0}.get(difficulty, 0.02)

    @staticmethod
    def get_initial_condition_variance(difficulty: str) -> float:
        """How much patient condition can vary initially."""
        return {"easy": 0.0, "medium": 0.1, "hard": 0.25}.get(difficulty, 0.1)
