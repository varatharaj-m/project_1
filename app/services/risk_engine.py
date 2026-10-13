WEIGHTS = {
    "cyberbullying": 1.0,
    "misinformation": 0.9,
    "image_manipulation": 0.8,
    "video_deepfake": 1.0,
    "synthetic_audio": 0.8,
}

RISK_NUMERIC = {"low": 0.2, "medium": 0.55, "high": 0.85, "needs_review": 0.6}


class RiskEngine:
    def assess(self, results):
        if not results:
            return {"risk_category": "needs_review", "signals": [], "warnings": ["No model results available."]}
        weighted = 0.0
        total = 0.0
        warnings = []
        signals = []
        for result in results:
            module = result["module_name"]
            weight = WEIGHTS.get(module, 0.5)
            weighted += RISK_NUMERIC.get(result["risk_category"], 0.6) * weight
            total += weight
            signals.append({"module": module, "risk_category": result["risk_category"], "status": result["availability_status"]})
            if result["availability_status"] != "available":
                warnings.append(f"{module} result is {result['availability_status']}.")
        score = weighted / total if total else 0.6
        category = "high" if score >= 0.75 else "medium" if score >= 0.45 else "low"
        if warnings:
            category = "needs_review" if category == "low" else category
        return {
            "risk_category": category,
            "score": round(score, 3),
            "signals": signals,
            "warnings": warnings,
            "formula": "weighted average of module risk priorities; unavailable modules add uncertainty but are not treated as proof.",
            "recommended_next_steps": ["Review model explanations.", "Escalate uncertain or high-risk items for human review."],
        }
