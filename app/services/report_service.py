from .risk_engine import RiskEngine


class ReportService:
    def build(self, analysis):
        results = [result.to_dict() for result in analysis.results]
        risk = RiskEngine().assess(
            [
                {
                    "module_name": result.module_name,
                    "risk_category": result.risk_category,
                    "availability_status": result.availability_status,
                }
                for result in analysis.results
            ]
        )
        return {"analysis": analysis.to_dict(), "results": results, "risk": risk}
