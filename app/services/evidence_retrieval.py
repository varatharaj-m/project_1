class EvidenceRetriever:
    def retrieve(self, _query):
        return {
            "status": "unavailable",
            "sources": [],
            "explanation": "No external evidence provider is configured. Verification is unavailable.",
        }
