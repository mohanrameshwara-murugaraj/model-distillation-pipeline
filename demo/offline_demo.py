"""No external requests. Shows the routing contract, NOT student quality."""
from dataclasses import dataclass

from distill_lab.providers import Prediction
from distill_lab.router import InferenceRouter


@dataclass
class FixtureStudent:
    def predict(self, text):
        if "unknown" in text.lower():
            raise ValueError("Fixture: unsupported label")
        return Prediction("card_arrival", 4.0, confidence=.97, model="fixture-student")


@dataclass
class FixtureTeacher:
    def predict(self, text):
        return Prediction("change_pin", 10.0, model="fixture-teacher")


def main():
    router = InferenceRouter(FixtureStudent(), FixtureTeacher(),
                             ["card_arrival", "change_pin"], min_confidence=.6)
    for question in ["Where is my card?", "Unknown query that fails student fixture"]:
        result = router.predict(question)
        print({"route": result.route, "intent": result.prediction.intent,
               "escalated": result.escalated, "reason": result.reason})
    print("FIXTURES ONLY — no accuracy, GPU or API results are implied.")


if __name__ == "__main__":
    main()
