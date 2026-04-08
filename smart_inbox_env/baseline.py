def evaluate_performance(scores):
    avg = sum(scores) / len(scores)

    if avg >= 0.8:
        rating = "Excellent"
    elif avg >= 0.6:
        rating = "Good"
    elif avg >= 0.4:
        rating = "Average"
    else:
        rating = "Poor"

    return avg, rating

from env.environment import SmartInboxEnv
from env.models import EmailAction


def get_agent_action(observation):
    text = (observation.subject + " " + observation.body).lower()

    if any(word in text for word in ["free", "win", "lottery", "offer", "click", "verify account", "suspend", "prize"]):
        return EmailAction(
            classification="spam",
            action="ignore",
            response=""
        )

    if any(word in text for word in ["outage", "crash", "server down", "cpu", "urgent", "immediate"]):
        return EmailAction(
            classification="important",
            action="escalate",
            response=""
        )

    if any(word in text for word in ["refund", "not received", "interview", "deadline", "login", "security"]):
        return EmailAction(
            classification="important",
            action="reply",
            response="We have received your request and will respond shortly."
        )

    return EmailAction(
        classification="normal",
        action="ignore",
        response=""
    )


def main():
    env = SmartInboxEnv()

    for task in ["easy", "medium", "hard"]:
        obs = env.reset(task=task)
        action = get_agent_action(obs)

        _, reward, _, _ = env.step(action)

        print(f"{task.upper()} SCORE: {reward.score}")

    avg, rating = evaluate_performance(scores)

    print("\nFINAL EVALUATION")
    print("Average Score:", round(avg, 2))
    print("Performance:", rating)


if __name__ == "__main__":
    main()
