from env.environment import SmartInboxEnv
from env.models import EmailAction


def get_agent_action(observation):
    """
    Simple rule-based agent (no API needed)
    """

    text = (observation.subject + " " + observation.body).lower()

    # Rule-based logic
    if "free" in text or "win" in text:
        return EmailAction(
            classification="spam",
            action="ignore",
            response=""
        )

    elif "refund" in text or "not received" in text:
        return EmailAction(
            classification="important",
            action="reply",
            response="We are processing your request."
        )

    elif "meeting" in text:
        return EmailAction(
            classification="important",
            action="reply",
            response="Noted. Thanks for the update."
        )

    else:
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


if __name__ == "__main__":
    main()