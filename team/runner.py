import numpy as np
def run_episode(env, agent, seed):
    total_reward = 0.0
    steps = 0

    obs, info = env.reset(seed=seed)

    while True:
        action = agent.solution(obs, info, env)

        if not env.action_space.contains(action):
            raise ValueError(f"Action không hợp lệ: {action}")
        
        mask = info.get("mask")
        if mask is None:
            mask = env.action_masks()

        if mask is None:
            raise RuntimeError("Không lấy được action mask.")

        mask = np.asarray(mask)

        if mask.shape != (env.action_space.n,):
            raise ValueError(
                f"Mask không khớp action_space: {mask.shape}"
            )

        if not bool(mask[int(action)]):
            raise ValueError(
                f"{agent.name}: action {action} bị mask loại."
            )
        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1
        total_reward += float(reward)

        if steps % 10 == 0 or terminated or truncated:
            print(f"Step={steps}, reward={reward}")

        if terminated or truncated:
            break

    return {
        "steps": steps,
        "total_reward": total_reward,
        "request_blocking_ratio": float(
            info["episode_service_blocking_rate"]
        ),
        "bit_rate_blocking_ratio": float(
            info["episode_bit_rate_blocking_rate"]
        ),
    }