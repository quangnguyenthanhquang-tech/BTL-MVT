"""Run from repository root: python -m team.env_test."""

import argparse
import html
import importlib.metadata
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from team.agents.first_fit import FirstFitAgent
from team.env_factory import create_env


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def git_info(root):
    result = {}
    for key, args in (("commit", ["rev-parse", "HEAD"]),
                      ("working_tree", ["status", "--porcelain"])):
        try:
            result[key] = subprocess.check_output(
                ["git", "-C", str(root), *args], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            result[key] = "unavailable"
    return result


def write_report(report, folder, stamp):
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / f"env_test_{stamp}.json"
    html_path = folder / f"env_test_{stamp}.html"
    json_path.write_text(json.dumps(serializable(report), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding="utf-8")
    esc = lambda x: html.escape(str(x))
    tests = "".join(
        f"<tr><td>{esc(t['name'])}</td><td>{esc(t['result'])}</td>"
        f"<td>{esc(t['details'])}</td></tr>" for t in report["checks"]
    )
    rows = "".join(
        "<tr>" + "".join(f"<td>{esc(s.get(k, ''))}</td>" for k in
        ("step", "action", "valid_actions_including_reject", "mask_selected",
         "reward", "status", "terminated", "truncated")) + "</tr>"
        for s in report["steps"]
    )
    summary = esc(json.dumps(serializable({k: v for k, v in report.items()
                                          if k not in ("steps", "checks")}),
                            ensure_ascii=False, indent=2))
    html_path.write_text(f"""<!doctype html><html lang="vi"><meta charset="utf-8">
<title>Kiểm thử môi trường nhóm</title><style>
body{{font:16px system-ui;max-width:1100px;margin:35px auto;padding:0 18px;color:#172033}}
table{{border-collapse:collapse;width:100%;margin:18px 0}}td,th{{border:1px solid #ccd3df;padding:9px;text-align:left}}
th{{background:#edf2fa}}pre{{background:#f4f6fa;padding:16px;overflow:auto;white-space:pre-wrap}}
.scroll{{overflow:auto}}</style><h1>Kiểm thử môi trường nhóm</h1>
<p>Kết quả chức năng của lần chạy này. Không xác nhận độ chính xác vật lý GN/EGN
hoặc hiệu năng thống kê dài hạn. Báo cáo không thu thập trạng thái phổ từng link.</p>
<h2>Các phép kiểm tra</h2><table><tr><th>Kiểm tra</th><th>Kết quả</th><th>Chi tiết</th></tr>{tests}</table>
<h2>Cấu hình, phiên bản và tổng kết</h2><pre>{summary}</pre>
<h2>Từng bước</h2><p>Số action hợp lệ tính tất cả phần tử 1 trong mask, bao gồm action từ chối nếu được bật.</p>
<div class="scroll"><table><tr><th>Bước</th><th>Action</th><th>Action hợp lệ</th><th>Mask[action]</th>
<th>Reward</th><th>Status</th><th>Terminated</th><th>Truncated</th></tr>{rows}</table></div>
</html>""", encoding="utf-8")
    return json_path, html_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load", type=float, default=300.0)
    parser.add_argument("--episode-length", type=int, default=100)
    args = parser.parse_args()
    if args.episode_length <= 0 or not np.isfinite(args.load) or args.load <= 0:
        parser.error("episode-length và load phải dương; load phải hữu hạn")
    root = Path(__file__).resolve().parents[1]
    now = datetime.now(timezone.utc)
    report = {
        "generated_at_utc": now.isoformat(),
        "config": {"seed": args.seed, "load": args.load,
                   "episode_length": args.episode_length},
        "python": sys.version, "platform": platform.platform(),
        "versions": {n: version(n) for n in
                     ("optical-networking-gym", "gymnasium", "numpy")},
        "git": git_info(root), "algorithm": "first_fit", "checks": [],
        "steps": [], "summary": {},
        "factory_source": (root / "team" / "env_factory.py").read_text(encoding="utf-8"),
    }

    def check(name, condition, details):
        passed = bool(condition)
        report["checks"].append({"name": name,
                                  "result": "PASS" if passed else "FAIL",
                                  "details": str(details)})
        if not passed:
            raise AssertionError(f"{name}: {details}")

    env = None
    started = perf_counter()
    try:
        env = create_env(report["config"])
        agent = FirstFitAgent()
        obs, info = env.reset(seed=args.seed)
        check("reset_info_dictionary", isinstance(info, dict), type(info).__name__)
        check("reset_observation_in_space", env.observation_space.contains(obs),
              str(env.observation_space))
        check("discrete_action_space", hasattr(env.action_space, "n")
              and getattr(env.action_space, "start", 0) == 0, str(env.action_space))
        report["action_space"] = str(env.action_space)
        report["observation_space"] = str(env.observation_space)
        total_reward = 0.0
        for index in range(args.episode_length):
            mask = info.get("mask")
            if mask is None:
                mask = env.action_masks()
            if mask is None:
                raise AssertionError("Không có action mask")
            mask = np.asarray(mask).copy()
            if mask.shape != (int(env.action_space.n),):
                raise AssertionError(f"Mask shape không khớp: {mask.shape}")
            if not np.all((mask == 0) | (mask == 1)):
                raise AssertionError("Mask có giá trị khác 0/1")
            action = agent.solution(obs, info, env)
            if not env.action_space.contains(action):
                raise AssertionError(f"Action ngoài action_space: {action}")
            if not bool(mask[int(action)]):
                raise AssertionError(f"Action {action} bị mask loại")
            obs, reward, terminated, truncated, info = env.step(action)
            if not isinstance(info, dict):
                raise AssertionError("step info không phải dictionary")
            if not env.observation_space.contains(obs):
                raise AssertionError("step observation ngoài observation_space")
            if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
                raise AssertionError("Cờ kết thúc không phải boolean")
            reward = float(reward)
            if not np.isfinite(reward):
                raise AssertionError("Reward không hữu hạn")
            total_reward += reward
            report["steps"].append({
                "step": index + 1, "action": int(action),
                "valid_actions_including_reject": int(np.count_nonzero(mask)),
                "mask_selected": int(mask[int(action)]), "mask_before_action": mask.tolist(),
                "reward": reward, "status": serializable(info.get("status")),
                "terminated": bool(terminated), "truncated": bool(truncated),
            })
            if (index + 1) % 10 == 0:
                print(f"Step={index + 1}, reward={reward}")
            if terminated or truncated:
                break
        check("per_step_contract_and_mask", True,
              f"Đã kiểm tra {len(report['steps'])} bước: mask, action, observation, reward, cờ kết thúc")
        check("episode_end_signal", terminated or truncated,
              f"terminated={terminated}, truncated={truncated}")
        check("configured_episode_length", len(report["steps"]) == args.episode_length,
              f"actual={len(report['steps'])}, expected={args.episode_length}")
        metrics = {}
        for key in ("episode_service_blocking_rate", "episode_bit_rate_blocking_rate"):
            value = float(info[key])
            check(key + "_range", np.isfinite(value) and 0 <= value <= 1, value)
            metrics[key] = value
        report["summary"] = {"steps": len(report["steps"]),
                             "total_reward": total_reward, **metrics}
        report["outcome"] = "PASS"
    except Exception:
        report["outcome"] = "FAIL"
        report["error"] = traceback.format_exc()
        report["checks"].append({"name": "run_completed_without_error",
                                  "result": "FAIL", "details": report["error"]})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                report["outcome"] = "FAIL"
                report["checks"].append({"name": "close", "result": "FAIL",
                                          "details": traceback.format_exc()})
    report["runtime_seconds"] = perf_counter() - started
    paths = write_report(report, root / "team" / "results",
                         now.strftime("%Y%m%dT%H%M%S_%fZ"))
    print("Kết quả:", report["outcome"])
    for path in paths:
        print(path)
    if report.get("error"):
        print(report["error"])
    return 0 if report["outcome"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
