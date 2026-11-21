from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any


# =====================================================================
# 1. XỬ LÝ ĐƯỜNG DẪN KHI team/ VÀ sim_core.py LỆCH CẤP
# =====================================================================
def _setup_paths() -> None:
  """Tự động dò tìm và thêm thư mục chứa `team/` và `sim_core.py` vào `sys.path`."""
  here = Path(__file__).resolve().parent
  candidates = [
      here,
      here.parent,
      here.parent.parent,
      *[p.parent for p in here.rglob("sim_core.py")],
      *[p.parent for p in here.parent.rglob("sim_core.py")],
  ]
  for path in candidates:
    if path.exists() and path.is_dir():
      s = str(path)
      if s not in sys.path:
        sys.path.insert(0, s)


_setup_paths()

from sim_core import load_topology_view, make_env_for, modulation_table, step_request
from team.config import SCENARIOS, TOPOLOGIES


# =====================================================================
# 2. CẤU TRÚC DỮ LIỆU CHO BÀI TOÁN RMSA
# =====================================================================
@dataclass
class RMSADecision:
  """Kết quả lựa chọn của thuật toán RMSA cho 1 yêu cầu."""

  accepted: bool
  path_index: int | None = None
  modulation_name: str | None = None
  start_slot: int | None = None
  num_slots_req: int | None = None
  reason: str = ""


# =====================================================================
# 3. KHUNG THUẬT TOÁN RMSA (SKELETON)
# =====================================================================
class RMSASkeletonAlgorithm:
  """
    Mã khung giải quyết bài toán RMSA gồm 3 pha:
      1. Routing (Định tuyến): Chọn tuyến đường từ src -> dst.
      2. Modulation (Điều chế): Chọn định dạng điều chế phù hợp ngưỡng GSNR/OSNR & khoảng cách.
      3. Spectrum Assignment (Phân bổ phổ): Cấp phát các khe phổ liên tục & liền kề.
    """

  def __init__(self, mod_order: list[str] | None = None) -> None:
    self.mod_table = modulation_table()
    order = mod_order or list(self.mod_table.keys())
    self.ranked_mods = sorted(
        [m for m in order if m in self.mod_table],
        key=lambda m: self.mod_table[m][1],
        reverse=True,
    )

  def select_route(self, candidate_paths: list[Any]) -> list[tuple[int, Any]]:
    """[Pha 1 - Routing] Trả về danh sách các cặp (path_index, path) theo thứ tự ưu tiên duyệt."""
    return list(enumerate(candidate_paths))

  def select_modulation(
      self, path: Any, bit_rate: float, available_mods: list[Any] | None = None
  ) -> tuple[str, int] | None:
    """[Pha 2 - Modulation] Chọn mức điều chế và số khe phổ (slots) cần dùng."""
    path_km = float(getattr(path, "length_km", 0.0))
    allowed_names = (
        {m.name for m in available_mods} if available_mods else None
    )

    for mod_name in self.ranked_mods:
      if allowed_names is not None and mod_name not in allowed_names:
        continue
      min_osnr_db, se_gbps_per_slot = self.mod_table[mod_name]

      # TODO: Kiểm tra điều kiện chất lượng tín hiệu (GSNR >= min_osnr_db / giới hạn tầm xa)
      if path_km < 0:
        continue

      slots_needed = max(1, int(np_ceil_div(bit_rate, se_gbps_per_slot)))
      return mod_name, slots_needed

    return None

  def assign_spectrum(
      self,
      path: Any,
      num_slots_needed: int,
      slot_occupancy: Any,
  ) -> int | None:
    """[Pha 3 - Spectrum Assignment] Tìm chỉ số slot bắt đầu hợp lệ."""
    link_ids = list(getattr(path, "link_ids", []))
    if not link_ids or slot_occupancy is None:
      return None

    num_links, total_slots = slot_occupancy.shape
    if num_slots_needed > total_slots:
      return None

    for start_slot in range(0, total_slots - num_slots_needed + 1):
      end_slot = start_slot + num_slots_needed
      is_free_on_all_links = True

      for lid in link_ids:
        if lid < 0 or lid >= num_links:
          is_free_on_all_links = False
          break
        if slot_occupancy[lid, start_slot:end_slot].any():
          is_free_on_all_links = False
          break

      if is_free_on_all_links:
        return start_slot

    return None

  def plan_request(
      self,
      candidate_paths: list[Any],
      bit_rate: float,
      slot_occupancy: Any,
      available_mods: list[Any] | None = None,
  ) -> RMSADecision:
    """Điều phối cả 3 pha R -> M -> SA cho một yêu cầu kết nối."""
    for p_idx, path in self.select_route(candidate_paths):
      mod_choice = self.select_modulation(path, bit_rate, available_mods)
      if mod_choice is None:
        continue
      mod_name, slots_needed = mod_choice
      start_slot = self.assign_spectrum(path, slots_needed, slot_occupancy)
      if start_slot is not None:
        return RMSADecision(
            accepted=True,
            path_index=p_idx,
            modulation_name=mod_name,
            start_slot=start_slot,
            num_slots_req=slots_needed,
            reason="OK",
        )
    return RMSADecision(accepted=False, reason="No feasible R-M-SA allocation")


def np_ceil_div(a: float, b: float) -> int:
  q, r = divmod(float(a), float(b))
  return int(q) + (1 if r > 0 else 0)


# =====================================================================
# 4. TEST TOÀN BỘ TOPOLOGY TRONG TOPOLOGIES
# =====================================================================
def test_all_topologies(
    load: float = 150.0,
    n_requests: int = 10,
    steps_per_topo: int = 5,
    seed: int = 42,
) -> None:
  all_topos = list(TOPOLOGIES.keys())
  print(f"Tổng số topology cần test ({len(all_topos)}): {all_topos}")

  scenario_name = next(iter(SCENARIOS))
  sc = SCENARIOS[scenario_name]
  print(
      f"Kịch bản: '{scenario_name}' | slots={sc['num_slots']} | "
      f"bit_rates={sc['bit_rates']} | load={load} | seed={seed}"
  )

  summary: list[tuple[str, str, int, int, bool, str]] = []

  for topo_name in all_topos:
    mapped_scenario = TOPOLOGIES[topo_name]
    print(f"\n================ [{topo_name}] (mapped -> {mapped_scenario}) ================")
    try:
      view = load_topology_view(topo_name)
      print(
          f"Nodes ({len(view.names)}): {view.names} | Links: {len(view.links)} | "
          f"has_geo={view.has_geo} | file={view.source_file}"
      )

      env = make_env_for(
          topology=topo_name,
          load=load,
          n_requests=n_requests,
          num_slots=sc["num_slots"],
          bit_rates=sc["bit_rates"],
          seed=seed,
      )
      try:
        _, info = env.reset(seed=seed)
        for step_idx in range(steps_per_topo):
          rec, info, done, _ = step_request(env, "KSP-FF", info)
          print(
              f"  Req {step_idx}: {rec['src']}->{rec['dst']} | {rec['bit_rate']}G | "
              f"accepted={rec['accepted']} | mod={rec['modulation']} | "
              f"km={rec['path_km']} | slot={rec['slot']} | status={rec['status']}"
          )
          if done:
            break
      finally:
        env.close()

      summary.append((
          topo_name,
          view.source_file,
          len(view.names),
          len(view.links),
          True,
          "OK",
      ))
    except Exception as exc:
      print(f"  [LỖI] {type(exc).__name__}: {exc}")
      summary.append((topo_name, "-", -1, -1, False, f"{type(exc).__name__}: {exc}"))

  print("\n================ BẢNG TỔNG HỢP ================")
  for name, fname, n_nodes, n_links, ok, msg in summary:
    status_str = "PASS" if ok else "FAIL"
    print(f"- {name:12s} | {status_str:4s} | file={fname:16s} | nodes={n_nodes:2d} | links={n_links:2d} | {msg}")


if __name__ == "__main__":
  test_all_topologies()