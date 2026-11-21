"""Shared experiment presets used by CLI and demo."""
MODULATIONS = "BPSK, QPSK, 8QAM, 16QAM, 32QAM, 64QAM"
TOPOLOGIES = {"nobel-eu": "nobel_eu_baseline", "ring_4": "ring4_quickstart", "apan": "nobel_eu_baseline", "hcm": "nobel_eu_baseline"}
SCENARIOS = {
    "KB1 – Mạng chuẩn (320 slot)": dict(num_slots=320, bit_rates=(10, 40, 100, 400), desc="320 slot/link, lưu lượng 10/40/100/400 Gb/s."),
    "KB2 – Siết phổ (80 slot)": dict(num_slots=80, bit_rates=(10, 40, 100, 400), desc="80 slot/link; khảo sát khi tài nguyên phổ hạn chế."),
    "KB3 – Đa dịch vụ (40/100/400G)": dict(num_slots=320, bit_rates=(40, 100, 400), desc="320 slot/link, lưu lượng 40/100/400 Gb/s."),
}
