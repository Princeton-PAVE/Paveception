DEPTH_WIDTH  = 848          # Lower res = lower MinZ (closer range)
DEPTH_HEIGHT = 480
FPS          = 30
 
# Advanced mode settings
DISPARITY_SHIFT = 0         # Increase (e.g. 50–128) to push MinZ closer
                            # WARNING: reduces far range proportionally
DEPTH_UNITS = 0.001         # 0.001 = 1mm steps (up to ~65m range)
                            # 0.0001 = 0.1mm steps (up to ~6.5m, finer near detail)
 
# Threshold filter — the depth range you actually want to keep (meters)
THRESHOLD_MIN_DIST = 0.1    # meters — ignore anything closer than this
THRESHOLD_MAX_DIST = 10.0   # meters — raise from default 4m to see far objects
 
# Laser power (0–360) — higher = better returns at distance
LASER_POWER = 360



# ── Advanced mode helpers ──────────────────────────────────────────────────────
DS5_PRODUCT_IDS = [
    "0AD1","0AD2","0AD3","0AD4","0AD5","0AF6",
    "0AFE","0AFF","0B00","0B01","0B03","0B07",
    "0B3A","0B5C","0B64"
]
 
def find_device():
    ctx = rs.context()
    for dev in ctx.query_devices():
        if (dev.supports(rs.camera_info.product_id) and
                str(dev.get_info(rs.camera_info.product_id)) in DS5_PRODUCT_IDS):
            print(f"Found: {dev.get_info(rs.camera_info.name)}")
            return dev
    raise RuntimeError("No compatible D4xx RealSense device found.")
 
def enable_advanced_mode(device):
    adv = rs.rs400_advanced_mode(device)
    if not adv.is_enabled():
        print("Enabling advanced mode (device will reconnect briefly)...")
        adv.toggle_advanced_mode(True)
        time.sleep(5)
        device = find_device()          # re-acquire after reconnect
        adv = rs.rs400_advanced_mode(device)
    print(f"Advanced mode active: {adv.is_enabled()}")
    return adv
 
def apply_advanced_settings(adv):
    depth_table = adv.get_depth_table()
    depth_table.disparityShift = DISPARITY_SHIFT
    # API expects depth units in microseconds (1e-6 m), so 0.001 m → 1000
    depth_table.depthUnits = int(DEPTH_UNITS * 1_000_000)
    adv.set_depth_table(depth_table)
    print(f"  Disparity shift : {DISPARITY_SHIFT}")
    print(f"  Depth units     : {DEPTH_UNITS} m ({DEPTH_UNITS * 1000:.4f} mm/step)")
 