# Room Capture (DA3NESTED-GIANT-LARGE)

Reconstruct a full room in metric 3D from ~6 iPhone 16 Pro main-lens photos using
[Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3)
(`DA3NESTED-GIANT-LARGE`).

## Setup

Install the base DA3 package (from the sibling checkout) first, then the extras
for this pipeline:

```powershell
# From d:\My Projects\PAVE\Paveception
pip install -e .\depth-anything-3            # DA3 core
pip install --no-build-isolation `
    git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
pip install -r .\room_capture\requirements.txt
```

GPU requirement: ~10-14 GB VRAM for 6 images at `process_res=504`. If you
out-of-memory, pass `--no-gs` (skips Gaussian Splatting export) or switch to
`DA3NESTED-GIANT-LARGE-1.1` (retrained, same footprint, better for many scenes).

## HuggingFace authentication

`DA3NESTED-GIANT-LARGE` is hosted on HuggingFace under a CC BY-NC 4.0 license.
You need an HF token to download it reliably (anonymous downloads get
rate-limited; gated models fail outright).

1. Create a token at <https://huggingface.co/settings/tokens>. A "fine-grained"
   token with **Read access to public gated repos** is sufficient.
2. Make the token available to the pipeline in one of these ways (first match
   wins, from highest to lowest priority):

   ```powershell
   # Option A - .env file next to capture_room.py (recommended, stays local)
   cp .\room_capture\.env.example .\room_capture\.env
   # Then edit .env and replace the placeholder with your token.

   # Option B - environment variable (per-session)
   $env:HF_TOKEN = "hf_xxx..."

   # Option C - HuggingFace CLI login (machine-wide, cached in ~/.cache/huggingface)
   pip install huggingface-hub
   huggingface-cli login

   # Option D - CLI flag (leaks into shell history, use sparingly)
   python -m room_capture.capture_room --hf-token hf_xxx...
   ```

The script prints a line like `[hf_auth] Authenticated as <you> via env:HF_TOKEN`
on startup so you can confirm the token was picked up.

## Usage

1. Drop 6 iPhone 16 Pro photos (HEIC/JPG/PNG) into `room_capture/input/`.
   Capture tip: stand near the center of the room, rotate ~60 degrees between
   shots, and keep meaningful overlap + parallax (pure rotation hurts pose).
2. Run:

   ```powershell
   cd "d:\My Projects\PAVE\Paveception"
   python -m room_capture.capture_room
   ```

   Useful flags:
   - `--input room_capture/input` / `--output room_capture/output`
   - `--model depth-anything/DA3NESTED-GIANT-LARGE`
   - `--hf-token hf_xxx` - explicit token (last-resort; prefer `.env` or env var)
   - `--lens {auto,main,ultrawide}` - iPhone 16 Pro lens profile.
     `main` = 1x (24mm-equiv, HFoV ~73.7 deg), `ultrawide` = 0.5x
     (13mm-equiv, HFoV ~108.4 deg). `auto` (default) trusts EXIF and falls back
     to `main` if tags are missing. Use `ultrawide` for tight rooms; the extra
     FoV captures more wall area per shot but has more barrel distortion (DA3
     still assumes a pure pinhole model, so pose can be a touch noisier).
   - `--no-gs` to skip gs_ply/gs_video
   - `--rrd PATH` to write a Rerun recording file instead of spawning the viewer
     (rerun allows only one sink at a time, so viewer and file are mutually
     exclusive - pick one). Open the file later with `rerun PATH`.
   - `--no-rerun` for fully headless mode (no viewer, no .rrd, just GLB/npz/GS)
   - `--process-res 504`

3. Artifacts land in `room_capture/output/`:
   - `*.rrd` - Rerun recording (only when `--rrd` is passed; open with `rerun <file>.rrd`)
   - `*.glb` - point cloud + camera wireframes
   - `mini.npz` - depth, conf, extrinsics, intrinsics
   - `*_3dgs.ply` - 3D Gaussian Splat (open in
     [SuperSplat](https://superspl.at/editor))
   - `*_3dgs.mp4` - rasterized Gaussian trajectory video

## Notes

- Output depth is already in **meters** (DA3NESTED). No rescaling needed.
- Intrinsics are read from EXIF when available. Fallback profiles:
  - `main` (1x): 73.74 deg HFoV (24mm-equivalent)
  - `ultrawide` (0.5x): 108.41 deg HFoV (13mm-equivalent)
- For mixed captures (some 1x, some 0.5x), leave `--lens auto` and the EXIF
  auto-classifier will pick the right profile per photo.
- The script tries to pass known intrinsics to DA3. If the DA3 API rejects
  `intrinsics` without `extrinsics`, it falls back to full pose estimation and
  logs a diagnostic comparison to Rerun so you can eyeball focal drift.
