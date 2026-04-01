import io
import math
import time
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision.utils import make_grid


# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Anime + Pokemon GAN Studio",
    page_icon="🎴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------
# Theme / Styling
# ------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
  --bg: #0a0a0f;
  --panel: #11111a;
  --panel-2: #161625;
  --text: #f5f7ff;
  --muted: #a2a7c4;
  --cyan: #20e3ff;
  --pink: #ff49b6;
  --purple: #8b5cf6;
  --line: rgba(255,255,255,0.09);
}

html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
  background:
    radial-gradient(1200px 600px at -10% -20%, rgba(139,92,246,0.2), transparent 55%),
    radial-gradient(1000px 500px at 110% 0%, rgba(255,73,182,0.18), transparent 50%),
    radial-gradient(1200px 700px at 50% 120%, rgba(32,227,255,0.12), transparent 45%),
    var(--bg);
  color: var(--text);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0c0c14, #121225) !important;
  border-right: 1px solid var(--line);
}

h1, h2, h3 {
  font-family: 'Orbitron', sans-serif !important;
  letter-spacing: 0.5px;
}

.kicker {
  color: var(--muted);
  font-size: 0.95rem;
  margin-top: -0.6rem;
}

.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 14px 16px;
  backdrop-filter: blur(8px);
}

.stat {
  background: linear-gradient(160deg, rgba(139,92,246,0.2), rgba(32,227,255,0.09));
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 14px;
  padding: 10px 14px;
}

.stButton > button {
  border: 0 !important;
  border-radius: 12px !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  background: linear-gradient(90deg, var(--purple), var(--pink)) !important;
  box-shadow: 0 6px 24px rgba(139,92,246,0.34);
}

.stDownloadButton > button {
  border: 0 !important;
  border-radius: 12px !important;
  color: #041018 !important;
  font-weight: 700 !important;
  background: linear-gradient(90deg, var(--cyan), #84ffe6) !important;
}

[data-testid="stSlider"] [data-baseweb="thumb"] {
  background: var(--pink) !important;
}

small, .muted {
  color: var(--muted) !important;
}

@media (max-width: 768px) {
  .card, .stat {
    border-radius: 12px;
    padding: 10px 12px;
  }
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------
# GAN Generator Architecture
# ------------------------------
class BaseGenerator(nn.Module):
    def __init__(self, z_dim=100, channels=3, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.ConvTranspose2d(features, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class DCGANGenerator(BaseGenerator):
    pass


class WGANGenerator(BaseGenerator):
    pass


# ------------------------------
# Paths / Constants
# ------------------------------
ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 64
CHANNELS = 3
Z_DIM = 100


def resolve_checkpoint(preferred_names):
    for name in preferred_names:
        p = ROOT / name
        if p.exists():
            return p
    return None


DCGAN_CKPT = resolve_checkpoint([
    "dcgan_epoch_35.pt",
    "dcgan_latest.pt",
    "dcgan_epoch_30.pt",
])

WGANGP_CKPT = resolve_checkpoint([
    "wgan_epoch_60.pt",
    "wgan_gp_latest.pt",
    "wgangp_epoch_60.pt",
    "wgangp_latest.pt",
])


@st.cache_resource(show_spinner=False)
def load_generator(model_type):
    if model_type == "DCGAN":
        ckpt = DCGAN_CKPT
        model = DCGANGenerator(z_dim=Z_DIM, channels=CHANNELS, features=64)
    else:
        ckpt = WGANGP_CKPT
        model = WGANGenerator(z_dim=Z_DIM, channels=CHANNELS, features=64)

    if ckpt is None:
        raise FileNotFoundError(f"Checkpoint not found for {model_type}.")

    state = torch.load(ckpt, map_location="cpu")

    if isinstance(state, dict):
        if "model_g" in state:
            state = state["model_g"]
        elif "generator" in state:
            state = state["generator"]
        elif "state_dict" in state:
            state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)
    return model, ckpt.name


@torch.no_grad()
def generate_batch(model, n_images, seed):
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    z = torch.randn(n_images, Z_DIM, 1, 1, device=DEVICE)
    out = model(z)
    out = (out + 1.0) / 2.0
    out = out.clamp(0.0, 1.0)
    return out.cpu()


def to_pil_grid(batch, n_images):
    cols = int(math.ceil(math.sqrt(n_images)))
    grid = make_grid(batch, nrow=cols, padding=2)
    arr = grid.permute(1, 2, 0).numpy()
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def batch_to_zip_bytes(batch, prefix):
    mem = io.BytesIO()
    with ZipFile(mem, "w", ZIP_DEFLATED) as zf:
        for i in range(batch.size(0)):
            img = batch[i].permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            pil = Image.fromarray(img)
            one = io.BytesIO()
            pil.save(one, format="PNG")
            zf.writestr(f"{prefix}_{i+1:02d}.png", one.getvalue())
    mem.seek(0)
    return mem


def render_model_info():
    st.markdown("### Model Information")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### DCGAN")
        st.markdown("- Generator: ConvTranspose2d stack (100 → 512 → 256 → 128 → 64 → 3)")
        st.markdown("- Activations: ReLU + Tanh output")
        st.markdown("- Output: 64x64 RGB")
        st.markdown("- Domain: Anime faces + Pokemon sprites")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### WGAN-GP")
        st.markdown("- Generator: ConvTranspose2d stack (100 → 512 → 256 → 128 → 64 → 3)")
        st.markdown("- Activations: ReLU + Tanh output")
        st.markdown("- Output: 64x64 RGB")
        st.markdown("- Domain: High-quality anime faces")
        st.markdown("</div>", unsafe_allow_html=True)

    loss_candidates = sorted(ROOT.glob("*loss*.*"))
    if loss_candidates:
        st.markdown("### Training/Loss Artifacts")
        cols = st.columns(min(3, len(loss_candidates)))
        for i, f in enumerate(loss_candidates[:3]):
            with cols[i]:
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    st.image(str(f), caption=f.name, use_container_width=True)
                else:
                    st.write(f.name)
    else:
        st.caption("No explicit loss graph file found in the app directory.")


# ------------------------------
# Header
# ------------------------------
st.markdown("# Anime + Pokemon GAN Studio")
st.markdown(
    "<p class='kicker'>Generate stylized characters using dual GAN generators in a modern comparison workflow.</p>",
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("<div class='stat'><b>Image Output</b><br><span class='muted'>64x64 RGB</span></div>", unsafe_allow_html=True)
with m2:
    st.markdown("<div class='stat'><b>Latent Space</b><br><span class='muted'>100-D noise vector</span></div>", unsafe_allow_html=True)
with m3:
    st.markdown(f"<div class='stat'><b>Runtime Device</b><br><span class='muted'>{str(DEVICE).upper()}</span></div>", unsafe_allow_html=True)


# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.markdown("## Controls")
mode = st.sidebar.radio("Mode", ["Single Model", "Comparison"], horizontal=False)
model_choice = st.sidebar.selectbox("Model", ["DCGAN", "WGAN-GP"], index=0, disabled=(mode == "Comparison"))
n_images = st.sidebar.slider("Number of Images", min_value=1, max_value=64, value=16, step=1)
seed = st.sidebar.number_input("Seed (same seed = same output)", min_value=0, max_value=999999, value=42, step=1)
seed_help_text = "Seed info: same seed = same output, different seed = new variation"
print(seed_help_text)
st.sidebar.write(seed_help_text)
preview_width = st.sidebar.slider("Preview Size", min_value=320, max_value=1200, value=760, step=20)

if mode == "Comparison":
    st.sidebar.caption("Comparison mode will generate from both models at once.")

generate_btn = st.sidebar.button("Generate", use_container_width=True)


# ------------------------------
# Main Render
# ------------------------------
render_model_info()

if generate_btn:
    try:
        start = time.perf_counter()

        with st.spinner("Generating images..."):
            if mode == "Single Model":
                model, _ = load_generator(model_choice)
                batch = generate_batch(model, n_images, seed)
                grid = to_pil_grid(batch, n_images)
                duration = time.perf_counter() - start

                st.markdown("### Generated Output")
                st.caption(f"Model: {model_choice} | Time: {duration:.2f}s")
                st.image(grid, width=preview_width)

                png_buf = io.BytesIO()
                grid.save(png_buf, format="PNG")
                png_buf.seek(0)
                st.download_button(
                    label="Download Grid (PNG)",
                    data=png_buf,
                    file_name=f"{model_choice.lower().replace('-', '').replace(' ', '_')}_grid.png",
                    mime="image/png",
                    use_container_width=True,
                )

                zip_buf = batch_to_zip_bytes(batch, model_choice.lower().replace("-", "").replace(" ", "_"))
                st.download_button(
                    label="Download All Images (ZIP)",
                    data=zip_buf,
                    file_name=f"{model_choice.lower().replace('-', '').replace(' ', '_')}_images.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

            else:
                dcgan_model, _ = load_generator("DCGAN")
                wgan_model, _ = load_generator("WGAN-GP")

                batch_dc = generate_batch(dcgan_model, n_images, seed)
                batch_wg = generate_batch(wgan_model, n_images, seed)
                grid_dc = to_pil_grid(batch_dc, n_images)
                grid_wg = to_pil_grid(batch_wg, n_images)
                duration = time.perf_counter() - start

                st.markdown("### Side-by-Side Comparison")
                st.caption(f"Time: {duration:.2f}s")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### DCGAN")
                    st.image(grid_dc, width=min(preview_width, 560))
                    b1 = io.BytesIO()
                    grid_dc.save(b1, format="PNG")
                    b1.seek(0)
                    st.download_button(
                        label="Download DCGAN Grid",
                        data=b1,
                        file_name="dcgan_grid.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                with c2:
                    st.markdown("#### WGAN-GP")
                    st.image(grid_wg, width=min(preview_width, 560))
                    b2 = io.BytesIO()
                    grid_wg.save(b2, format="PNG")
                    b2.seek(0)
                    st.download_button(
                        label="Download WGAN-GP Grid",
                        data=b2,
                        file_name="wgangp_grid.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                all_zip = io.BytesIO()
                with ZipFile(all_zip, "w", ZIP_DEFLATED) as zf:
                    one = io.BytesIO()
                    grid_dc.save(one, format="PNG")
                    zf.writestr("dcgan_grid.png", one.getvalue())
                    two = io.BytesIO()
                    grid_wg.save(two, format="PNG")
                    zf.writestr("wgangp_grid.png", two.getvalue())
                all_zip.seek(0)
                st.download_button(
                    label="Download Comparison Bundle (ZIP)",
                    data=all_zip,
                    file_name="gan_comparison_bundle.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

    except Exception as e:
        st.error(f"Generation failed: {e}")
else:
    st.info("Use the sidebar controls and click Generate.")
