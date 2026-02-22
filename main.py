from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pillow_heif
import base64
import io
import os
from dotenv import load_dotenv
from pathlib import Path
import uuid
from datetime import datetime
import httpx
import asyncio

load_dotenv()
pillow_heif.register_heif_opener()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found")

app = FastAPI(title="Toon Avatar Generator API", version="FINAL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ORIGINAL_IMAGES = Path("original_images")
CARTOON_IMAGES = Path("cartoon_images")
ORIGINAL_IMAGES.mkdir(exist_ok=True)
CARTOON_IMAGES.mkdir(exist_ok=True)

MAX_IMAGE_SIZE = 5 * 1024 * 1024

TOON_PROMPT = """
Create a flat vector-style cartoon avatar inspired by ToonApp,
while STRICTLY preserving the exact facial identity of the provided reference image.
"""

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/generate-cartoon")
async def generate_cartoon(image: UploadFile = File(...)):
    if image.content_type not in {
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/heic",
        "image/heif"
    }:
        raise HTTPException(400, "Invalid image type")

    image_bytes = await image.read()
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(400, "Image too large")

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)

    image_id = f"{datetime.utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/images/edits",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"image": ("image.png", buf, "image/png")},
            data={
                "model": "gpt-image-1",
                "prompt": TOON_PROMPT,
                "size": "1024x1024"
            }
        )

    if response.status_code != 200:
        raise HTTPException(500, "OpenAI image generation failed")

    result = response.json()
    cartoon_bytes = base64.b64decode(result["data"][0]["b64_json"])

    return StreamingResponse(
        io.BytesIO(cartoon_bytes),
        media_type="image/png",
        headers={"X-Image-Id": image_id}
    )

# =========================
# HEAD SEGMENTATION (BUG-FIXED ONLY)
# =========================

from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
from dataclasses import dataclass
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from PIL import ImageFilter

# ---------- MediaPipe safe lazy loader (Python 3.13 fix)

_mp_face = None

def get_face_detector():
    global _mp_face
    if _mp_face is None:
        import mediapipe as mp
        _mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.4
        )
    return _mp_face

# ---------- Model (UNCHANGED)

@dataclass
class HeadSegmentationModelOutput(BaseOutput):
    sample: torch.Tensor

class SimpleResNetEncoder(nn.Module):
    def __init__(self, out_channels, depth=5, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self._in_channels = 64
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(self._in_channels, out_channels, stride, downsample)]
        self._in_channels = out_channels
        layers.extend(BasicBlock(out_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        feats = [x]
        x = self.relu(self.bn1(self.conv1(x)))
        feats.append(x)
        x = self.maxpool(x)
        x = self.layer1(x); feats.append(x)
        x = self.layer2(x); feats.append(x)
        x = self.layer3(x); feats.append(x)
        x = self.layer4(x); feats.append(x)
        return feats

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class HeadSegmentationModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, input_resolution=512):
        super().__init__()
        self.encoder = SimpleResNetEncoder((3,64,64,128,256,512))
        self.decoder = DecoderBlock(512, 256, 256)
        self.head = nn.Conv2d(256, 2, 1)

    def forward(self, x):
        feats = self.encoder(x)
        x = self.decoder(feats[-1], feats[-2])
        return HeadSegmentationModelOutput(sample=self.head(x))

# ---------- Pipeline singleton (reload/meta fix)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_pipe = None

def get_pipe():
    global _pipe
    if _pipe is None:
        model = HeadSegmentationModel.from_pretrained("okaris/head-segmentation")
        model.eval()
        _pipe = model
    return _pipe

# ---------- Helpers (UNCHANGED LOGIC)

def detect_face_box(img):
    rgb = np.array(img)
    r = get_face_detector().process(rgb)
    if not r.detections:
        return None
    d = r.detections[0].location_data.relative_bounding_box
    W, H = img.size
    return int(d.xmin*W), int(d.ymin*H), int(d.width*W), int(d.height*H)

def smooth(img):
    a = img.split()[3]
    img.putalpha(a.filter(ImageFilter.GaussianBlur(3)))
    return img

def pad(img):
    w, h = img.size
    p = int(w * 0.15)
    canvas = Image.new("RGBA", (w + 2*p, h), (0,0,0,0))
    canvas.paste(img, (p, 0), img)
    return canvas

# ---------- API

class Req(BaseModel):
    image_url: str

@app.post("/extract-face")
async def extract_face(req: Req):
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(req.image_url)

    if r.status_code != 200:
        raise HTTPException(400, "Image download failed")

    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    box = detect_face_box(img)

    if box:
        x,y,w,h = box
        cx,cy = x+w//2, y+h//2
        s = int(max(w,h)*1.8)
        img = img.crop((max(0,cx-s), max(0,cy-s), cx+s, cy+s))

    arr = np.array(img.resize((512,512))) / 255.0
    t = torch.tensor(arr).permute(2,0,1).unsqueeze(0).float()
    with torch.no_grad():
        mask = get_pipe()(t).sample.argmax(1).squeeze().numpy()

    rgba = img.convert("RGBA")
    np_img = np.array(rgba)
    np_img[...,3] = (mask > 0) * 255

    out = pad(smooth(Image.fromarray(np_img)))

    buf = io.BytesIO()
    out.save(buf, "PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
