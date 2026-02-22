# Toon Avatar Generator API 

A high-performance FastAPI-based service that transforms real-life portraits into flat vector-style cartoon avatars while strictly preserving facial identity. It features advanced head segmentation and background removal for clean, ready-to-use assets.

## Key Features

*   **Cartoon Transformation:** Converts photos into ToonApp-inspired vector avatars using OpenAI's image models.
*   **Identity Preservation:** Specifically tuned to maintain the exact facial features of the reference subject.
*   **Head Segmentation:** Automatically detects and extracts the head from images using a custom ResNet-based segmentation model.
*   **Smart Pre-processing:** Includes automatic face detection (via MediaPipe), cropping, smoothing, and padding.
*   **Multi-format Support:** Handles JPEG, PNG, and HEIC/HEIF formats.

## Getting Started

### Prerequisites

*   Python 3.10+
*   OpenAI API Key
*   CUDA-capable GPU (optional, for faster segmentation)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/toon-avatar-api.git
    cd toon-avatar-api
    ```

2.  **Install dependencies:**
    ```bash
    pip install fastapi uvicorn pillow pillow-heif httpx torch torchvision diffusers mediapipe python-dotenv
    ```

3.  **Configure Environment:**
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=your_actual_api_key_here
    ```

4.  **Run the Server:**
    ```bash
    uvicorn main:app --reload
    ```

## 🛠 API Endpoints

### 1. Generate Cartoon Avatar
Transforms an uploaded image into a cartoon version.

*   **Endpoint:** `POST /generate-cartoon`
*   **Content-Type:** `multipart/form-data`
*   **Parameters:** `image` (File)
*   **Returns:** A PNG image stream of the cartoonized avatar.

### 2. Extract Face & Segment
Detects the face, crops the image, and removes the background.

*   **Endpoint:** `POST /extract-face`
*   **Content-Type:** `application/json`
*   **Body:** `{"image_url": "https://example.com/photo.jpg"}`
*   **Returns:** A transparent PNG image stream of the segmented head.

### 3. Health Check
*   **Endpoint:** `GET /health`
*   **Returns:** `{"status": "ok"}`

## Technical Architecture

*   **Framework:** FastAPI for high-concurrency asynchronous processing.
*   **Image Processing:** PIL (Pillow) with HEIF support.
*   **Face Detection:** MediaPipe Face Detection (Short-range model).
*   **Segmentation Model:** Custom `HeadSegmentationModel` based on ResNet-34 architecture, hosted on Hugging Face (`okaris/head-segmentation`).
*   **AI Generation:** OpenAI DALL-E/Image Edit API integration.

## Notes

*   The application automatically creates `original_images` and `cartoon_images` directories for storage.
*   Maximum upload size is capped at **5MB**.
*   The segmentation pipeline uses a singleton pattern to ensure models are only loaded into memory once.
