from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import shutil
import os
import io
from PIL import Image
import warnings
import time
import base64
import requests
from io import BytesIO

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing; lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- CONFIG ----------
# Fix IP cam URL here (use the correct URL from your phone IP Webcam app)
ip_cam_url = "http://192.168.1.21:8080/video"   # <<-- update this

MODEL_PATH = r"best.pt"
OUTPUT_DIR = r"outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
@app.get("/proxy_ipcam")
def proxy_ipcam():
    cap = cv2.VideoCapture("http://192.168.1.21:8080/video")
    def gen():
        while True:
            success, frame = cap.read()
            if not success:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')
# Frame skip to reduce inference load
SKIP_N = 3  # process 1 of every 3 frames
#temp
# @app.post("/detect-nutrient-deficiency")
# Replace the old detect_deficiency endpoint with this one
import base64
from io import BytesIO
from PIL import Image

@app.post("/detect-nutrient-deficiency")
async def detect_deficiency(file: UploadFile = File(...)):
    """Process an uploaded image (frame) and return model output + annotated image (base64)."""
    try:
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # BGR

        if frame is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        # Default fallback prediction variables
        pred_label = "Unknown"
        confidence = 0.0
        annotated = frame.copy()

        # If you have a Keras/Tensorflow model loaded as nutrient_model, run it:
        if nutrient_model is not None:
            try:
                # Convert BGR -> RGB, resize to model expected size (example 224x224)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                target_size = nutrient_model.input_shape[1:3] if hasattr(nutrient_model, "input_shape") else (224,224)
                img_pil = img_pil.resize((target_size[1], target_size[0]))  # PIL size is (width, height)
                x = image.img_to_array(img_pil)
                x = np.expand_dims(x, axis=0)
                x = x / 255.0  # normalization if model expects it

                preds = nutrient_model.predict(x)
                # preds can be logits/probs depending on model; we assume probs
                if preds.ndim == 2:
                    prob_arr = preds[0]
                else:
                    prob_arr = np.array(preds).flatten()

                idx = int(np.argmax(prob_arr))
                confidence = float(np.max(prob_arr))
                # Map index to labels - adjust label list to your model
                labels = nutrient_class_labels if 'nutrient_class_labels' in globals() else ['class0','class1','class2']
                pred_label = labels[idx] if idx < len(labels) else str(idx)

                # Draw prediction text on annotated image
                text = f"{pred_label} ({confidence*100:.1f}%)"
                cv2.putText(annotated, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)

            except Exception as e:
                print("[nutrient model predict error]", e)
                pred_label = "Model error"
                confidence = 0.0

        else:
            # Fallback heuristic: use green-mask to estimate 'healthy' vs 'low green'
            try:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower = np.array([25, 40, 40]); upper = np.array([95, 255, 255])
                mask = cv2.inRange(hsv, lower, upper)
                green_area = int(np.sum(mask > 0))
                total = mask.shape[0] * mask.shape[1]
                green_ratio = green_area / (total + 1e-9)
                if green_ratio > 0.15:
                    pred_label = "Healthy"
                    confidence = min(0.98, 0.5 + green_ratio)  # some heuristic confidence
                else:
                    pred_label = "Possible deficiency"
                    confidence = min(0.9, 0.5 + (0.15 - green_ratio))
                cv2.putText(annotated, f"{pred_label} ({confidence*100:.1f}%)", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
            except Exception as e:
                print("[fallback heuristic error]", e)
                pred_label = "Error"
                confidence = 0.0

        # Optionally draw more overlays (segmentation/detection) if yolo_model exists
        try:
            if yolo_model is not None:
                dets = yolo_model(frame.copy())
                annotated = draw_detection_overlay(annotated, dets)
        except Exception as e:
            print("[yolo overlay error]", e)

        # Encode annotated image to JPEG + base64
        try:
            ret, buf = cv2.imencode('.jpg', annotated)
            if not ret:
                raise RuntimeError("JPEG encode failed")
            b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
            data_uri = f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            print("[encode output image error]", e)
            data_uri = None

        return JSONResponse({
            "prediction": pred_label,
            "confidence": f"{confidence*100:.1f}%",
            "output_image": data_uri
        })

    except Exception as exc:
        print("[detect_deficiency error]", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)
import requests
from fastapi.responses import StreamingResponse
from io import BytesIO

from weather import get_weather_forecast
@app.get("/get_frame")
def get_frame():
    try:
        ipcam_url = "http://192.168.1.21:8080/shot.jpg"  # single frame URL
        resp = requests.get(ipcam_url, timeout=3)
        if resp.status_code == 200:
            return StreamingResponse(BytesIO(resp.content), media_type="image/jpeg")
        else:
            return {"error": f"Camera returned {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# async def detect_deficiency(file: UploadFile = File(...)):
#     contents = await file.read()
#     npimg = np.frombuffer(contents, np.uint8)
#     frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#     # TODO: run your ML model here
#     return JSONResponse({"prediction": "Healthy", "confidence": "98%"})
# px -> cm ratio (keep your original mapping)
px_to_cm_ratio = 0.1



# ---------- LOAD MODELS ----------
# YOLO detection model (your trained weights)
try:
    yolo_model = YOLO(MODEL_PATH)
except Exception as e:
    print("Failed to load YOLO model:", e)
    yolo_model = None

# segmentation model (light seg or fallback to detection)
try:
    segmentation_model = YOLO("yolov8n-seg.pt")
except Exception:
    segmentation_model = None

# lightweight TF models loaded similarly (if available)
try:
    nutrient_model = load_model("nutrient_model.h5")
except Exception:
    nutrient_model = None

try:
    soil_model = load_model("soil_model.h5")
except Exception:
    soil_model = None

try:
    leaf_model = load_model("leaf_classifier_model.h5")
except Exception:
    leaf_model = None

nutrient_class_labels = ['Nitrogen Deficiency', 'Phosphorus Deficiency', 'Potassium Deficiency']
leaf_class_labels = ['dead', 'healthy', 'unhealthy']


@app.post("/check-irrigation")
async def check_irrigation(
    file: UploadFile = File(...),
    crop_type: str = Query("potato"),
    irrigation_method: str = Query("flood")
):
    """Analyze irrigation necessity from image or frame."""
    try:
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        annotated = frame.copy()

        # ------------------ ML / Heuristic Analysis ------------------
        leaf_status = "Unknown"
        soil_status = "Unknown"
        leaf_confidence = 0.0
        water_needed_mm = 0.0

        try:
            # Leaf health model inference (primary)
            if leaf_model is not None:
                img_rgb_leaf = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil_leaf = Image.fromarray(img_rgb_leaf)
                target_size_leaf = leaf_model.input_shape[1:3] if hasattr(leaf_model, "input_shape") else (224, 224)
                img_pil_leaf = img_pil_leaf.resize((target_size_leaf[1], target_size_leaf[0]))
                x_leaf = image.img_to_array(img_pil_leaf)
                x_leaf = np.expand_dims(x_leaf, axis=0)
                x_leaf = x_leaf / 255.0
                preds_leaf = leaf_model.predict(x_leaf)
                if preds_leaf.ndim == 2:
                    prob_leaf = preds_leaf[0]
                else:
                    prob_leaf = np.array(preds_leaf).flatten()
                idx_leaf = int(np.argmax(prob_leaf))
                leaf_confidence = float(np.max(prob_leaf))
                labels_leaf = leaf_class_labels if 'leaf_class_labels' in globals() else ['healthy', 'unhealthy']
                if idx_leaf < len(labels_leaf):
                    leaf_status = labels_leaf[idx_leaf].capitalize()
                else:
                    leaf_status = f"Class {idx_leaf}"
            else:
                # Heuristic fallback if model missing
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower = np.array([25, 40, 40]); upper = np.array([95, 255, 255])
                mask = cv2.inRange(hsv, lower, upper)
                green_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
                leaf_status = "Healthy" if green_ratio > 0.15 else "Stressed"

            # If you have a trained soil_model:
            if soil_model is not None:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                target_size = soil_model.input_shape[1:3] if hasattr(soil_model, "input_shape") else (224,224)
                img_pil = img_pil.resize((target_size[1], target_size[0]))
                x = image.img_to_array(img_pil)
                x = np.expand_dims(x, axis=0)
                x = x / 255.0
                preds = soil_model.predict(x)
                val = float(np.max(preds))
                if val > 0.5:
                    soil_status = "Dry"
                else:
                    soil_status = "Moist"
            else:
                # Simple heuristic fallback: soil dryness based on brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_val = np.mean(gray)
                soil_status = "Dry" if mean_val > 130 else "Moist"

            # Estimate water need based on these conditions
            leaf_is_stressed = leaf_status.lower() not in ["healthy"]
            if soil_status == "Dry" or leaf_is_stressed:
                water_needed_mm = 25.0 if irrigation_method == "flood" else 10.0
            else:
                water_needed_mm = 0.0

            # Annotate image
            leaf_conf_text = f" {leaf_confidence*100:.1f}%" if leaf_confidence > 0 else ""
            info_text = f"{crop_type.capitalize()} | {leaf_status}{leaf_conf_text}, {soil_status}, {water_needed_mm}mm"
            cv2.putText(annotated, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print("[irrigation analysis error]", e)
            leaf_status = "Error"
            soil_status = "Error"
            water_needed_mm = 0.0

        # Encode annotated image to base64
        ret, buf = cv2.imencode('.jpg', annotated)
        if not ret:
            raise RuntimeError("Failed to encode irrigation image")
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        data_uri = f"data:image/jpeg;base64,{b64}"

        return JSONResponse({
            "leaf_status": leaf_status,
            "soil_status": soil_status,
            "crop": crop_type,
            "irrigation_method": irrigation_method,
            "water_needed_mm": round(water_needed_mm, 2),
            "output_image": data_uri
        })

    except Exception as exc:
        print("[check_irrigation error]", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/weather")
async def weather_endpoint(location: str = Query("Chennai,IN")):
    """Return today's and tomorrow's rain probability for the given location."""
    try:
        today, tomorrow = get_weather_forecast(location)
        return {
            "location": location,
            "today_rain_probability": today,
            "tomorrow_rain_probability": tomorrow
        }
    except Exception as exc:
        print("[weather_endpoint error]", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)

# ---------- Processing helpers ----------
def draw_detection_overlay(frame, results):
    """Draw bounding boxes + labels on frame using ultralytics results."""
    try:
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_idx = int(box.cls[0].cpu().numpy())
                label = yolo_model.names[cls_idx] if yolo_model and hasattr(yolo_model, 'names') else str(cls_idx)
                txt = f"{label} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, txt, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    except Exception as e:
        # if something fails, just continue returning the original frame
        print("[draw_detection_overlay] error:", e)
    return frame

def segmentation_plot(frame):
    """Return a segmentation visualization for the frame (fallback to blank if missing)."""
    try:
        if segmentation_model is None:
            # fallback: return a gray placeholder
            h,w = frame.shape[:2]
            placeholder = np.zeros((h,w,3), dtype=np.uint8)
            cv2.putText(placeholder, "Segmentation not loaded", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            return placeholder
        # ultralytics predict accepts numpy images; .predict returns Results object
        res = segmentation_model.predict(source=frame, conf=0.3, save=False)
        seg_img = res[0].plot()  # plot() returns an RGB/BGR image depending on Ultralytics
        # ensure BGR for opencv stacking
        if seg_img is None:
            return np.zeros_like(frame)
        if seg_img.shape[2] == 3:
            return seg_img
        return cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("[segmentation_plot] error:", e)
        return np.zeros_like(frame)

def create_depth_heatmap(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, mask = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    depth = 255 - (blur & mask)
    norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    return heat

def edges_image(frame):
    edges = cv2.Canny(frame, 100, 200)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_bgr

def detect_biological_growth_simple(frame):
    """Simple green-mask growth overlay (like your detect_biological_growth_advanced)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([25,40,40]); upper = np.array([95,255,255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = frame.copy()
    total_area = 0
    for c in contours:
        a = cv2.contourArea(c)
        if a > 150:
            total_area += a
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(out,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.putText(out, f"Green area px: {int(total_area)}", (10, out.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
    return out

# ---------- MJPEG generator that processes live frames ----------
def video_stream_generator():
    cap = cv2.VideoCapture(ip_cam_url)
    if not cap.isOpened():
        print("ERROR: Cannot open IP camera stream. Check ip_cam_url.")
    frame_count = 0
    last_processed_jpeg = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            # if network hiccup, sleep a bit and continue
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (640, 480))
        frame_count += 1

        # Skip frames for performance but still send the last processed frame to keep stream alive
        if frame_count % SKIP_N != 0 and last_processed_jpeg is not None:
            # yield cached JPEG quickly
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + last_processed_jpeg + b'\r\n')
            continue

        # ----- PROCESS FRAME -----
        try:
            # 1) detection overlay
            detection_img = frame.copy()
            if yolo_model is not None:
                try:
                    dets = yolo_model(frame.copy())
                    detection_img = draw_detection_overlay(detection_img, dets)
                except Exception as e:
                    print("[yolo detect error]", e)

            # 2) segmentation visualization
            seg_img = segmentation_plot(frame.copy())

            # 3) biological growth mask
            growth_img = detect_biological_growth_simple(frame.copy())

            # 4) depth heatmap and edges
            depth_img = create_depth_heatmap(frame.copy())
            edges_img = edges_image(frame.copy())

            # Resize all tiles to same size and compose dashboard grid (2 rows x 3 columns)
            standard = (320, 240)
            try:
                d1 = cv2.resize(detection_img, standard)
                d2 = cv2.resize(seg_img, standard)
                d3 = cv2.resize(growth_img, standard)
                d4 = cv2.resize(depth_img, standard)
                d5 = cv2.resize(edges_img, standard)
                # simple placeholder for sixth tile
                d6 = np.zeros((standard[1], standard[0], 3), dtype=np.uint8)
                cv2.putText(d6, "Material / Stats", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            except Exception as e:
                print("[resize compose error]", e)
                # fallback single frame
                dashboard = frame
            else:
                top = np.hstack((d1, d2, d3))
                bottom = np.hstack((d4, d5, d6))
                dashboard = np.vstack((top, bottom))

            # Encode dashboard as JPEG
            ret2, jpeg = cv2.imencode('.jpg', dashboard)
            if not ret2:
                continue
            jpg_bytes = jpeg.tobytes()
            last_processed_jpeg = jpg_bytes

            # yield multipart MJPEG chunk
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')

        except GeneratorExit:
            break
        except Exception as e:
            print("[stream processing error]", e)
            time.sleep(0.05)
            continue

    cap.release()

@app.get("/video_feed")
async def video_feed():
    """MJPEG stream of processed live dashboard frames."""
    return StreamingResponse(video_stream_generator(),
                             media_type='multipart/x-mixed-replace; boundary=frame')

# ---------------------- existing REST endpoints (images) ----------------------
# [keep your existing endpoints: detect-nutrient-deficiency, count-crops, check-irrigation, etc.]
# (I didn't repeat them here to keep the snippet focused on live-stream integration.)
# Make sure the endpoints you provided earlier remain below this point in the file.

@app.get("/")
async def root():
    return {"message": "Smart Agriculture API is running!"}
