import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_CATALOG = {
    "lite0": {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
        ),
        "filename": "efficientdet_lite0.tflite",
    },
    "lite2": {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite"
        ),
        "filename": "efficientdet_lite2.tflite",
    },
}


def ensure_model_downloaded(model_name: str) -> Path:
    model_data = MODEL_CATALOG[model_name]
    model_path = Path(__file__).with_name(model_data["filename"])
    if model_path.exists():
        return model_path

    print(f"Descargando modelo '{model_name}' (solo primera vez)...")
    urllib.request.urlretrieve(model_data["url"], model_path)
    print(f"Modelo guardado en: {model_path}")
    return model_path


def parse_source(source_arg: str):
    source_arg = source_arg.strip()
    if source_arg.isdigit():
        return int(source_arg)
    return source_arg


def open_capture(source):
    attempts = []

    if isinstance(source, int):
        candidates = [
            (source, None, f"indice {source} (CAP_ANY)"),
            (source, cv2.CAP_MSMF, f"indice {source} (CAP_MSMF)"),
            (source, cv2.CAP_DSHOW, f"indice {source} (CAP_DSHOW)"),
        ]
    else:
        candidates = [(source, None, f"fuente '{source}'")]

    for src, backend, label in candidates:
        if backend is None:
            cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src, backend)

        if not cap.isOpened():
            attempts.append(f"{label}: no abre")
            cap.release()
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            attempts.append(f"{label}: abre pero no entrega frames")
            cap.release()
            continue

        attempts.append(f"{label}: OK")
        return cap, attempts

    return None, attempts


def create_detector(
    model_path: Path,
    running_mode: vision.RunningMode,
    score_threshold: float,
    max_results: int,
):
    options = vision.ObjectDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        score_threshold=score_threshold,
        max_results=max_results,
    )
    return vision.ObjectDetector.create_from_options(options)


def draw_detection(frame, detection):
    h, w = frame.shape[:2]
    bbox = detection.bounding_box

    x = max(0, int(bbox.origin_x))
    y = max(0, int(bbox.origin_y))
    bw = int(bbox.width)
    bh = int(bbox.height)

    x2 = min(w - 1, x + bw)
    y2 = min(h - 1, y + bh)

    cv2.rectangle(frame, (x, y), (x2, y2), (0, 180, 255), 2)

    label = "objeto"
    score = 0.0
    if detection.categories:
        category = detection.categories[0]
        label = category.category_name or "objeto"
        score = category.score

    text = f"{label} ({score:.2f})"
    cv2.putText(
        frame,
        text,
        (x + 4, max(18, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        frame,
        text,
        (x + 4, max(18, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )


def run_image_mode(
    image_path: Path,
    output_path: Path | None,
    score_threshold: float,
    max_results: int,
    model_path: Path,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

    detector = create_detector(model_path, vision.RunningMode.IMAGE, score_threshold, max_results)
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        detections = result.detections if result and result.detections else []
        for det in detections:
            draw_detection(image, det)

        cv2.putText(
            image,
            f"Detecciones: {len(detections)}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 30, 30),
            2,
        )

        if output_path is None:
            output_path = image_path.with_name(f"{image_path.stem}_detectado{image_path.suffix}")

        cv2.imwrite(str(output_path), image)
        print(f"Resultado guardado en: {output_path}")
        print(f"Detecciones encontradas: {len(detections)}")
    finally:
        detector.close()


def run_video_mode(source, score_threshold: float, max_results: int, model_path: Path):
    detector = create_detector(model_path, vision.RunningMode.VIDEO, score_threshold, max_results)
    cap, attempts = open_capture(source)

    if cap is None:
        detector.close()
        print("No se pudo abrir la camara/fuente de video.")
        if attempts:
            print("Intentos realizados:")
            for item in attempts:
                print(f"  - {item}")
        print("Sugerencias:")
        print("  1) Cerrar apps que usen camara (Teams, Zoom, navegador).")
        print("  2) En Windows: Configuracion > Privacidad > Camara > habilitar acceso.")
        print("  3) Probar otro indice: --source 1 o --source 2")
        print("  4) Probar con video: --source .\\video.mp4")
        return

    print("Detector de objetos listo")
    print(f"Fuente: {source}")
    print("Controles: ESC o Q para salir")

    try:
        timestamp_ms = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += 33

            result = detector.detect_for_video(mp_image, timestamp_ms)
            detections = result.detections if result and result.detections else []

            for det in detections:
                draw_detection(frame, det)

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (245, 245, 245), -1)
            cv2.putText(
                frame,
                f"Object Detector | detecciones: {len(detections)}",
                (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (30, 30, 30),
                2,
            )

            cv2.imshow("Clasificador de Objetos - MediaPipe", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Clasificador/Detector de objetos con MediaPipe Tasks")
    parser.add_argument(
        "--source",
        default="0",
        help="Indice de camara (0,1,2...) o ruta/URL de video. Default: 0",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Ruta de imagen para detectar objetos en modo imagen",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta de salida para modo imagen (opcional)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.35,
        help="Umbral minimo de confianza (0 a 1)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Cantidad maxima de detecciones por frame",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CATALOG.keys()),
        default="lite2",
        help="Modelo: lite2 (mas preciso) o lite0 (mas rapido)",
    )
    args = parser.parse_args()

    model_path = ensure_model_downloaded(args.model)

    if args.image:
        image_path = Path(args.image)
        output_path = Path(args.output) if args.output else None
        run_image_mode(
            image_path,
            output_path,
            args.score_threshold,
            args.max_results,
            model_path,
        )
        return

    source = parse_source(args.source)
    run_video_mode(source, args.score_threshold, args.max_results, model_path)


if __name__ == "__main__":
    main()
