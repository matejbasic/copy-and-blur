import os

import cv2
import gdown
import numpy as np
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector

from src import settings

detector, age_gender_model = None, None


def blur_faces_and_save(src_file: str, dst_file: str, device: str, max_age: int = None, verbose: bool = False) -> None:
    global detector, age_gender_model
    if not detector:
        detector = Detector(settings.detector_weights, device, verbose=verbose)
    if not age_gender_model:
        age_gender_model = MiVOLO(
            settings.checkpoint,
            device,
            half=True,
            use_persons=True,
            disable_faces=False,
            verbose=verbose,
        )

    img = cv2.imread(src_file)
    detected_objects = detector.predict(img)
    age_gender_model.predict(img, detected_objects)

    for i, result in enumerate(detected_objects.yolo_results):
        if i not in detected_objects.face_to_person_map:
            continue
        if max_age:
            if detected_objects.ages[i] > max_age:
                if detected_objects.face_to_person_map[i]:
                    if detected_objects.face_to_person_map[i] > max_age:
                        continue
                else:
                    continue

        x1, y1, x2, y2 = result.boxes.xyxy[0].tolist()
        x1 = int(x1 * 0.97)
        y1 = int(y1 * 0.97)
        x2 = int(img.shape[1] if x2 * 1.03 > img.shape[1] else x2 * 1.03)
        y2 = int(img.shape[0] if y2 * 1.03 > img.shape[0] else y2 * 1.03)
        w, h = x2 - x1, y2 - y1
        radius = min(w, h) // 2

        # Create a circular mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w // 2, h // 2), radius, 255, thickness=-1)

        face_roi = img[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 15)
        face_with_round_blur = np.where(mask[:, :, None], blurred_face, face_roi)
        img[y1:y2, x1:x2] = face_with_round_blur

    cv2.imwrite(dst_file, img)


def download_models_if_missing() -> None:
    if not os.path.isdir(settings.models_dir_path):
        os.mkdir(settings.models_dir_path)

    if not os.path.exists(settings.detector_weights):
        gdown.download(settings.detector_weights_url, settings.detector_weights, quiet=False)

    if not os.path.exists(settings.checkpoint):
        gdown.download(settings.checkpoint_url, settings.checkpoint, quiet=False)
