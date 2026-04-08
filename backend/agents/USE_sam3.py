"""
SAM3 세그멘테이션 클래스 - 텍스트 프롬프트로 객체 자동 추출
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

_SAM3_MODEL_PATH = str(Path(__file__).resolve().parent.parent.parent / "sam3.pt")


def _enhance_contrast(image):
    """CLAHE를 사용한 이미지 대비 향상"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


class SAM3Segmenter:
    """
    SAM3 텍스트 프롬프트 세그멘테이션 클래스

    모델을 한 번만 로드하고, segment()를 반복 호출하여 사용.

    Usage:
        seg = SAM3Segmenter()
        results = seg.segment(image, ["cell", "nucleus"])
        results = seg.segment("/path/to/img.png", "cell")
    """

    def __init__(self, model_path: str = _SAM3_MODEL_PATH, conf_threshold: float = 0.25):
        """
        Args:
            model_path: sam3.pt 경로
            conf_threshold: confidence threshold
        """
        print("[SAM3] 모델 로딩...")
        overrides = dict(
            conf=conf_threshold,
            task="segment",
            mode="predict",
            model=model_path,
            half=True,
            save=False,
            verbose=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self.conf_threshold = conf_threshold
        print("[SAM3] 모델 로딩 완료")

    def segment(self, image, prompts, output_dir: str = "outputs") -> list:
        """
        이미지에서 프롬프트에 맞는 객체를 찾아 반환.

        Args:
            image: numpy array (BGR) 또는 이미지 파일 경로 (str/Path)
            prompts: 텍스트 프롬프트 — 문자열 또는 문자열 배열
            output_dir: 결과 이미지·JSON 저장 디렉터리

        Returns:
            detected_objects (list[dict]): 기존 SAM 형식 JSON 호환
                [{"id", "center_x", "center_y", "center_type",
                  "pixels", "bbox", "prompt"}, ...]
            시각화 이미지와 JSON도 output_dir에 저장됨.
        """
        os.makedirs(output_dir, exist_ok=True)

        # 프롬프트 정규화
        if isinstance(prompts, str):
            prompts = [prompts]

        # 이미지 로드
        if isinstance(image, (str, Path)):
            image_path = str(image)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

        h, w = image.shape[:2]
        print(f"[SAM3] 추론 시작 — 크기: {w}x{h}, 프롬프트: {prompts}")

        self.predictor.set_image(image)
        results = self.predictor(text=prompts)

        detected_objects = []
        obj_id = 0
        vis_image = image.copy()

        for result_idx, result in enumerate(results):
            prompt_label = prompts[result_idx] if result_idx < len(prompts) else "unknown"

            if result.masks is None:
                print(f"[SAM3] '{prompt_label}': 객체 없음")
                continue

            masks = result.masks.data.cpu().numpy()   # [N, H, W]
            num_objects = len(masks)
            print(f"[SAM3] '{prompt_label}': {num_objects}개 발견")

            for i in range(num_objects):
                mask = masks[i]
                ys, xs = np.where(mask > 0.5)

                if len(xs) == 0:
                    continue

                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                center_x = int((x_min + x_max) / 2)
                center_y = int((y_min + y_max) / 2)

                detected_objects.append({
                    "id": obj_id,
                    "center_x": center_x,
                    "center_y": center_y,
                    "center_type": "bbox_center",
                    "pixels": [{"x": int(x), "y": int(y)} for x, y in zip(xs, ys)],
                    "bbox": [x_min, y_min, x_max, y_max],
                    "prompt": prompt_label,
                })

                # 시각화: 마스크 오버레이 + ID 텍스트
                mask_bool = mask > 0.5
                vis_image[mask_bool] = vis_image[mask_bool] * 0.6 + np.array([0, 255, 0]) * 0.4
                cv2.putText(vis_image, f"ID:{obj_id}",
                            (center_x - 20, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                print(f"       객체 {obj_id}: 중심=({center_x},{center_y}), 픽셀={len(xs)}개")
                obj_id += 1

        # 저장
        img_path = os.path.join(output_dir, "sam3_result.png")
        json_path = os.path.join(output_dir, "sam3_data.json")

        cv2.imwrite(img_path, vis_image)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(detected_objects, f, indent=2, ensure_ascii=False)

        print(f"[SAM3] 저장 완료 — 이미지: {img_path}, JSON: {json_path}")
        print(f"[SAM3] 총 {len(detected_objects)}개 객체 감지")

        return detected_objects


def main():
    IMAGE_PATH = r"C:\Users\seoja\Desktop\RamanGPT\RamanGPT\backend\tests\data\path3.png"
    TEXT_PROMPTS = ["cell"]
    OUTPUT_DIR = "./outputs"
    CONF_THRESHOLD = 0.5

    seg = SAM3Segmenter(conf_threshold=CONF_THRESHOLD)
    detected_objects = seg.segment(IMAGE_PATH, TEXT_PROMPTS, output_dir=OUTPUT_DIR)

    if detected_objects:
        obj = detected_objects[0]
        print(f"\n첫 번째 객체: ID={obj['id']}, 중심=({obj['center_x']},{obj['center_y']}), "
              f"픽셀={len(obj['pixels'])}개, 프롬프트='{obj['prompt']}'")

    return detected_objects


if __name__ == "__main__":
    main()
