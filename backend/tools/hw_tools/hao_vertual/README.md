# hao_vertual — 가상 장비 계층

장비 없는 PC 에서 도구 계층(`hw_tools/`)을 **실제로 실행**하기 위한 대역 4종.

## 켜기

```powershell
$env:RAMAN_VIRTUAL_HW = "1"
$env:RAMAN_VIRTUAL_SCENE = "default"          # data/virtual_stage/<이름>/
python -m backend.web_controller.main
```

기동 로그 한 줄에 `virtual:default` 이 찍힌다. 실물이면 `real-hw`. 설정은 전부
`backend/llm_config.py` 하나에서 읽는다(`VIRTUAL_HW`, `VIRTUAL_SCENE`, `VIRTUAL_SCENE_ROOT`).

## 씬 폴더

```
data/virtual_stage/<이름>/
  map.png       스테이지 전체 이미지. 이 한 장이 곧 시료다.
  scene.json    축척·앵커·흐림
```

`scene.json`

| 키 | 필수 | 뜻 |
|---|---|---|
| `um_per_px` | **예** | 이미지 1픽셀이 몇 µm 인가. 없으면 1.0 으로 뜨고 경고가 찍히며, 시야 배율이 실제와 맞지 않는다 |
| `image` | 아니오 | 쓸 이미지 파일명. 생략하면 폴더에서 첫 이미지 |
| `center_stage_mm` | 아니오 | 이미지 중심 픽셀이 대응하는 스테이지 좌표 `[x, y]`. 생략하면 `Config.ini` 의 `STAGE_CENTER_X/Y` |
| `psf_sigma_px` | 아니오 | 대물렌즈 흐림 흉내(가우시안 σ). 기본 0 = 꺼짐 |

이미지가 덮는 스테이지 범위는 `이미지 크기 × um_per_px` 다. 그 밖으로 나가면 화면이
검게 나온다 — 타일링이나 가장자리 늘리기를 하지 않는 이유는, 그러면 스테이지 한계 밖에서도
시료가 계속 보여 범위를 벗어난 이동이 정상처럼 보이기 때문이다.

## 좌표 규약

- 이미지 중심 픽셀 ↔ `center_stage_mm` (기본은 config 의 스테이지 중점)
- 축 부호는 `service/vision/optics_map.py` 의 `SIGN_X=-1`, `SIGN_Y=+1` 을 그대로 쓴다.
  즉 이미지 열이 커지면 스테이지 X 는 작아진다.
- 시야 크기는 `optics_map.fov_mm()` 이 정한다(카메라 렌즈 축척). 이미지 축척(`um_per_px`)과
  **다른 값**이다 — 하나는 지도의 축척, 하나는 카메라의 축척이다.

이 규약 덕분에 `move_to_pixel(px, py)` 왕복이 자동으로 맞는다. 화면 우하단을 찍으면
스테이지가 `dx = -FOV_w/2`, `dy = +FOV_h/2` 만큼 움직인다.

## 한계

| 항목 | 상태 |
|---|---|
| Z 축 | **없다.** 시료가 2D 라 존재하지 않는다. `run_autofocus` 는 `has_z=False` 를 보고 거절한다. `run_grid_scan` 은 `autofocus='none'` 으로만 돈다 |
| 스펙트럼 내용 | **아직 없다.** `virtual_ccd._synthesize()` 가 평탄한 신호를 만든다 — 아래 참고 |
| `reconnect_hardware` | 돈다. 다만 실제로 해제할 자원이 없어 항상 성공한다 |
| `set_camera_auto_exposure` | 소프트웨어 보정(목표 평균 밝기 118). 실물은 센서 기능 |
| 온도 | 20 °C/s 로 목표에 수렴한다(`VirtualCCD.TEMP_RATE_C_PER_S`). 실물은 수 분 |
| 촬영 시간 | 실제 노출만큼 기다린다(`VirtualCCD.TIME_SCALE`). 0 으로 두면 즉시 |

## 다음 단계 — 색 → 피크

`virtual_ccd.py` 의 `_synthesize(axis_cm1, color_bgr, ctx)` 하나만 채우면 된다.

- `axis_cm1` — 픽셀별 라만 시프트. 피크를 cm⁻¹ 로 적을 수 있게 넘어온다
- `color_bgr` — 지금 레이저가 놓인 자리의 시료 색(`scene.color_at`). **이것이 물질 식별자다**
- `ctx` — `exposure_s`, `power_pct`, `num_acc`, `laser_armed`

배경(형광)을 넣을 때 제약이 하나 있다: **다항식으로 근사되는 모양이어야 한다.** 도구 계층의
배경 제거가 `service/analyse/spectro_math.py` 의 `ipbsa()`(반복 다항 피팅, 기본 5차)이므로,
그것으로 못 없애는 배경을 만들면 배경 분리 과제 자체가 성립하지 않는다.

## 저장물 구분

가상으로 찍은 결과도 `data/results/` 에 실물과 **같은 형식**으로 쌓인다. 구분은
`data/runs/<세션>/manifest.json` 의 `"virtual": true`, `"scene": "<이름>"` 로만 가능하다.
채점·비교 전에 반드시 이 필드로 거를 것.
