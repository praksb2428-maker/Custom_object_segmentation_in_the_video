# CCTV 최종 실험 결과 — 2026년 11월 20일

사용자가 선택한 사람을 SAM 2.1로 분할·추적하고 영상으로 저장하는 기능을 확인했다. 프로젝트 작성자는 가운데 사람의 위치 일치율 98.4%를 이번 프로젝트의 완성 기준으로 삼았다. 이 수치는 아래에 명시한 한 영상의 조건에 해당한다.

## 영상과 평가 조건

- 자료: CAVIAR `Meet_Crowd.mp4`, 384×288, 25fps, 492프레임(19.68초).
- 모델: 사전 학습된 SAM 2.1 Hiera Small. 가중치를 바꾸거나 파인튜닝하지 않았다.
- 선택 시점: 160프레임(6.4초). 확인된 세 사람에게 각각 ID 1·2·3을 부여했다. 이때 가운데 사람은 ID 1이다.
- 초기 박스는 공식 자료를 참고해 작성하고 프로젝트 작성자가 화면을 보고 확인했다. 이후 프레임의 정답 박스는 모델 입력에 사용하지 않았다.
- 출력은 원본 492프레임 전체다. 처음 160프레임에는 선택 마스크가 없으며, 추적은 6.4초부터 시작한다.
- 채점은 공식 XML에서 대상이 보인다고 표시한 프레임만 사용했다. 영상 전체 길이와 채점 구간은 다르다.

## 같은 초기 선택으로 비교한 결과

| 실행 조건 | 평가 건수 | 평균 박스 IoU | 위치 일치 | 보이는 대상의 빈 출력 |
|---|---:|---:|---:|---:|
| 순수 SAM 2, 사람 3명 | 578 | 0.8653 | 575/578 (99.48%) | 0건 |
| 검출·추적 모듈 + SAM 2 이미지 분할 | 578 | 0.4888 | 317/578 (54.84%) | 225건 |
| 순수 SAM 2, 가운데 사람만 | 192 | 0.8273 | 189/192 (98.44%) | 0건 |

평가 1건은 한 프레임에서 한 사람을 확인한 것이다. 위치 일치는 출력 마스크를 감싼 박스와 정답 박스의 IoU가 0.5 이상인 경우다. **98.4%나 99.5%는 픽셀 단위 분할 정확도를 뜻하지 않는다.**

추가 모듈은 YOLO11s와 BoT-SORT의 검출 특징을 이용하는 방식이었다. 추적기가 대상을 놓치면 SAM 2에 전달할 박스도 사라져 분할이 중단됐다. 비교 결과가 나빠 기본 프로그램에 통합하지 않았으며, 배포 코드의 필수 의존성에도 추가하지 않았다. 사람 전용 ReID 모델을 따로 학습한 실험은 아니다.

## 가운데 사람 단독 추적

SAM 2에 ID 1 한 명만 등록하고 새로 추론했다. 사람이 보이는 192프레임에서 빈 출력은 없었으며, 189프레임이 위치 일치 기준을 통과했다. 10.80초·10.92초에는 분할 영역이 작아졌고, 14.04초에는 대상이 영상 아래로 나가며 영역이 줄었다. 마지막 보임 주석 이후 14.08초와 14.12초에 작은 경계 마스크가 남았다.

단독 추적의 492개 마스크를 이전 3명 추적 중 ID 1의 마스크와 전부 대조했으며 픽셀 차이는 0개였다. 이 영상에서는 선택 대상을 한 명으로 줄여도 해당 사람의 결과가 개선되거나 나빠지지 않았다.

## 재현하기

먼저 [README의 준비 절차](../README.md#준비)에 따라 환경과 Small 모델을 설치한다. 사용 가능한 영상을 직접 준비해야 하며 저장소에는 입력 영상을 포함하지 않는다. 예시 선택 파일은 SHA-256으로 고정한 `Meet_Crowd.mp4`에만 적용된다. 이름이 같아도 다른 변환본이면 좌표가 맞는지 확인해 새 선택 파일을 작성해야 한다.

다음 명령은 저장소 루트의 PowerShell에서 실행한다.

```powershell
# 가운데 사람(ID 1)만 선택
.\.venv\Scripts\python.exe tools/run_cctv_sam2.py --video input/caviar/Meet_Crowd.mp4 --selection experiments/meet-crowd-selection.json --object-ids 1 --output outputs/cctv-middle-final

# 같은 시점에서 세 사람을 함께 선택
.\.venv\Scripts\python.exe tools/run_cctv_sam2.py --video input/caviar/Meet_Crowd.mp4 --selection experiments/meet-crowd-selection.json --output outputs/cctv-three-final
```

이 도구는 공식 SAM 2 영상 예측 API를 사용하며, 프레임별 모델 영역을 그대로 이진 마스크로 내보낸다. 기존 `main.py` 기본 실행의 연속 추적 보정과는 별도 실행 경로다. 위 표는 이 순수 SAM 2 실험 경로에서 얻은 값이며, 기본 ▶ 실행의 성능 수치로 옮겨 쓰지 않는다. 긴 영상은 메모리와 처리 시간이 늘어난다.

출력 폴더에는 `overlay.mp4`, `foreground.mp4`, 객체별 `masks/`, 완료 기록 `run.json`이 생긴다. MP4에는 원래 음성을 넣지 않으며, `foreground.mp4`는 선택 대상 외 영역을 검게 표시한다. 기존 출력 폴더는 덮어쓰지 않는다. 모델 입력용 임시 JPEG는 실행 종료 시 정리한다.

공식 `Meet_Crowd.xml`을 직접 준비했다면 다음 명령으로 같은 박스 지표를 계산할 수 있다.

```powershell
.\.venv\Scripts\python.exe tools/evaluate_cctv_boxes.py --masks outputs/cctv-middle-final/masks --reference input/caviar/Meet_Crowd.xml --object-map 1:0 --start-frame 160 --output outputs/cctv-middle-metrics.json

.\.venv\Scripts\python.exe tools/evaluate_cctv_boxes.py --masks outputs/cctv-three-final/masks --reference input/caviar/Meet_Crowd.xml --object-map 1:0 2:1 3:2 --start-frame 160 --output outputs/cctv-three-metrics.json
```

## 확인 범위와 남은 한계

확정된 사람 작성 픽셀 마스크가 없어 픽셀 IoU·Dice는 계산하지 않았다. 다른 사람의 박스와 더 잘 겹치는지 확인하는 값도 신원 전환을 직접 센 지표는 아니다. 심한 가림 뒤 동일 인물 복구, 다른 장소·화질의 CCTV, 모든 작은 물체에 대한 성능은 추가 검증이 필요하다.

추가 모듈의 다른 두 영상(`LeftBag`, `Meet_Split_3rdGuy`) 시험은 기존 선택 시점의 초기 ID 등록 조건을 통과하지 못했다. 두 영상 전체에서 새 구성을 검증한 것으로 표시하지 않는다. 이전 2.76초 선택 실험은 시작 시점과 선택 조건이 달라 위 수치와 단순 개선율을 계산하지 않는다.

기존 [중간보고서](../mid-report-20261115-updated.md)는 11월 19일까지의 기록이다. 본 문서는 그 이후의 최종 실험을 설명한다. 입력·출력 영상, 원본 이미지, 정답 XML·마스크, 모델 가중치, 가상환경은 이번 업로드에서 제외했다.

## 자료와 결과 파일

- [CAVIAR 공식 자료 페이지](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/) — EC Funded CAVIAR project / IST 2001 37540.
- [최종 집계 JSON](../experiments/final-results-20261120.json)
- [최종 집계 CSV](../experiments/final-results-20261120.csv)
- [초기 선택 예시](../experiments/meet-crowd-selection.json) — 초기 시점만 포함하며 이후 정답은 포함하지 않는다.
