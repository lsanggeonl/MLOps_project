# -*- coding: utf-8 -*-
"""
yolo_retrain_4class_v2.py
===================================================================
목적:
  banana_4class_v1(베이스라인) 대비 첫 번째 개선 실험.
  바뀐 것은 EPOCHS 하나뿐(30 -> 50), 나머지(데이터셋/증강/batch/seed)는
  베이스라인과 동일하게 유지 — experiments.md 갱신 규칙("무엇이 바뀌었는지
  먼저 적을 것")에 맞춰 비교가 가능하도록 변수를 하나만 바꿈.

  변경 근거: banana_4class_v1의 정규화 혼동행렬을 보면 ripe·rotten 둘 다
  다른 클래스로 "오분류"되기보다 "background로 아예 놓침"(ripe 2%,
  rotten 9%)이 recall을 깎는 주된 경로였음. 30 epoch로는 아직 덜 수렴해
  자신 없는 케이스를 배경으로 흘려보내는 비율이 있을 수 있다고 보고,
  epoch을 늘려 이 비율이 줄어드는지 확인한다.

증강 설정은 프로젝트 지침(섹션 9) 준수, 베이스라인과 동일:
  - hsv_s = 0.0   (채도 증강 OFF, 색상=숙성도 신호 보존)
  - hsv_h = 0.0   (색조 증강 OFF)
  - hsv_v = 0.1   (밝기 +-10% 상한)
  - degrees = 15  (회전 +-15도)
  - fliplr = 0.5  (좌우반전 허용)
  - translate = 0.1, scale = 0.1
  - mosaic / mixup / erasing / auto_augment 모두 OFF (지침 외)

재현성:
  - seed = 42
  - workers = 0  (Windows CUDA 안정성)

채택 기준 (팀 확정, 2026-08-19):
  - ripe recall이 베이스라인(0.917)보다 올라가야 함
  - rotten recall이 베이스라인(0.781) 밑으로 떨어지면 채택 불가(하한선)

사용법:
  python yolo_retrain_4class_v2.py
  (크래시로 중단됐다면) python yolo_retrain_4class_v2.py --resume
===================================================================
"""

from pathlib import Path
import argparse
import re
import sys

# ============ [설정] 여기만 수정 ==================================
DATASET_ROOT = Path(__file__).resolve().parent  # 이 스크립트가 있는 폴더 (프로젝트 루트)
RUN_NAME     = "banana_4class_v3_11s"      # runs/detect/<이 이름>/ 에 결과 저장
MODEL        = "yolo11s.pt"              # n -> s (파라미터 2.6M -> 9.4M). 이번 실험의 변경점
EPOCHS       = 50                        # 베이스라인 30 -> 50 (이번 개선 실험의 유일한 변경점)
IMGSZ        = 640
BATCH        = 16                        # 11s 는 11n 보다 무거워 8GB VRAM 에서 32 는 OOM 위험.
                                         # 부족하면 8 -> 4 로 더 낮출 것. (개선실험1의 32 와 다름 = 교란변수)
DEVICE       = 0                         # GPU 0. CPU 강제 시 "cpu"
# ==================================================================

LOG_PATH = Path(__file__).resolve().parent / "train_log_4class_v3_11s.txt"


class Tee:
    """콘솔에는 그대로 실시간 출력하고, 로그 파일에는 진행바(\\r) 중간갱신을
    걸러내고 완성된 줄만 기록한다. cmd|powershell 파이프보다 훨씬 안정적."""

    def __init__(self, console, logfile):
        self.console = console
        self.logfile = logfile
        self._buf = ""

    def write(self, data):
        self.console.write(data)
        self.console.flush()
        self._buf += data
        # "...내용...\r" 형태(진행바 중간갱신)는 최종 상태만 남기고 버린다
        self._buf = re.sub(r"[^\r\n]*\r", "", self._buf)
        if "\n" in self._buf:
            *lines, self._buf = self._buf.split("\n")
            for line in lines:
                if line.strip():
                    self.logfile.write(line + "\n")
            self.logfile.flush()

    def flush(self):
        self.console.flush()
        self.logfile.flush()


def setup_logging():
    log_file = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

# 프로젝트 지침 준수 증강 설정 (Ultralytics 기본값 덮어쓰기)
AUG_CONFIG = {
    # 색상 관련 -----------------------------------------------------
    "hsv_h": 0.0,   # hue: 지침에서 금지 -> OFF
    "hsv_s": 0.0,   # saturation: 지침에서 금지 -> OFF (핵심)
    "hsv_v": 0.1,   # value(밝기): 지침 상한 +-10%
    # 기하 변환 -----------------------------------------------------
    "degrees": 15.0,     # 회전 +-15도 (지침)
    "translate": 0.1,    # 이동 +-10%
    "scale": 0.1,        # 확대/축소 +-10%
    "shear": 0.0,        # 지침 외 -> OFF
    "perspective": 0.0,  # 지침 외 -> OFF
    "fliplr": 0.5,       # 좌우반전 50% (지침 허용)
    "flipud": 0.0,       # 상하반전 지침 없음 -> OFF (바나나 방향 유지)
    # 지침 외 강한 증강 전부 OFF -----------------------------------
    "mosaic": 0.0,       # OFF
    "mixup": 0.0,        # OFF
    "copy_paste": 0.0,   # OFF
    "erasing": 0.0,      # OFF
    "auto_augment": None,  # randaugment/autoaug OFF
}


def fix_stale_checkpoint_paths(pt_path: Path, data_yaml: Path, detect_dir: Path, run_name: str):
    """last.pt 안에 저장된 옛 프로젝트 경로(project/data/save_dir)를 현재 경로로 고쳐쓴다.

    체크포인트 파일에는 학습 당시의 project/data/save_dir 경로가 함께 저장되는데,
    프로젝트 폴더가 Downloads/banana_4class -> Desktop/banana_4class_2 로
    이동한 뒤에도 이 메모는 옛 경로를 그대로 가리킨다. resume=True 로 재개할 때
    Ultralytics 가 이 옛 data 경로를 찾지 못하면 자기 기본값(coco8.yaml, nc=80)으로
    조용히 대체해버려서, 이미 nc=4로 저장된 가중치와 크기가 맞지 않아 충돌한다
    (size mismatch).
    save_dir 은 project/name 조합보다 항상 우선 적용되므로(Ultralytics get_save_dir
    로직) project만 고쳐써도 save_dir이 옛 경로로 남아있으면 소용없다. 그렇다고
    save_dir 키를 그냥 지워버리면, exist_ok=False 설정 때문에 폴더가 이미 존재할 때마다
    get_save_dir 이 이름을 banana_..-2, -3 처럼 매번 새로 증가시켜버려서 결과물이
    계속 다른 폴더로 흩어진다(2026-08-19 실제로 이 두 문제 다 발생: 처음엔 결과가
    엉뚱한 Downloads 폴더로, 그 다음엔 재개할 때마다 -2, -3 폴더로 계속 새로 생김).
    그래서 save_dir을 아예 지우지 않고 "정확히 이 경로"로 고정해서, 항상 같은 폴더에
    이어서 저장되게 한다.
    """
    import torch

    save_dir = str(detect_dir / run_name)
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("train_args")
    if not train_args:
        return
    changed = False
    if train_args.get("data") != str(data_yaml):
        train_args["data"] = str(data_yaml)
        changed = True
    if train_args.get("project") != str(detect_dir):
        train_args["project"] = str(detect_dir)
        changed = True
    if train_args.get("save_dir") != save_dir:
        train_args["save_dir"] = save_dir
        changed = True
    if changed:
        torch.save(ckpt, pt_path)
        print(f"[정리] {pt_path.name} 안에 저장된 옛 경로(project/data/save_dir)를 현재 프로젝트 경로로 고쳤습니다.")


def assert_resumable(pt_path: Path):
    """last.pt 가 실제로 재개 가능한 상태인지 미리 확인하고, 아니면 바로 멈춘다.

    Ultralytics 는 resume=True 로 넘긴 체크포인트에 epoch/optimizer 정보가
    없으면(예: 학습이 정상 종료돼서 이미 한 번 정리(strip)된 파일) 에러를 내는 대신
    "재개 불가 - 새 학습을 시작합니다" 경고만 찍고 조용히 우리 설정을 전부 무시한 채
    기본값(coco8 샘플 데이터셋, nc=80, epoch 100)으로 새 학습을 시작해버린다.
    자동 재시작 루프와 만나면 이게 매번 반복되며 매 시도마다 새 폴더가 쌓이는데,
    아무 에러도 안 나서 눈치채기 어렵다 - 실제로 이 문제로 하룻밤 새 쓸모없는 폴더가
    948개 생긴 사고가 있었다(2026-08-19~20). 그래서 Ultralytics에게 넘기기 전에
    우리가 먼저 검사해서, 재개 불가능하면 바로 에러를 내고 멈춘다.
    """
    import torch

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", -1)
    has_optimizer = ckpt.get("optimizer") is not None
    if epoch < 0 or not has_optimizer:
        print(f"[X] {pt_path} 는 재개 불가능한 체크포인트입니다 (epoch={epoch}, optimizer 있음={has_optimizer}).")
        print("    십중팔구 학습이 이미 완주되어 정리(strip)된 파일입니다.")
        print("    --resume 을 쓰지 말고, results.csv 마지막 epoch을 확인해 학습이 진짜 끝났는지부터 보세요.")
        sys.exit(1)


def iter_run_dirs(detect_dir: Path, run_name: str):
    """RUN_NAME, RUN_NAME-2, RUN_NAME-3 ... 를 전부 모은다.

    Ultralytics 는 exist_ok=False 일 때 같은 이름 폴더가 있으면 결과를 <name>-2,
    -3 으로 분기시킨다. 고정 경로 하나만 감시하면 완주를 영원히 못 본다
    (CLAUDE.md 규칙 ②). 그래서 진행도 판정은 항상 이 목록 전체를 훑는다.
    """
    if not detect_dir.exists():
        return []
    return sorted(p for p in detect_dir.glob(f"{run_name}*") if p.is_dir())


def max_saved_epoch(detect_dir: Path, run_name: str) -> int:
    """RUN_NAME* 폴더들의 results.csv 중 가장 큰 epoch 번호. 하나도 없으면 0."""
    best = 0
    for run_dir in iter_run_dirs(detect_dir, run_name):
        csv_path = run_dir / "results.csv"
        if not csv_path.exists():
            continue
        try:
            lines = [ln for ln in csv_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        except OSError:
            continue
        if len(lines) < 2:          # 헤더뿐이면 진행분 없음
            continue
        try:
            best = max(best, int(float(lines[-1].split(",")[0])))
        except (ValueError, IndexError):
            continue
    return best


def find_complete_marker(detect_dir: Path, run_name: str):
    """완주 표식 파일이 있으면 그 경로를 돌려준다."""
    for run_dir in iter_run_dirs(detect_dir, run_name):
        marker = run_dir / "TRAINING_COMPLETE.txt"
        if marker.exists():
            return marker
    return None


def mark_complete(run_dir: Path, epochs: int):
    """완주 표식을 남긴다.

    자동 재시작 루프(auto_resume_4class_v2.bat)는 이 파일 하나로 "더 돌릴 필요
    없음"을 판정한다. results.csv 숫자만 보고 판정하던 방식이 949회 폭주의
    한 고리였다 (CLAUDE.md 규칙 ②).
    """
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "TRAINING_COMPLETE.txt").write_text(
            f"detection training finished. epochs={epochs}\n", encoding="utf-8"
        )
        print(f"[표식] 완주 표식을 남겼습니다: {run_dir / 'TRAINING_COMPLETE.txt'}")
    except OSError as e:
        print(f"[!] 완주 표식을 쓰지 못했습니다(학습 자체는 정상): {e}")


def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="last.pt 에서 이어서 학습")
    args = parser.parse_args()

    # 데이터셋 존재 확인
    data_yaml = DATASET_ROOT / "data.yaml"
    if not data_yaml.exists():
        print(f"[X] data.yaml 을 찾지 못했습니다: {data_yaml}")
        print("    prepare_4class.py 를 먼저 실행하세요.")
        sys.exit(1)

    # nc=4 확인
    import yaml
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("nc") != 4:
        print(f"[X] data.yaml 의 nc 가 4 가 아닙니다: nc={cfg.get('nc')}")
        print("    prepare_4class.py 가 정상 완료됐는지 확인하세요.")
        sys.exit(1)
    print(f"[OK] data.yaml 확인: nc=4, names={cfg.get('names')}")

    # 증강 설정 로그 출력 (지침 준수 확인)
    print("=" * 67)
    print(" [증강 설정 - 지침 준수 확인]")
    for k, v in AUG_CONFIG.items():
        print(f"   {k:14s} = {v}")
    print("=" * 67)

    # --- 완주 표식 확인 (무거운 import 전에 빠르게 빠져나가기 위해 먼저) -------
    # 종료코드 3 = "이미 끝났다". auto_resume_4class_v2.bat 이 이 코드를 보고
    # 재시작 루프를 정상 종료한다. 일반 오류(1)와 구분해야 루프가 헛돌지 않는다.
    detect_dir = DATASET_ROOT / "runs" / "detect"
    marker = find_complete_marker(detect_dir, RUN_NAME)
    if marker is not None:
        print(f"[완주] 이미 끝난 학습입니다: {marker}")
        print("    정말 다시 돌리려면 이 표식 파일을 지우고 실행하세요.")
        sys.exit(3)

    # Ultralytics import 는 데이터 확인 후에 (import 자체 무거움)
    from ultralytics import YOLO

    last_pt = detect_dir / RUN_NAME / "weights" / "last.pt"

    if args.resume:
        # Ultralytics 의 resume=True 는 체크포인트에 저장된 이름(project/name)을
        # 그대로 이어받는데, exist_ok=False 때문에 이 이름이 RUN_NAME-2, -3 처럼
        # 계속 밀릴 수 있다. 그러면 다음번 재개 때 고정 경로(last_pt)는 옛날 진행분을
        # 가리키게 되므로, RUN_NAME* 폴더들 중 "실제로 가장 최근에 저장된" last.pt를
        # 찾아서 그걸 이어받는다 (진행분 유실 방지).
        candidates = [
            p for p in detect_dir.glob(f"{RUN_NAME}*/weights/last.pt") if p.is_file()
        ]
        if not candidates:
            print(f"[X] --resume 옵션을 줬지만 last.pt 가 없습니다: {last_pt}")
            sys.exit(1)
        latest_pt = max(candidates, key=lambda p: p.stat().st_mtime)
        if latest_pt != last_pt:
            print(f"[정리] 실제 최신 진행분을 {latest_pt} 에서 찾았습니다. 정식 경로로 복사합니다.")
            last_pt.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(latest_pt, last_pt)
        fix_stale_checkpoint_paths(last_pt, data_yaml, detect_dir, RUN_NAME)

        # 표식은 없지만 results.csv 상으로는 이미 목표 epoch 을 채운 경우.
        # 이 상태에서 --resume 을 걸면 Ultralytics 가 "재개 불가"를 조용히 삼키고
        # 기본값(coco8)으로 새 학습을 시작한다 = 949회 폭주의 직접 원인.
        # 그래서 재개를 시도하기 전에 여기서 끊고 완주(3)로 보고한다.
        done = max_saved_epoch(detect_dir, RUN_NAME)
        if done >= EPOCHS:
            print(f"[완주] results.csv 기준 이미 {done}/{EPOCHS} epoch 을 채웠습니다. 재개하지 않습니다.")
            mark_complete(last_pt.parent.parent, EPOCHS)
            sys.exit(3)

        assert_resumable(last_pt)
        print(f"[OK] {last_pt} 에서 이어서 학습합니다.")
        model = YOLO(str(last_pt))
        results = model.train(resume=True)
    else:
        run_dir = last_pt.parent.parent  # runs/detect/<RUN_NAME>
        if run_dir.exists():
            if last_pt.exists():
                print(f"[X] {run_dir} 에 이미 진행 중인 학습(last.pt)이 있습니다.")
                print("    이어서 하려면 --resume 옵션을 쓰세요. 정말 새로 시작하려면 이 폴더를 직접 지우고 재실행하세요.")
                sys.exit(1)
            # last.pt 없는 죽은 시도(0 epoch 등) -> exist_ok=False 가 -2,-3.. 으로
            # 새 폴더를 만들어버려서 다음 재개 시 이 폴더를 못 찾는 문제 방지
            import shutil
            print(f"[정리] 이전에 실패한 빈 시도 폴더를 지웁니다: {run_dir}")
            shutil.rmtree(run_dir)

        model = YOLO(MODEL)
        results = model.train(
            data=str(data_yaml),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=0,             # Windows CUDA 안정성
            cache=False,
            seed=42,               # 재현성 (지침)
            project=str(Path(__file__).resolve().parent / "runs" / "detect"),  # 절대경로 (project 중첩 방지)
            name=RUN_NAME,
            exist_ok=False,        # 같은 이름 폴더 있으면 오류 -> 안전
            plots=True,            # results.png, confusion_matrix.png 자동
            save=True,
            verbose=True,
            **AUG_CONFIG,
        )

    # 실제로 결과가 저장된 폴더에 완주 표식을 남긴다.
    # (RUN_NAME-2 등으로 분기됐을 수 있으므로 고정 경로가 아니라 trainer 가
    #  알려준 save_dir 을 쓴다. 없으면 정식 경로로 폴백.)
    save_dir = Path(getattr(results, "save_dir", detect_dir / RUN_NAME))
    mark_complete(save_dir, EPOCHS)

    print("=" * 67)
    print(" 학습 완료.")
    print(f" 결과 폴더: {save_dir}")
    print(" 주요 산출물:")
    print("   weights/best.pt         (배포용 가중치)")
    print("   results.png             (학습 곡선)")
    print("   confusion_matrix.png    (4클래스 오분류 행렬)")
    print("   confusion_matrix_normalized.png")
    print("   BoxPR_curve.png, BoxF1_curve.png")
    print("   args.yaml               (실제 적용된 하이퍼파라미터 저장 - 검증용)")
    print("=" * 67)


if __name__ == "__main__":
    main()
