#!/bin/bash
# ECG Dataset Downloader
# - PhysioNet datasets: PTB-XL, MIMIC-IV-ECG, CinC2021 (requires account + data use agreement)
# - Zenodo dataset: CODE-15 (public)
#
# PhysioNet 계정 등록 및 데이터 사용 동의 후 사용하세요:
# https://physionet.org/register/

set -e

# ─── 설정 ────────────────────────────────────────────────────────────────────
DOWNLOAD_DIR="${1:-./data}"          # 첫 번째 인자로 저장 경로 지정 가능 (기본: ./data)
PHYSIONET_USER="${PHYSIONET_USER:-}"  # 환경변수로 설정하거나 아래에 직접 입력
PHYSIONET_PASS="${PHYSIONET_PASS:-}"  # 환경변수로 설정하거나 아래에 직접 입력

# PhysioNet 계정이 환경변수로 없으면 입력 받기
if [ -z "$PHYSIONET_USER" ]; then
  read -rp "PhysioNet username: " PHYSIONET_USER
fi
if [ -z "$PHYSIONET_PASS" ]; then
  read -rsp "PhysioNet password: " PHYSIONET_PASS
  echo
fi

mkdir -p "$DOWNLOAD_DIR"

# ─── 공통 함수 ───────────────────────────────────────────────────────────────
physionet_download() {
  local url="$1"
  local dest="$2"
  mkdir -p "$dest"
  echo "[PhysioNet] Downloading: $url → $dest"
  curl -u "${PHYSIONET_USER}:${PHYSIONET_PASS}" \
       -L --retry 4 --retry-delay 5 \
       -# \
       -o /dev/null \
       --fail \
       "$url" 2>/dev/null || { echo "  ✗ 인증 실패. PhysioNet 계정과 데이터 사용 동의를 확인하세요."; exit 1; }

  # wget으로 재귀 다운로드 (curl은 재귀 미지원)
  wget --user="$PHYSIONET_USER" --password="$PHYSIONET_PASS" \
       -r -N -c -np \
       --directory-prefix="$dest" \
       --no-host-directories --cut-dirs=3 \
       "$url"
}

# ─── 1. PTB-XL ───────────────────────────────────────────────────────────────
echo ""
echo "=== [1/4] PTB-XL ==="
PTB_XL_URL="https://physionet.org/files/ptb-xl/1.0.3/"
PTB_XL_DIR="$DOWNLOAD_DIR/ptb-xl"
physionet_download "$PTB_XL_URL" "$PTB_XL_DIR"
echo "  ✓ PTB-XL 완료: $PTB_XL_DIR"

# ─── 2. MIMIC-IV-ECG ─────────────────────────────────────────────────────────
echo ""
echo "=== [2/4] MIMIC-IV-ECG ==="
echo "  ⚠ MIMIC-IV-ECG는 credentialed access가 필요합니다."
echo "    https://physionet.org/content/mimic-iv-ecg/ 에서 별도 동의 필요"
MIMIC_URL="https://physionet.org/files/mimic-iv-ecg/1.0/"
MIMIC_DIR="$DOWNLOAD_DIR/mimic-iv-ecg"
physionet_download "$MIMIC_URL" "$MIMIC_DIR"
echo "  ✓ MIMIC-IV-ECG 완료: $MIMIC_DIR"

# ─── 3. CinC2021 묶음 (Chapman-Shaoxing, CPSC, CPSC-Extra, Georgia, Ningbo, PTB, St-Petersburg) ──
echo ""
echo "=== [3/4] CinC2021 묶음 ==="
CINC_URL="https://physionet.org/files/challenge-2021/1.0.3/training/"
CINC_DIR="$DOWNLOAD_DIR/cinc2021"

mkdir -p "$CINC_DIR"

for dataset in chapman_shaoxing cpsc_2018 cpsc_2018_extra georgia ningbo ptb st_petersburg_incart; do
  echo "  → $dataset"
  wget --user="$PHYSIONET_USER" --password="$PHYSIONET_PASS" \
       -r -N -c -np \
       --directory-prefix="$CINC_DIR/$dataset" \
       --no-host-directories --cut-dirs=4 \
       "${CINC_URL}${dataset}/"
  echo "    ✓ $dataset 완료"
done
echo "  ✓ CinC2021 묶음 완료: $CINC_DIR"

# ─── 4. CODE-15 (Zenodo, 공개) ───────────────────────────────────────────────
echo ""
echo "=== [4/4] CODE-15 (Zenodo) ==="
CODE15_DIR="$DOWNLOAD_DIR/code-15"
mkdir -p "$CODE15_DIR"

# Zenodo record 4916206 파일 목록 조회 후 다운로드
echo "  Zenodo에서 파일 목록 조회 중..."
FILES=$(curl -s "https://zenodo.org/api/records/4916206" \
        | python3 -c "import sys,json; [print(f['links']['self']) for f in json.load(sys.stdin)['files']]")

echo "  다운로드 시작..."
while IFS= read -r file_url; do
  filename=$(basename "$file_url" | sed 's/?.*$//')
  echo "  → $filename"
  curl -L --retry 4 --retry-delay 5 -# \
       -o "$CODE15_DIR/$filename" \
       "$file_url"
done <<< "$FILES"

echo "  ✓ CODE-15 완료: $CODE15_DIR"

# ─── 완료 ────────────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo "모든 다운로드 완료!"
echo ""
echo "다음 단계: HuggingFace Dataset으로 변환"
echo ""
echo "  python -m scripts.convert_to_hf_dataset --data-dir \"$PTB_XL_DIR\" --dataset ptb-xl --out /path/to/ptb-xl-hf --verbose"
echo "  python -m scripts.convert_to_hf_dataset --data-dir \"$MIMIC_DIR\" --dataset mimic-iv-ecg --out /path/to/mimic-iv-ecg-hf --verbose"
echo "  python -m scripts.convert_to_hf_dataset --data-dir \"$CINC_DIR/chapman_shaoxing\" --dataset chapman-shaoxing --out /path/to/chapman-shaoxing-hf --verbose"
echo "  python -m scripts.convert_to_hf_dataset --data-dir \"$CINC_DIR/cpsc_2018\" --dataset cpsc --out /path/to/cpsc-hf --verbose"
echo "  python -m scripts.convert_to_hf_dataset --data-dir \"$CINC_DIR/cpsc_2018_extra\" --dataset cpsc-extra --out /path/to/cpsc-extra-hf --verbose"
echo "  python -m scripts.convert_to_hf_dataset --data-dir \"$CINC_DIR/georgia\" --dataset georgia --out /path/to/georgia-hf --verbose"
echo "  python -m scripts.convert_to_hf_dataset --data-dir \"$CINC_DIR/ningbo\" --dataset ningbo --out /path/to/ningbo-hf --verbose"
echo "  python -m scripts.convert_to_hf_dataset --data-dir \"$CINC_DIR/ptb\" --dataset ptb --out /path/to/ptb-hf --verbose"
echo "  python -m scripts.convert_to_hf_dataset --data-dir \"$CINC_DIR/st_petersburg_incart\" --dataset st-petersburg --out /path/to/st-petersburg-hf --verbose"
echo "  python -m scripts.convert_to_hf_dataset --data-dir \"$CODE15_DIR\" --dataset code-15 --out /path/to/code-15-hf --verbose"
