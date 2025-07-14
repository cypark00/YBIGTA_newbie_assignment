
# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
## TODO
if ! command -v conda &> /dev/null; then
    echo "[INFO] Miniconda 설치 중..."
    apt update && apt install -y wget
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
else
    echo "[INFO] conda가 이미 설치되어 있습니다."
fi


# Conda 환셩 생성 및 활성화
## TODO
if ! conda info --envs | grep -q "myenv"; then
    echo "[INFO] 가상환경 생성 중..."
    conda create -y -n myenv python=3.10
fi

source ~/miniconda/etc/profile.d/conda.sh
conda activate myenv 


## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
## TODO
pip install mypy

# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    ## TODO
    base_name=$(basename "$file" .py)
    problem_id="$base_name"

    input_file="../input/${problem_id}_input"
    output_file="../output/${problem_id}_output"
    echo "[INFO] 실행 중: $file"
    python "$file" < "$input_file" > "$output_file"
done

# mypy 테스트 실행 및 mypy_log.txt 저장
## TODO
mypy *.py > ../mypy_log.txt

# conda.yml 파일 생성
## TODO
pip freeze > ../conda.yml
# 가상환경 비활성화
## TODO
conda deactivate