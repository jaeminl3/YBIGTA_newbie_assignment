#!/bin/bash

# miniconda가 존재하지 않을 경우 설치
if ! command -v conda &> /dev/null
then
    echo "미니콘다가 설치되지 않았습니다. 설치를 진행합니다."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    rm miniconda.sh
    echo "미니콘다가 설치되었습니다."
else
    echo "미니콘다가 이미 설치되어있습니다."
fi

# Conda 환경 생성 및 활성화
if ! conda info --envs | grep -q "myenv"
then
    echo "'myenv'가상환경을 생성합니다."
    conda create -y -n myenv python=3.9
fi

source $HOME/miniconda/bin/activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "가상환경 활성화: 성공"
else
    echo "가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
pip install mypy > /dev/null 2>&1

# Submission 폴더 파일 실행
cd submission || { echo "submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    base_name=$(basename "$file" .py)
    input_file="../input/${base_name}_input"
    output_file="../output/${base_name}_output"

    if [ -f "$input_file" ]; then
        python "$file" < "$input_file" > "$output_file"
    fi
done

# mypy 테스트 실행
for file in *.py; do
    mypy "$file"
done

# 가상환경 비활성화
echo "가상환경 비활성화"
conda deactivate
