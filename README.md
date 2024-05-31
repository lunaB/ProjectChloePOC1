# Boilerplate-Python
Python DevContainer Boilerplate  
Python + DevContainer + Docker-Compose + pipenv + dotenv

## Run DevContainer
...

## Run Local
```
# 파이썬 설치
pyenv install 3.10.13  
# 파이썬 경로 설정
pyenv local 3.10.13  
# 설정 잘 됬나 확인
python --version
# pip 업그레이드
pip install --upgrade pip
# pipenv 설치
pip install pipenv
# venv 생성
pipenv --python 3.10.13
# 종속성 설치
pipenv install
```

## Issue
typeguard = "==2.13.3", https://github.com/PaddlePaddle/PaddleSpeech/issues/3051
pyobjc, https://stackoverflow.com/questions/12767669/import-error-no-module-named-appkit