# Callnote · 통화 인사이트

Windows 데스크톱 앱으로 WAV 고객센터 상담을 분석합니다. 음성 인식은 **OpenAI 원본 Whisper 구현의 `large-v3` 체크포인트**를 사용하며, `faster-whisper`는 사용하지 않습니다. 상담 요약과 전사 질의응답은 로컬 [Qwen3 4B Instruct 2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) 모델이 처리합니다. 이 공개 모델은 Apache-2.0 라이선스를 사용합니다.

## 다운로드 및 실행

GitHub Releases에서 사용할 환경에 맞는 파일 하나를 내려받습니다.

- CPU용 `Callnote-Windows-CPU.zip`: 압축을 푼 뒤 `Callnote\Callnote.exe`를 실행합니다.
- Titan Xp 등 Pascal GPU용 `Callnote-Pascal-GPU.exe`: 실행해서 원하는 폴더에 압축을 푼 뒤, 생성된 `Callnote\Callnote.exe`를 실행합니다. 이 파일은 자동 압축해제 패키지이며 별도 압축 프로그램은 필요 없습니다. NVIDIA 드라이버가 설치되어 있어야 합니다.

Python, PyTorch, FFmpeg를 별도로 설치할 필요가 없습니다. 앱의 코드와 실행 종속성은 패키지에 포함되어 있습니다. GPU 패키지는 GitHub의 파일당 2GiB 제한에 맞추기 위해 7-Zip으로 압축한 단일 EXE로 배포합니다.

### v1.1.0에서 작은 패치로 업데이트

이미 v1.1.0을 사용 중이면 전체 패키지 대신 해당 CPU/GPU 버전의 `Callnote-v1.1.0-to-v1.1.1-…-Patch.zip`만 받을 수 있습니다. 앱을 종료하고 패치 ZIP을 임시 폴더에 압축 해제한 뒤, 그 안의 `Callnote` 폴더 내용을 기존 `Callnote` 폴더 안으로 복사해 같은 이름의 파일을 덮어쓰세요. 기존 `_internal` 폴더는 남겨 두고 파일을 합쳐야 합니다. CPU와 GPU 패치를 서로 바꾸어 적용하지 마세요. 기존 모델은 다시 받지 않습니다.

첫 실행에는 앱이 모델을 자동으로 다운로드합니다. Whisper large-v3와 Qwen3 모델 합계 약 10.9GB를 받기 위해 인터넷 연결과 저장 공간이 필요합니다. 내려받은 모델은 `%LOCALAPPDATA%\Callnote\models`에 저장되고, 이후 통화 분석은 오프라인으로 실행됩니다. 다운로드가 중단되면 앱을 다시 열고 **모델 다시 받기**를 누르세요. NVIDIA GPU용 빌드는 GPU를 자동 사용하고, GPU가 없으면 CPU로 동작합니다. 일반 빌드는 CPU 전용입니다.

## 개발 및 빌드

Python 3.12를 사용할 경우 **3.12.1 이상**이 필요합니다. 3.12.0에는 PyInstaller 패키징 후 요약 모델을 불러올 때 `name 'name' is not defined` 오류를 일으키는 인터프리터 문제가 있으므로 빌드 스크립트에서 차단합니다. 배포 파일에는 수정된 런타임이 포함되며, 최종 사용자가 Python을 설치할 필요는 없습니다.

이미 생성한 가상 환경의 Python 버전은 자동으로 바뀌지 않습니다. 새 인터프리터를 지정하려면 `-PythonExecutable "C:\path\to\python.exe" -EnvironmentPath ".venv-pascal-new"`처럼 새 환경 경로와 함께 사용하세요.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Set-ExecutionPolicy -Scope Process Bypass
.\build_windows.ps1
```

Pascal 세대 GPU(예: Titan Xp)용 빌드는 별도 환경에서 다음과 같이 만듭니다.

```powershell
.\build_windows.ps1 -Gpu
```

RTX 20 시리즈 이상 최신 NVIDIA GPU는 다음 빌드를 사용합니다.

```powershell
.\build_windows.ps1 -ModernGpu
```

일반 빌드는 `dist\Callnote`, GPU 빌드는 `dist-pascal\Callnote` 또는 `dist-modern\Callnote`에 생성됩니다. GPU 빌드에는 CUDA 지원 PyTorch가 포함되므로 용량이 더 큽니다. Pascal GPU는 CUDA 12.6 프로필을 사용하고, 최신 GPU는 CUDA 13.2 프로필을 사용합니다. 모델 가중치는 빌드 산출물이나 GitHub Release에 포함하지 않으며, 최종 사용자 앱이 첫 실행 때 받습니다.

단일 배포 파일은 다음 명령으로 만듭니다. GPU 패키징에는 빌드 PC의 7-Zip 도구가 필요하며 최종 사용자에게는 필요하지 않습니다. GPU 패키지는 압축 무결성과 GitHub의 파일당 2GiB 제한을 확인합니다.

```powershell
.\package_windows.ps1
.\package_windows.ps1 -Profile pascal -SevenZipPath "C:\Program Files\7-Zip\7z.exe"
```

## 상담 분석 내용

- 고객 문의 및 불편 사항, 상담원의 안내와 공감
- 해결 결과와 미해결 항목
- 고객과 상담원의 후속 조치 및 일정
- 감정 흐름과 상담 품질 개선 제안
- 통화 전사 내용에 근거한 다중 행 질문과 답변

모델을 받은 뒤 오디오와 전사 내용은 앱 외부로 전송되지 않습니다. Qwen3 모델의 세부 기능과 Apache-2.0 라이선스 조건은 위 모델 카드에서 확인할 수 있습니다.
