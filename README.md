# Callnote · 통화 인사이트

Windows 데스크톱 앱으로 WAV 고객센터 상담을 분석합니다. 음성 인식은 **OpenAI 원본 Whisper 구현의 `large-v3` 체크포인트**를 사용하며, `faster-whisper`는 사용하지 않습니다. 상담 요약과 전사 질의응답은 로컬 [Qwen3 4B Instruct 2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) 모델이 처리합니다. 이 공개 모델은 Apache-2.0 라이선스를 사용합니다.

## 다운로드 및 실행

GitHub Releases에서 CPU용 또는 Titan Xp 등 Pascal GPU용 패키지를 내려받아 압축을 푼 뒤 `Callnote\Callnote.exe`를 실행합니다. Python, PyTorch, FFmpeg, 모델을 별도로 설치할 필요가 없습니다. 앱의 코드와 실행 종속성은 패키지에 포함되어 있습니다. GPU용 분할 패키지는 모든 파트를 같은 위치에 압축 해제해야 합니다.

첫 실행에는 앱이 모델을 자동으로 다운로드합니다. Whisper large-v3와 Qwen3 모델 합계 약 10.9GB를 받기 위해 인터넷 연결과 저장 공간이 필요합니다. 내려받은 모델은 `%LOCALAPPDATA%\Callnote\models`에 저장되고, 이후 통화 분석은 오프라인으로 실행됩니다. 다운로드가 중단되면 앱을 다시 열고 **모델 다시 받기**를 누르세요. NVIDIA GPU용 빌드는 GPU를 자동 사용하고, GPU가 없으면 CPU로 동작합니다. 일반 빌드는 CPU 전용입니다.

## 개발 및 빌드

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

## 상담 분석 내용

- 고객 문의 및 불편 사항, 상담원의 안내와 공감
- 해결 결과와 미해결 항목
- 고객과 상담원의 후속 조치 및 일정
- 감정 흐름과 상담 품질 개선 제안
- 통화 전사 내용에 근거한 다중 행 질문과 답변

모델을 받은 뒤 오디오와 전사 내용은 앱 외부로 전송되지 않습니다. Qwen3 모델의 세부 기능과 Apache-2.0 라이선스 조건은 위 모델 카드에서 확인할 수 있습니다.
