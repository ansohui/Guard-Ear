# Siren Detection Prototype


## Pipeline
1. Frame Buffer (0.96s collect, 0.48s hop, 50% overlap)
2. Preprocessor
   - 16kHz mono 변환
   - Rule-based pre-filter (freq, dB)
3. Detection
   - YamNet (521 classes prob)
   - MLP Head (2 layers) -> siren prob
4. Postprocessor
   - Hysteresis & Rules
5. Server
   - API -> Notification

## 환경변수 예시
SIREN_SMTP_HOST=smtp.gmail.com   
SIREN_SMTP_PORT=587   
SIREN_SMTP_USER=your_account@gmail.com   
SIREN_SMTP_PASS=your_app_password   
SIREN_FROM=your_account@gmail.com   
SIREN_TO=receiver@example.com   

## Run
```bash
pip install -r requirements.txt
python src/main.py


