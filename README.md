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

## Run
```bash
pip install -r requirements.txt
python src/main.py
