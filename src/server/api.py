import datetime
from typing import Dict, Any

from src.server.notification import MailNotifier


_notifier = MailNotifier()


def notify_alert(payload: Dict[str, Any]) -> None:
    """
    시그널 감지 결과를 받아서 메일로 전송.
    payload 예시:
    {
        "file": "siren_001.wav",
        "p_rule": 0.53,
        "p_ml": 0.74,
        "p_fused": 0.68,
        "alert": True,
        "last_state": "ALARM"
    }
    """
    # 메일 제목/본문 구성
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"[SIREN-PROTOTYPE] 경보 감지 ({ts})"

    body_lines = [
        "SIREN-PROTOTYPE에서 경보 후보 신호를 감지했습니다.",
        "",
        f"- 파일/소스: {payload.get('file', 'N/A')}",
        f"- rule score: {payload.get('p_rule', 'N/A')}",
        f"- ml score: {payload.get('p_ml', 'N/A')}",
        f"- fused score: {payload.get('p_fused', 'N/A')}",
        f"- 최종 상태: {payload.get('last_state', 'N/A')}",
        f"- 경보 여부: {payload.get('alert', False)}",
        "",
        f"발생 시각: {ts}",
    ]
    body = "\n".join(body_lines)

    sent = _notifier.send_alert_mail(subject, body)
    if not sent:
        print("[API] 메일을 보낼 수 없었습니다. SMTP 설정을 확인하세요.")
