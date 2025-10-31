"""
notification.py

메일/슬랙/웹훅 등 알림 채널을 묶어두는 모듈.
현재는 SMTP 기반 이메일 전송만 구현.
환경변수에서 메일 설정을 읽어오도록 해서 깃 공개 시에도 안전하게 유지한다.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from typing import Optional


class MailNotifier:
    def __init__(self):
        # 환경변수에서 설정 읽기
        self.smtp_host = os.getenv("SIREN_SMTP_HOST", "")
        self.smtp_port = int(os.getenv("SIREN_SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SIREN_SMTP_USER", "")
        self.smtp_pass = os.getenv("SIREN_SMTP_PASS", "")
        self.from_addr = os.getenv("SIREN_FROM", self.smtp_user)
        self.to_addr = os.getenv("SIREN_TO", "")

    def is_configured(self) -> bool:
        # 최소한의 설정이 있는지 확인
        return bool(self.smtp_host and self.smtp_user and self.smtp_pass and self.to_addr)

    def send_alert_mail(
        self,
        subject: str,
        body: str,
        to: Optional[str] = None,
    ) -> bool:
        """
        간단한 텍스트 메일 전송.
        설정이 안 돼 있으면 False만 돌려서 상위에서 로깅하게 한다.
        """
        if not self.is_configured():
            
            print("[MailNotifier] SMTP 설정이 없어 메일을 전송하지 않습니다.")
            return False

        to_addr = to or self.to_addr

        msg = MIMEText(body, _charset="utf-8")
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"] = self.from_addr
        msg["To"] = to_addr

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            print(f"[MailNotifier] 메일 전송 완료 → {to_addr}")
            return True
        except Exception as e:
            print(f"[MailNotifier] 메일 전송 실패: {e}")
            return False
