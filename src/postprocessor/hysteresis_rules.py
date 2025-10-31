"""
Hysteresis & Rules
- 점수가 한번 올라갔다고 바로 알람으로 가지 않게
- 일정 프레임 이상 유지되면 알람,
  알람 상태에서는 바로 떨어지지 않고 HOLD 후 떨어짐
"""

class HysteresisDetector:
    def __init__(self,
                 th_up: float = 0.55,
                 th_down: float = 0.35,
                 min_run: int = 3,
                 min_hold: int = 5):
        self.th_up = th_up
        self.th_down = th_down
        self.min_run = min_run
        self.min_hold = min_hold

        self.state = "IDLE"
        self.run = 0
        self.hold = 0
        self.alert = False

    def update(self, score: float) -> bool:
        """
        score: 0~1
        return: 이번 프레임에서 '알람'이라고 판단되는지
        """
        if self.state == "IDLE":
            if score >= self.th_up:
                self.state = "CANDIDATE"
                self.run = 1
        elif self.state == "CANDIDATE":
            if score >= self.th_up:
                self.run += 1
                if self.run >= self.min_run:
                    self.state = "ALARM"
                    self.hold = 0
                    self.alert = True
            elif score < self.th_down:
                # 다시 초기화
                self.state = "IDLE"
                self.run = 0
                self.alert = False
        elif self.state == "ALARM":
            self.hold += 1
            if score < self.th_down and self.hold >= self.min_hold:
                self.state = "IDLE"
                self.alert = False
        return self.alert

    def reset(self):
        self.state = "IDLE"
        self.run = 0
        self.hold = 0
        self.alert = False
