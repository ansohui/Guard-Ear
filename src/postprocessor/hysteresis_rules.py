class Hysteresis:
    def __init__(self, enter_th=0.7, exit_th=0.3, hold_frames=3):
        self.enter_th = enter_th
        self.exit_th = exit_th
        self.hold_frames = hold_frames
        self.active = False
        self.counter = 0

    def update(self, is_high):
        """
        is_high: 이번 프레임에서 모델이 '높다'고 본지 여부
        return: 최종적으로 이번 프레임을 siren으로 간주할지
        """
        if not self.active:
            if is_high:
                self.counter += 1
                if self.counter >= self.hold_frames:
                    self.active = True
            else:
                self.counter = 0
        else:
            if not is_high and self.counter <= 0:
                self.active = False
            else:
                self.counter -= 1

        return self.active
