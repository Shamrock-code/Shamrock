class ContactDiscontinuity:
    def __init__(self, speed, pressure):
        self.speed = speed
        self.pressure = pressure

    def __str__(self):
        return f"Contact Discontinuity: v={self.speed:.3f}, pressure={self.pressure:.3f}"

    def __repr__(self):
        return str(self)
