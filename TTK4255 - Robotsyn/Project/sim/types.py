class Intrinsics:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    @classmethod
    def from_file(cls, source_file):     
        with open(source_file) as fh:
            for line in fh:
                words = line.split()
                if not words:
                    break
                if words[0] == "fx":
                    fx = float(words[2])
                elif words[0] == "fy":
                    fy = float(words[2])
                elif words[0] == "cx":
                    cx = float(words[2])
                elif words[0] == "cy":
                    cy = float(words[2])
        return cls(fx, fy, cx, cy)

    def __str__(self):
        return "fx: {}\nfy: {}\ncx: {}\ncy: {}".format(self.fx, self.fy, self.cx, self.cy)