class averager:
    def __init__(self, max_len=20):
        self.max_len = max_len
        self.nums = []

    def clear(self):
        self.nums.clear()

    def append(self, num):
        while self.max_len != -1 and len(self.nums) >= self.max_len:
            self.pop()
        self.nums.append(num)

    def pop(self):
        self.nums.pop(0)

    def get_avg(self):
        if len(self.nums) < 1:
            return 0.0
        return sum(self.nums) * 1.0 / len(self.nums)