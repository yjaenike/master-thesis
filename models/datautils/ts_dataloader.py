class TSDataloader():
    def __init__(self, dataset, batch_size, drop_last, num_workers, pin_memory):
        self.dataset = 