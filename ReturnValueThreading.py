class Queue(object):
    def __init__(self):
        self.item = []
        
    def __str__(self):
        return "{}".format(self.item)

    def __repr__(self):
        return "{}".format(self.item)

    def enque(self, item):
        self.item.insert(0, item)
        return True

    def size(self):
        return len(self.item)

    def dequeue(self):
        if self.size() == 0:
            return None
        else:
            return self.item.pop()

    def peek(self):
        if self.size() == 0:
            return None
        else:
            return self.item[-1]

    def isEmpty(self):
        if self.size() == 0:
            return True
        else:
            return False
