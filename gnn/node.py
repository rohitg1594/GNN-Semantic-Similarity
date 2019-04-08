class Node:
    def __init__(self, name, id, is_main):
        self.name = name
        self.id = id
        self.is_main = is_main

    def __repr__(self):
        return self.name
