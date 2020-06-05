def g():
    b = 1
    a = [b,2, 3, 4]

    class temp:
        def __init__(self):
            self.a = a

    def f(arr):
        arr[0] += 1

    d = temp()
    print(d.a)
g()