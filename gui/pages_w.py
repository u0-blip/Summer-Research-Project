import tkinter as tk

class Page(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
    def show(self):
        self.lift()

class Page1(Page):
   def __init__(self, *args, **kwargs):
       Page.__init__(self, *args, **kwargs)
       label = tk.Label(self, text="This is page 1")
       label.grid(row = 5)

class Page2(Page):
   def __init__(self, *args, **kwargs):
       Page.__init__(self, *args, **kwargs)
       label = tk.Label(self, text="This is page 2")
       label.grid(row = 0)

class Page3(Page):
   def __init__(self, *args, **kwargs):
       Page.__init__(self, *args, **kwargs)
       label = tk.Label(self, text="This is page 3")
       label.grid(row = 0)

class MainView(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        buttonframe = tk.Frame(self)
        container = tk.Frame(self)
        buttonframe.grid(row = 0)
        container.grid(row = 10, rowspan=10, columnspan=10)

        p1 = Page1(container)
        # p2 = Page2(container)
        # p3 = Page3(container)

        p1.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        # p2.place(in_=container, x=0, y=10, relwidth=1, relheight=1)
        # p3.place(in_=container, x=0, y=10, relwidth=1, relheight=1)

        b1 = tk.Button(buttonframe, text="Page 1")
        b2 = tk.Button(buttonframe, text="Page 2")
        b3 = tk.Button(buttonframe, text="Page 3")


        label = tk.Label(buttonframe, text="This is buttonframe")
        label.grid(row = 0)

        label = tk.Label(container, text="This is container")
        label.grid(row = 5)

        b1.grid(row = 0, column = 0)
        b2.grid(row = 0, column = 1)
        b3.grid(row = 0, column = 2)

        # p1.show()

if __name__ == "__main__":
    root = tk.Tk()
    main = MainView(root)
    main.grid()
    root.wm_geometry("400x400+800+300")
    root.mainloop()