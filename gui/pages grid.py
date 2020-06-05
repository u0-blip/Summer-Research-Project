import tkinter as tk


pi = 3.1415926

class Page(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

    def show(self):
        self.lift()

class geo(Page):
    sweep_fields = {
        'shape' : [3, 3, 1],
        'particle_size' : [0.05, 0.12, 5],
        'x_loc' : [0, -3.4, 1],
        'distance' : [1, 3, 1],
        'fill_factor' : [0.5, 0.7, 5],
        'std' : [0.1, 0.3, 1]
    }

    fields = {
        'rotation' : 7*pi/6,
        'pml_thick' : 0.5,
        'solid_center' : [-2, 0, 0],
        'shape_types' : ['sphere', 'triangle', 'hexagon', 'cube'],
        'num_particles' : 2,
        'cell_size':  [10, 10, 10],
    }
    master_row = 0

    def __init__(self, *args, **kwargs):
        self.distri_type = tk.StringVar()
        Page.__init__(self, *args, **kwargs)
        label = tk.Label(self, text='Particle size distribution type')
        label.grid(row = self.master_row)
        self.master_row += 1
        self.size_distri_btn()

        self.entry = []
        # for k, v in self.sweep_fields.items():  
        self.sweep_fields_input('k', [1, 2, 3])

    def sweep_fields_input(self, field, default):
        vcmd = self.register(self.validate) # we have to wrap the command
        for i in range(3):
            self.entry.append(tk.Entry(self, validate="key", validatecommand=(vcmd, '%P')))
            self.entry[-1].insert(0, field)
            if i == 2:
                self.entry[-1].grid(row = self.master_row)
                self.master_row += 1
            else:
                self.entry[-1].grid(row = self.master_row)
                self.master_row += 1


    def validate(self, new_text):
        if not new_text: # the field is being cleared
            self.guess = None
            return True

        try:
            guess = int(new_text)
            if 1 <= guess <= 100:
                self.guess = guess
                return True
            else:
                return False
        except ValueError:
            return False


    def size_distri_btn(self):
        distri_types = ['fixed', 'gaussian']

        R1 = tk.Radiobutton(self, text=distri_types[0], variable=self.distri_type, value=distri_types[0], command=self.sel)
        R1.grid(row = self.master_row)
        self.master_row += 1

        R2 = tk.Radiobutton(self, text=distri_types[1], variable=self.distri_type, value=distri_types[1], command=self.sel)
        R2.grid(row = self.master_row)
        self.master_row += 1

        label = tk.Label(self)
        label.grid(row = self.master_row)
        self.master_row += 1

    def sel(self):
        print(self.distri_type.get())

class sim(Page):
    master_row = 5
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        label = tk.Label(self, text="This is page 2")
        label.grid(row = 5)
        self.master_row += 1 

class general(Page):
    master_row = 0
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        label = tk.Label(self, text="This is page 3")
        label.grid(row = self.master_row)
        self.master_row += 1

class MainView(tk.Frame):
    master_row = 0
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        p1 = sim(self)
        p2 = sim(self)
        p3 = general(self)

        buttonframe = tk.Frame(self)
        container = tk.Frame(self)
        buttonframe.grid(row = self.master_row)
        btn_row = self.master_row
        self.master_row += 1
        container.grid(row = 10)
        self.master_row += 1

        p1.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p2.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p3.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        b1 = tk.Button(container, text="geometry", command=p1.lift)
        b2 = tk.Button(buttonframe, text="simulation", command=p2.lift)
        b3 = tk.Button(buttonframe, text="general process", command=p3.lift)

        b1.grid(row = 0, column = 0)
        b2.grid(row = 0, column = 1)
        b3.grid(row = 0, column = 2)
        p1.show()

if __name__ == "__main__":
    root = tk.Tk()
    main = MainView(root)
    main.grid(row = 0)
    root.wm_geometry("400x400+800+300")
    root.mainloop()
