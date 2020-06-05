# from tkinter import filedialog
# from tkinter import *
# root = Tk()
# root.withdraw()
# folder_selected = filedialog.askopenfiles()
# print(folder_selected) 

import matplotlib.pyplot as plt

plt.figure()
plt.subplot(2,1,1)
plt.plot([1,2,3])
plt.subplot(2,1,2)
plt.plot([4,5,6])
plt.subplots_adjust(hspace = 0.5)
plt.show()