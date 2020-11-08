import sys
import os

sys.path.append(os.getcwd())
if __name__ == "__main__":
    if os.name == 'nt':
        from my_meep.win_main import win_main
        win_main()
    elif os.name == 'posix':
        from my_meep.wsl_main import wsl_main
        for none in wsl_main():
            pass
        