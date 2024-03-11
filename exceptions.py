from tkinter import messagebox

def error_popup(msg=None):
    if msg == None:
        msg = "Uncaught unknown exception has occurred, please view the terminal for details"
    messagebox.showerror("Error", msg)

def warning_popup(msg):
    messagebox.showwarning("Warning", msg)

def warning_prompt(msg):
    return messagebox.askokcancel("Warning", msg)