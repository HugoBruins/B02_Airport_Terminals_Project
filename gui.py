import tkinter as tk
import threading

import dummy


class GUI:
    def __init__(self):
        self.data = [None, None, None, None, None]
        self.window = self.create_window()
        self.fill_window()
        self.window.mainloop()

    def create_window(self):
        window = tk.Tk()
        window.config(bg="LightBlue")
        window.minsize(300, 300) # Set the window size
        window.columnconfigure(0, weight=1)
        for i in range(9):
            window.rowconfigure(i, weight=1)
        window.title("B02 RF Model for predicting airport performance")
        return window

    def fill_window(self):
        input_1 = tk.Entry(self.window)
        input_1.insert(0, "Enter <INPUT1> here")
        input_1.bind("<FocusIn>", lambda event: input_1.delete(0, 'end'))
        input_1.grid(column=0, row=0, sticky="SEW")

        input_2 = tk.Entry(self.window)
        input_2.insert(0, "Enter <INPUT2> here")
        input_2.bind("<FocusIn>", lambda event: input_2.delete(0, 'end'))
        input_2.grid(column=0, row=1, sticky="SEW")

        button_1 = tk.Button(self.window, text="Import Data", command=lambda: self.run_function(dummy.main2, button_1, [input_1.get(), input_2.get()]))
        button_1.grid(column=0, row=3, sticky="SEW")
        button_2 = tk.Button(self.window, text="Process Data", state="disabled", command=lambda f=dummy.main: self.run_function(f, button_2, self.data[1]))
        button_2.grid(column=0, row=4, sticky="SEW")
        button_3 = tk.Button(self.window, text="Train Model", state="disabled", command=lambda f=dummy.main: self.run_function(f, button_3, self.data[2]))
        button_3.grid(column=0, row=5, sticky="SEW")
        button_4 = tk.Button(self.window, text="Run Model", state="disabled", command=lambda f=dummy.main: self.run_function(f, button_4, self.data[3]))
        button_4.grid(column=0, row=6, sticky="SEW")
        button_5 = tk.Button(self.window, text="Plot Results", state="disabled", command=lambda f=dummy.main: self.run_function(f, button_5, self.data[4]))
        button_5.grid(column=0, row=7, sticky="SEW")

        self.buttons = [button_1, button_2, button_3, button_4, button_5]


        self.results_label = tk.Label(self.window, text="Results:\n\nWaiting for results...")
        self.results_label.grid(column=0, row=8, sticky="S")

    def run_function(self, function, button, input):
        button.configure(bg="WHITE")
        for item in self.buttons:
            item.configure(state="disabled")
        for item in self.buttons[self.buttons.index(button) + 1::]:
            item.configure(bg="WHITE")
        running_function_thread = threading.Thread(target=self.start_function, args=(function, input, button.cget("text")))
        running_function_thread.start()

        listen_for_completion_thread = threading.Thread(target=self.listen_for_completion, args=(running_function_thread, button))
        listen_for_completion_thread.start()

    def start_function(self, function, input, name):
        if input:
            self.result = (name, function(input))
        else:
            self.result = (name, function())

    def listen_for_completion(self, thread, button):
        thread.join()
        button.configure(bg="GREEN")
        for item in self.buttons[:self.buttons.index(button)+2:]:
            item.configure(state="normal")
        self.results_label.configure(text=f"Results:\n\nFunction name: {self.result[0]}\n{self.result[1]}")
        self.data[self.buttons.index(button)] = self.result[1]

GUI()