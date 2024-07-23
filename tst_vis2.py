import tkinter as tk
from tkinter import messagebox

def on_button_click():
    label.config(text="Hello, Tkinter!")
    messagebox.showinfo("Information", "Button was clicked!")

# 创建主窗口
root = tk.Tk()
root.title("Simple Tkinter GUI")

# 设置窗口的大小和位置
root.geometry('300x200+100+100')  # 宽 x 高 + X偏移 + Y偏移

# 创建一个标签组件
label = tk.Label(root, text="Press the button", font=('Arial', 12))
label.pack(pady=20)  # 添加到主窗口并垂直外边距

# 创建一个按钮组件，点击时调用 on_button_click 函数
button = tk.Button(root, text="Click Me", command=on_button_click)
button.pack(pady=20)

# 进入主事件循环
root.mainloop()
