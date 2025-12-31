import tkinter as tk


root = tk.Tk()

root.title("Job Recommender")
root.geometry("500x500")
# label = tk.Label(root, text="Hello world", font=("Hello world", 16))
# label.pack(pady=50)

Label1 = tk.Label(root, text='Job Title').grid(row=0)
Entry1 = tk.Entry(root)
Entry1.grid(row=0, column=1)

Label2 = tk.Label(root, text='Category').grid(row=1)
Entry2 = tk.Entry(root)
Entry2.grid(row=1, column=1)

Label3 = tk.Label(root, text='Skills').grid(row=2)
Entry3 = tk.Entry(root)
Entry3.grid(row=2, column=1)

Label4 = tk.Label(root, text='Description').grid(row=3)
Entry4 = tk.Entry(root)
Entry4.grid(row=3, column=1)

def show_input():
    user_Job_Title = Entry1.get()
    user_Category = Entry2.get()
    user_Skills = Entry3.get()
    user_Description = Entry4.get()
    print(user_Job_Title)
    print(user_Category)
    print(user_Skills)
    print(user_Description)

    label1_var.set(user_Job_Title)
    label2_var.set(user_Category)
    label3_var.set(user_Skills)
    label4_var.set(user_Description)

button = tk.Button(root, text='enter', command=show_input)
button.grid(row=4, column=1)

label1_var = tk.StringVar()
label1_var.set("Job title")
label1 = tk.Label(root, textvariable=label1_var).grid(row=5)

label2_var = tk.StringVar()
label2_var.set("Category")
label2 = tk.Label(root, textvariable=label2_var).grid(row=6)

label3_var = tk.StringVar()
label3_var.set("Skills")
label3 = tk.Label(root, textvariable=label3_var).grid(row=7)

label4_var = tk.StringVar()
label4_var.set("Description")
label4 = tk.Label(root, textvariable=label4_var).grid(row=8)

root.mainloop()