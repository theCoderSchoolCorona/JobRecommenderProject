import tkinter as tk
from inf import recommend_jobs
import re

class Screen :
    def __init__(self, encoder, encoders, job_embeddings, df):
        root = tk.Tk()

        root.title("Job Recommender")
        root.geometry("500x500")
        # label = tk.Label(root, text="Hello world", font=("Hello world", 16))
        # label.pack(pady=50)
        self.encoder=encoder
        self.encoders=encoders
        self.job_embeddings=job_embeddings
        self.df=df


        self.Label1 = tk.Label(root, text='Job Title').grid(row=0)
        self.Entry1 = tk.Entry(root)
        self.Entry1.grid(row=0, column=1)

        self.Label2 = tk.Label(root, text='Category').grid(row=1)
        self.Entry2 = tk.Entry(root)
        self.Entry2.grid(row=1, column=1)

        self.Label3 = tk.Label(root, text='Skills').grid(row=2)
        self.Entry3 = tk.Entry(root)
        self.Entry3.grid(row=2, column=1)

        self.Label4 = tk.Label(root, text='Description').grid(row=3)
        self.Entry4 = tk.Entry(root)
        self.Entry4.grid(row=3, column=1)

        self.button = tk.Button(root, text='enter', command=self.show_input)
        self.button.grid(row=4, column=1)

        self.label1_var = tk.StringVar()
        self.label1_var.set("")
        self.label1 = tk.Label(root, textvariable=self.label1_var).grid(row=5)

        self.label2_var = tk.StringVar()
        self.label2_var.set("")
        self.label2 = tk.Label(root, textvariable=self.label2_var).grid(row=6)

        self.label3_var = tk.StringVar()
        self.label3_var.set("")
        self.label3 = tk.Label(root, textvariable=self.label4_var).grid(row=7)

        root.mainloop()
    def show_input(self):
        user_Job_Title = self.Entry1.get()
        user_Category = self.Entry2.get().upper()
        user_Skills = self.Entry3.get().strip()
        user_Description = self.Entry4.get()

        patterns = r"[,;]"
        user_Skills = re.split(patterns, user_Skills)

        # self.label1_var.set(user_Job_Title)
        # self.label2_var.set(user_Category)
        # self.label3_var.set(user_Skills)
        # self.label4_var.set(user_Description)
        reccs=recommend_jobs(user_Description,user_Skills,user_Category,user_Job_Title, self.encoder, self.encoders, self.job_embeddings, self.df, top_n=1)
        for idx, row in reccs.iterrows():
            getattr(self, f'label{idx+1}_var').set(f"{row["category"]}: {row["job_title"]} \n {row["job_description"][:100]}")
        # user_Job_Title=reccs['job_title']
        # user_Category=reccs['category']
        # user_Description=reccs['job_description']

        # self.label1_var.set(user_Job_Title)
        # self.label2_var.set(user_Category) 
        # self.label3_var.set(user_Description)

        

    
