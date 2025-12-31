import tkinter as tk
from tkinter import ttk
from inf import recommend_jobs
import re

class Screen:
    def __init__(self, encoder, encoders, job_embeddings, df):
        root = tk.Tk()
        root.title("Job Recommender")
        root.geometry("600x500")  # Slightly wider to accommodate results
        
        self.encoder = encoder
        self.encoders = encoders
        self.job_embeddings = job_embeddings
        self.df = df

        # --- Input Section ---
        input_frame = tk.Frame(root, padx=10, pady=10)
        input_frame.pack(fill='x')

        # Job Title
        tk.Label(input_frame, text='Job Title').grid(row=0, column=0, sticky='e', padx=5, pady=2)
        self.Entry1 = tk.Entry(input_frame, width=40)
        self.Entry1.grid(row=0, column=1, sticky='w', pady=2)

        # Category - USE THE ACTUAL CATEGORIES FROM THE ENCODER
        tk.Label(input_frame, text='Category').grid(row=1, column=0, sticky='e', padx=5, pady=2)
        categories = list(self.encoders["category"].categories_[0])  # Get actual categories
        self.combo_box = ttk.Combobox(input_frame, values=categories, width=37, state='readonly')
        self.combo_box.grid(row=1, column=1, sticky='w', pady=2)
        self.combo_box.set(categories[0])  # Default to first category

        # Skills
        tk.Label(input_frame, text='Skills').grid(row=2, column=0, sticky='e', padx=5, pady=2)
        self.Entry3 = tk.Entry(input_frame, width=40)
        self.Entry3.grid(row=2, column=1, sticky='w', pady=2)

        # Description
        tk.Label(input_frame, text='Description').grid(row=3, column=0, sticky='e', padx=5, pady=2)
        self.Entry4 = tk.Entry(input_frame, width=40)
        self.Entry4.grid(row=3, column=1, sticky='w', pady=2)

        # Submit Button
        self.button = tk.Button(input_frame, text='Get Recommendations', command=self.show_input)
        self.button.grid(row=4, column=1, sticky='w', pady=10)

        # --- Results Section ---
        results_frame = tk.Frame(root, padx=10, pady=10)
        results_frame.pack(fill='both', expand=True)

        tk.Label(results_frame, text="Recommendations:", font=('Arial', 10, 'bold')).pack(anchor='w')

        # Create result labels with proper wrapping
        self.label_vars = []
        self.labels = []
        
        for i in range(3):
            var = tk.StringVar()
            var.set("")
            self.label_vars.append(var)
            
            # Frame for each result (adds visual separation)
            result_frame = tk.Frame(results_frame, relief='groove', borderwidth=1, padx=5, pady=5)
            result_frame.pack(fill='x', pady=5)
            
            # Label with wraplength for proper text wrapping
            label = tk.Label(
                result_frame, 
                textvariable=var, 
                wraplength=550,  # Wrap text at this width
                justify='left',
                anchor='w'
            )
            label.pack(fill='x')
            self.labels.append(label)

        # Keep references for backward compatibility with getattr approach
        self.label1_var = self.label_vars[0]
        self.label2_var = self.label_vars[1]
        self.label3_var = self.label_vars[2]

        root.mainloop()

    def show_input(self):
        user_Job_Title = self.Entry1.get()
        user_Category = self.combo_box.get()  # No need for .upper(), already correct format
        user_Skills = self.Entry3.get().strip()
        user_Description = self.Entry4.get()

        # Split skills by comma or semicolon
        patterns = r"[,;]"
        user_Skills = [s.strip() for s in re.split(patterns, user_Skills) if s.strip()]


        reccs = recommend_jobs(
            user_Description,
            user_Skills,
            user_Category,
            user_Job_Title,
            self.encoder,
            self.encoders,
            self.job_embeddings,
            self.df,
            top_n=3
        ).reset_index()

        # Update result labels
        for idx, row in reccs.iterrows():
            result_text = (
                f"{row['category']}: {row['job_title']}\n"
                f"Similarity: {row['similarity_score']:.2%}\n"
                f"{row['job_description'][:150]}..."
            )
            self.label_vars[idx].set(result_text)

    def select(self, event):
        selected_item = self.combo_box.get()
        self.combo_box.set(selected_item)