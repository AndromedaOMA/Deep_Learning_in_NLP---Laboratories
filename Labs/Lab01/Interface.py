import tkinter as tk
import random
from generate_words import generate_words
from tkinter import messagebox

# Am facut un random list insa cuvintele sunt prea complicate, cred ca putem hardcoda
# WORDS = generate_words(on_event="Start")
WORDS = ["Cop", "Library", "Face", "Electricity", "Frozen", "Ship", "Dinosaur", "Museum", "Sound", "Sun"]


class Interface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WordNet Game")
        self.geometry("600x700")
        self.configure(bg="#0a0014")

        self.running = False
        self.game_over = False
        self.timer_delay = 5000

        self.current_word = tk.StringVar(value=WORDS[0])

        self.prompt = tk.Label(
            self,
            text='Type the first thing that comes to mind\nwhen you think of:',
            fg="#9a89ff",
            bg="#0a0014",
            font=("Consolas", 14),
        )
        self.prompt.pack(pady=(40, 5))

        self.word_label = tk.Label(
            self,
            textvariable=self.current_word,
            fg="#00ffff",
            bg="#0a0014",
            font=("Consolas", 24, "bold")
        )
        self.word_label.pack(pady=(0, 30))

        self.entry = tk.Entry(
            self,
            width=40,
            fg="#00ff99",
            bg="#1a0033",
            insertbackground="#00ff99",
            font=("Consolas", 14)
        )
        self.entry.pack()
        self.entry.bind("<Return>", self.submit_word)
        self.entry.focus()

        frame = tk.Frame(self, bg="#0a0014")
        frame.pack(pady=(30, 0))

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.word_list = tk.Listbox(
            frame,
            width=40,
            height=8,
            fg="#aaccff",
            bg="#0a0014",
            font=("Consolas", 13),
            highlightthickness=0,
            borderwidth=0,
            activestyle="none",
            yscrollcommand=scrollbar.set
        )
        for w in WORDS[:1]:
            self.word_list.insert(tk.END, w)
        self.word_list.pack(side=tk.LEFT, fill=tk.BOTH)
        scrollbar.config(command=self.word_list.yview)

        self.size = self.word_list.size()

    def start_timer(self):
        if not self.running:
            self.running = True
            self.increase_list()

    def increase_list(self):
        if self.game_over:
            return
        self.word_list.insert(tk.END, random.choice(WORDS))
        self.word_list.see(tk.END)

        self.size += 1
        if self.size >= 20:
            self.end_game()
            return

        self.after(self.timer_delay, self.increase_list)

    def submit_word(self, event=None):
        user_word = self.entry.get().strip()
        if not user_word or self.game_over:
            return

        if not self.running:
            self.start_timer()

        self.entry.delete(0, tk.END)

        if self.word_list.size() > 0:
            self.word_list.delete(0)
            self.size -= 1

        new_word = random.choice(WORDS)
        self.word_list.insert(tk.END, new_word)
        self.word_list.see(tk.END)

        if self.word_list.size() > 0:
            next_target = self.word_list.get(0)
            self.current_word.set(next_target)

    def end_game(self):
        self.game_over = True
        messagebox.showinfo("Game Over", "Too many words! Game over.")
        self.destroy()


if __name__ == "__main__":
    app = Interface()
    app.mainloop()
