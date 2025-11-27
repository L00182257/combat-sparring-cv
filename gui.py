import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import sys

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from count_video import process_video


class PunchCounterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Punch Counter")
        self.root.geometry("500x300")

        self.video_path = None

        # Title
        self.label_title = tk.Label(root, text="Punch Counter", font=("Arial", 20))
        self.label_title.pack(pady=10)

        # Video path label
        self.label_file = tk.Label(root, text="No video selected", fg="gray")
        self.label_file.pack(pady=5)

        # Select video button
        self.btn_select = tk.Button(root, text="Select Video", command=self.choose_file)
        self.btn_select.pack(pady=5)

        # Run button
        self.btn_run = tk.Button(root, text="Run Punch Counter", command=self.run_pipeline)
        self.btn_run.pack(pady=10)

        # Status box
        self.status_text = tk.Text(root, height=8, width=55, state="disabled")
        self.status_text.pack(pady=5)

    def choose_file(self):
        file_path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video Files", "*.mp4 *.mov *.avi")]
        )
        if file_path:
            self.video_path = file_path
            self.label_file.config(text=os.path.basename(file_path), fg="black")

    def log(self, message):
        self.status_text.config(state="normal")
        self.status_text.insert("end", message + "\n")
        self.status_text.config(state="disabled")
        self.status_text.see("end")
        self.root.update()

    def run_pipeline(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first.")
            return

        # Run in separate thread so GUI stays responsive
        thread = threading.Thread(target=self._process_video_thread)
        thread.start()

    def _process_video_thread(self):
        try:
            self.log("Running punch counter...")
            results = process_video(self.video_path, target_fps=5)

            self.log("\n--- RESULTS ---")
            self.log(f"Total punches: {results['total_punches']}")
            self.log(f"Left punches: {results['left_punches']}")
            self.log(f"Right punches: {results['right_punches']}")

            # Open the visualized video file automatically
            viz_path = os.path.join(
                "data", "results",
                os.path.splitext(os.path.basename(self.video_path))[0] + "_visualized.mp4"
            )

            if os.path.exists(viz_path):
                self.log(f"\nOpening visualization: {viz_path}")
                os.startfile(viz_path)
            else:
                self.log("Visualization file not found.")

            messagebox.showinfo("Done", "Punch counting complete!")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.log(f"\nERROR: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PunchCounterGUI(root)
    root.mainloop()