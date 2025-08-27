# app.py
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, ttk
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import ListedColormap
import torch

from mask_propagation.mask_component import MaskComponent


def show_mask(mask, ax, obj_id=None, random_color=False, cmap="Paired", alpha=0.7):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        cmap_instance = plt.get_cmap(cmap)
        color = np.array([*cmap_instance(obj_id if obj_id is not None else 0)[:3], alpha])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


class SettingsWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master.root)
        self.master_app = master
        self.title("Settings")
        self.transient(master.root)
        self.grab_set()

        self.model_checkpoint_var = tk.StringVar(value=self.master_app.model_checkpoint)
        self.model_cfg_var = tk.StringVar(value=self.master_app.model_cfg)
        self.point_color_var = tk.StringVar(value=self.master_app.point_color)
        self.box_color_var = tk.StringVar(value=self.master_app.box_color)

        self._setup_widgets()

    def _setup_widgets(self):
        main_frame = tk.Frame(self, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Model paths
        path_frame = tk.LabelFrame(main_frame, text="Model Paths")
        path_frame.pack(fill=tk.X, pady=5)

        tk.Label(path_frame, text="Checkpoint (.pt):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(path_frame, textvariable=self.model_checkpoint_var, width=60).grid(row=0, column=1, sticky="ew")
        tk.Button(path_frame, text="...", command=lambda: self.browse_file(self.model_checkpoint_var)).grid(row=0, column=2, padx=5)

        tk.Label(path_frame, text="Config (.yaml):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(path_frame, textvariable=self.model_cfg_var, width=60).grid(row=1, column=1, sticky="ew")
        tk.Button(path_frame, text="...", command=lambda: self.browse_file(self.model_cfg_var)).grid(row=1, column=2, padx=5)
        path_frame.columnconfigure(1, weight=1)

        # Prompt colors
        color_frame = tk.LabelFrame(main_frame, text="Prompt Colors")
        color_frame.pack(fill=tk.X, pady=5)

        tk.Label(color_frame, text="Point Color:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.point_color_label = tk.Label(color_frame, text="          ", bg=self.point_color_var.get())
        self.point_color_label.grid(row=0, column=1, sticky="w")
        tk.Button(color_frame, text="Choose...", command=lambda: self.choose_color(self.point_color_var, self.point_color_label)).grid(row=0, column=2, padx=5)

        tk.Label(color_frame, text="Box Color:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.box_color_label = tk.Label(color_frame, text="          ", bg=self.box_color_var.get())
        self.box_color_label.grid(row=1, column=1, sticky="w")
        tk.Button(color_frame, text="Choose...", command=lambda: self.choose_color(self.box_color_var, self.box_color_label)).grid(row=1, column=2, padx=5)

        # Actions
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        tk.Button(button_frame, text="Save & Close", command=self.save_and_close).pack(side=tk.RIGHT)
        tk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def browse_file(self, var):
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def choose_color(self, var, label):
        color_code = colorchooser.askcolor(title="Choose color", initialcolor=var.get())
        if color_code and color_code[1]:
            var.set(color_code[1])
            label.config(bg=color_code[1])

    def save_and_close(self):
        self.master_app.model_checkpoint = self.model_checkpoint_var.get()
        self.master_app.model_cfg = self.model_cfg_var.get()
        self.master_app.point_color = self.point_color_var.get()
        self.master_app.box_color = self.box_color_var.get()

        self.master_app._save_settings()

        messagebox.showinfo(
            "Settings Saved",
            "Settings updated. Re-initialize to apply model path changes.",
            parent=self
        )
        self.destroy()


class PromptWindow(tk.Toplevel):
    def __init__(self, master, frame_idx, image):
        super().__init__(master.root)
        self.master_app = master
        self.frame_idx = frame_idx
        self.image = image
        self.title(f"Annotating Frame {frame_idx}")

        self.point_coords = []
        self.box_coords = None
        self.drawn_patches = []

        self.point_color = self.master_app.point_color
        self.box_color = self.master_app.box_color
        self.rect_selector = None
        self.point_cid = None

        self._setup_widgets()
        self._setup_canvas()
        self.switch_prompt_mode()

    def _setup_widgets(self):
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(top_frame, text="Particle ID:").pack(side=tk.LEFT)
        self.particle_id_entry = tk.Entry(top_frame, width=10)
        self.particle_id_entry.pack(side=tk.LEFT, padx=5)

        self.prompt_type = tk.StringVar(value="box")
        tk.Radiobutton(top_frame, text="Box Prompt", variable=self.prompt_type, value="box",
                       command=self.switch_prompt_mode).pack(side=tk.LEFT)
        tk.Radiobutton(top_frame, text="Point Prompt", variable=self.prompt_type, value="points",
                       command=self.switch_prompt_mode).pack(side=tk.LEFT)

        self.done_button = tk.Button(top_frame, text="Generate Mask", command=self.on_done_click)
        self.done_button.pack(side=tk.LEFT, padx=5)

        tk.Button(top_frame, text="Close & Save Prompts", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _setup_canvas(self):
        fig, self.ax = plt.subplots()
        self.ax.imshow(self.image, cmap='gray')
        self.canvas_widget = FigureCanvasTkAgg(fig, master=self)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        bottom_frame = tk.Frame(self)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        tk.Label(bottom_frame, text="X:").pack(side=tk.LEFT)
        self.x_entry = tk.Entry(bottom_frame, width=8)
        self.x_entry.pack(side=tk.LEFT)
        tk.Label(bottom_frame, text="Y:").pack(side=tk.LEFT)
        self.y_entry = tk.Entry(bottom_frame, width=8)
        self.y_entry.pack(side=tk.LEFT)
        tk.Button(bottom_frame, text="Add Point", command=self.add_manual_coord).pack(side=tk.LEFT, padx=5)
        tk.Button(bottom_frame, text="Clear Last", command=self.clear_last_prompt).pack(side=tk.LEFT, padx=5)

    def switch_prompt_mode(self):
        if self.prompt_type.get() == 'points':
            if self.rect_selector:
                self.rect_selector.set_active(False)
            self.point_cid = self.canvas_widget.figure.canvas.mpl_connect('button_press_event', self.on_point_click)
        else:
            if self.point_cid:
                self.canvas_widget.figure.canvas.mpl_disconnect(self.point_cid)
                self.point_cid = None
            if not self.rect_selector:
                self.rect_selector = RectangleSelector(
                    self.ax, self.on_box_select, useblit=True, button=[1],
                    minspanx=5, minspany=5, spancoords='pixels', interactive=True,
                    props=dict(facecolor='none', edgecolor=self.box_color, linewidth=1.5)
                )
            self.rect_selector.set_active(True)

    def on_point_click(self, event):
        if event.inaxes != self.ax:
            return
        self.point_coords.append((event.xdata, event.ydata))
        point = self.ax.plot(event.xdata, event.ydata, 'x', color=self.point_color, markersize=8)[0]
        self.drawn_patches.append(point)
        self.canvas_widget.draw()

    def on_box_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.box_coords = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def add_manual_coord(self):
        try:
            x = float(self.x_entry.get()); y = float(self.y_entry.get())
            self.point_coords.append((x, y))
            point = self.ax.plot(x, y, 'x', color=self.point_color, markersize=8)[0]
            self.drawn_patches.append(point)

            if self.prompt_type.get() == 'box' and len(self.point_coords) >= 2:
                p1, p2 = self.point_coords[-2], self.point_coords[-1]
                self.box_coords = [
                    min(p1[0], p2[0]), min(p1[1], p2[1]),
                    max(p1[0], p2[0]), max(p1[1], p2[1])
                ]
                rect = Rectangle(
                    (self.box_coords[0], self.box_coords[1]),
                    self.box_coords[2] - self.box_coords[0],
                    self.box_coords[3] - self.box_coords[1],
                    fill=False, edgecolor=self.box_color, linewidth=1.5
                )
                self.ax.add_patch(rect)
                self.drawn_patches.append(rect)
                self.point_coords.clear()

            self.canvas_widget.draw()
            self.x_entry.delete(0, tk.END); self.y_entry.delete(0, tk.END)
            self.x_entry.focus_set()
        except (ValueError, IndexError):
            messagebox.showerror("Input Error", "Please enter valid numeric coordinates.", parent=self)

    def clear_last_prompt(self):
        if not self.drawn_patches:
            return
        last_patch = self.drawn_patches.pop()
        last_patch.remove()

        if isinstance(last_patch, Rectangle):
            self.box_coords = None
            if self.drawn_patches:
                last_point_patch = self.drawn_patches.pop()
                last_point_patch.remove()
            if self.point_coords:
                self.point_coords.pop()
        elif self.point_coords:
            self.point_coords.pop()

        self.canvas_widget.draw()

    def on_done_click(self):
        try:
            particle_id = int(self.particle_id_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Particle ID must be an integer.", parent=self)
            return

        self.done_button.config(text="Generating...", state=tk.DISABLED)
        self.update_idletasks()

        try:
            if self.prompt_type.get() == "box" and self.box_coords:
                obj_ids, masks = self.master_app.backend.add_box(self.frame_idx, particle_id, self.box_coords)
                self.box_coords = None
            elif self.prompt_type.get() == "points" and self.point_coords:
                points = np.array(self.point_coords, dtype=np.float32)
                labels = np.ones(len(self.point_coords), dtype=np.float32)
                obj_ids, masks = self.master_app.backend.add_points(self.frame_idx, particle_id, points, labels)
                self.point_coords = []
            else:
                messagebox.showwarning("Warning", "No prompt to generate mask from.", parent=self)
                self.done_button.config(text="Generate Mask", state=tk.NORMAL)
                return
        except Exception as e:
            messagebox.showerror("Mask Error", f"Failed to generate mask:\n{e}", parent=self)
            self.done_button.config(text="Generate Mask", state=tk.NORMAL)
            return

        # mark that prompts were added
        self.master_app.prompts_added = True

        # clear drawn prompts
        for p in self.drawn_patches:
            p.remove()
        self.drawn_patches.clear()

        if self.rect_selector:
            self.rect_selector.extents = (0, 0, 0, 0)

        # visualize masks
        for obj_id, m in zip(obj_ids, masks):
            show_mask(np.squeeze(m), self.ax, obj_id=obj_id, cmap=self.master_app.cmap_custom)

        self.canvas_widget.draw()
        self.done_button.config(text="Generate Mask", state=tk.NORMAL)


class MaskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM-EM Mask Generation - Main")

        self.APP_DIR = Path(__file__).resolve().parent
        self.SETTINGS_PATH = self.APP_DIR / ".sam_em" / "settings.json"
        self.SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Config defaults (user can change in Settings)
        self.model_checkpoint = "../checkpoints/finetuned_sam2.1.pt"
        self.model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        self.point_color = 'red'
        self.box_color = 'green'

        # Runtime
        self.backend: MaskComponent | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_dir = None
        self.output_dir = None
        self.frame_names = []
        self.prompts_added = False
        self._load_settings()
        # Colormap for overlay
        self.colors_custom = [
            "#E69F00", "#D55E00", "#56B4E9", "#009E73",
            "#0072B2", "#CC79A7", "#F0E442", "#B3D100", "#9E1B32"
        ]
        self.cmap_custom = ListedColormap(self.colors_custom, name="cmap_custom")

        self._build_ui()
    def _load_settings(self):
        """Load settings from JSON if it exists."""
        try:
            if self.SETTINGS_PATH.is_file():
                data = json.load(self.SETTINGS_PATH.open("r", encoding="utf-8"))
                # core settings
                self.model_checkpoint = data.get("model_checkpoint", self.model_checkpoint)
                self.model_cfg = data.get("model_cfg", self.model_cfg)
                self.point_color = data.get("point_color", self.point_color)
                self.box_color = data.get("box_color", self.box_color)
                # optional paths
                self.video_dir = data.get("video_dir", self.video_dir)
                self.output_dir = data.get("output_dir", self.output_dir)
        except Exception as e:
            messagebox.showwarning("Settings", f"Could not load settings file:\n{e}")

    def _save_settings(self):
        """Persist current settings to JSON."""
        data = {
            "model_checkpoint": self.model_checkpoint,
            "model_cfg": self.model_cfg,
            "point_color": self.point_color,
            "box_color": self.box_color,
            "video_dir": self.video_dir,
            "output_dir": self.output_dir,
        }
        try:
            with self.SETTINGS_PATH.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            messagebox.showwarning("Settings", f"Could not save settings:\n{e}")
        messagebox.showinfo("Settings", f"Settings saved at {self.SETTINGS_PATH}")
        

    def _build_ui(self):
        top_bar_frame = tk.Frame(self.root)
        top_bar_frame.pack(padx=10, pady=(5, 0), fill="x")
        tk.Button(top_bar_frame, text="Segmentation", command=self.open_settings).pack(side=tk.LEFT)
        tk.Button(top_bar_frame, text="Particle Tracking", command=self.open_settings).pack(side=tk.LEFT)
        tk.Button(top_bar_frame, text="Settings", command=self.open_settings).pack(side=tk.RIGHT)

        path_frame = tk.LabelFrame(self.root, text="Data and Output Paths")
        path_frame.pack(padx=10, pady=5, fill="x")

        tk.Label(path_frame, text="Video Directory:").grid(row=0, column=0, sticky="w")
        self.video_dir_entry = tk.Entry(path_frame, width=50)
        self.video_dir_entry.grid(row=0, column=1, padx=5)
        tk.Button(path_frame, text="Browse...", command=self.select_video_dir).grid(row=0, column=2)

        tk.Label(path_frame, text="Output Directory:").grid(row=1, column=0, sticky="w")
        self.output_dir_entry = tk.Entry(path_frame, width=50)
        self.output_dir_entry.grid(row=1, column=1, padx=5)
        tk.Button(path_frame, text="Browse...", command=self.select_output_dir).grid(row=1, column=2)

        self.init_button = tk.Button(path_frame, text="Load Video & Initialize", command=self.initialize_inference)
        self.init_button.grid(row=2, column=0, columnspan=3, pady=5)

        self.init_progress_label = tk.Label(path_frame, text="")
        self.init_progress_label.grid(row=3, column=0, columnspan=3, sticky="ew")
        self.init_progress = ttk.Progressbar(path_frame, orient="horizontal", mode='indeterminate')
        self.init_progress.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 5), padx=5)
        self.init_progress.grid_remove()

        seg_frame = tk.LabelFrame(self.root, text="Segmentation Controls")
        seg_frame.pack(padx=10, pady=5, fill="x")

        tk.Label(seg_frame, text="Frame to Annotate:").grid(row=0, column=0)
        self.frame_idx_entry = tk.Entry(seg_frame, width=10)
        self.frame_idx_entry.grid(row=0, column=1)

        tk.Button(seg_frame, text="Annotate Frame", command=self.annotate_frame).grid(row=1, column=0, pady=5)
        tk.Button(seg_frame, text="Reset All Predictions", command=self.reset_predictions).grid(row=1, column=1)

        prop_frame = tk.LabelFrame(self.root, text="Propagation")
        prop_frame.pack(padx=10, pady=5, fill="x")

        self.prop_button = tk.Button(prop_frame, text="Propagate Masks Through Video", command=self.propagate_masks)
        self.prop_button.pack(pady=5)

        self.prop_progress_label = tk.Label(prop_frame, text="")
        self.prop_progress_label.pack(fill="x")
        self.prop_progress = ttk.Progressbar(prop_frame, orient="horizontal", mode='determinate')
        self.prop_progress.pack(pady=(0, 5), padx=5, fill="x")
        self.prop_progress.pack_forget()

    # ---------- UI helpers ----------
    def _validate_model_paths(self) -> bool:
        """Return True if both model paths exist, else show a helpful error and return False."""
        missing = []
        if not os.path.isfile(self.model_cfg):
            missing.append(f"Config (.yaml) not found:\n{self.model_cfg}")
        if not os.path.isfile(self.model_checkpoint):
            missing.append(f"Checkpoint (.pt) not found:\n{self.model_checkpoint}")

        if missing:
            messagebox.showerror(
                "Model Files Missing",
                "Please fix the following before initializing:\n\n" + "\n\n".join(missing) +
                "\n\nTip: Open Settings to update paths."
            )
            return False
        return True

    def open_settings(self):
        SettingsWindow(self)

    def select_video_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.video_dir_entry.delete(0, tk.END)
            self.video_dir_entry.insert(0, dir_path)

    def select_output_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, dir_path)

    # ---------- Init / Annotate / Reset ----------

    def initialize_inference(self):
        self.video_dir = self.video_dir_entry.get()
        self.output_dir = self.output_dir_entry.get()
        if not self.video_dir or not os.path.isdir(self.video_dir):
            messagebox.showerror("Error", "Please select a valid video directory.")
            return
        if not self.output_dir or not os.path.isdir(self.output_dir):
            messagebox.showerror("Error", "Please select a valid output directory.")
            return
        if not self._validate_model_paths():
            messagebox.showerror("Error", "Please fix the model paths in Settings.")
            return

        # Proceed with async init
        self._save_settings()
        self.init_button.config(state=tk.DISABLED)
        self.init_progress.grid()
        self.init_progress_label.config(text="Initializing, please wait...")
        self.init_progress.start()

        threading.Thread(target=self._initialize_worker, daemon=True).start()

    def _initialize_worker(self):
        try:
            # Build backend using current settings
            self.backend = MaskComponent(self.model_cfg, self.model_checkpoint, device=self.device)
            self.frame_names = self.backend.init_video(self.video_dir)
            self.prompts_added = False
            success = True
            err_msg = ""
        except Exception as e:
            success = False
            err_msg = str(e)

        def finalize():
            self.init_progress.stop()
            self.init_progress.grid_remove()
            self.init_button.config(state=tk.NORMAL)
            self.init_progress_label.config(text="Initialization Complete!" if success else "Initialization Failed.")
            if success:
                messagebox.showinfo("Success", "Inference state initialized.")
            else:
                messagebox.showerror("Initialization Error", err_msg)

        self.root.after(0, finalize)

    def annotate_frame(self):
        if not self.backend or not self.backend.inference_state:
            messagebox.showerror("Error", "Please initialize inference first.")
            return

        try:
            frame_idx = int(self.frame_idx_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Frame Index must be an integer.")
            return

        if not (0 <= frame_idx < len(self.frame_names)):
            messagebox.showerror("Error", f"Frame Index must be between 0 and {len(self.frame_names) - 1}.")
            return

        image_path = os.path.join(self.video_dir, self.frame_names[frame_idx])
        image = Image.open(image_path)
        PromptWindow(self, frame_idx, image)

    def reset_predictions(self):
        if self.backend and self.backend.inference_state:
            try:
                self.backend.reset()
                self.prompts_added = False
                messagebox.showinfo("Success", "Predictions have been reset.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reset predictions:\n{e}")
        else:
            messagebox.showwarning("Warning", "Inference state not initialized.")

    # ---------- Propagation ----------

    def propagate_masks(self):
        if not self.backend or not self.backend.inference_state:
            messagebox.showerror("Error", "Please initialize inference first.")
            return
        if not self.prompts_added:
            messagebox.showerror("Error", "Please add at least one prompt before propagating.")
            return

        self.prop_button.config(state=tk.DISABLED)
        self.prop_progress.pack()
        self.prop_progress_label.config(text="Starting propagation...")
        self.prop_progress['value'] = 0
        self.prop_progress['maximum'] = len(self.frame_names)

        threading.Thread(target=self._propagate_worker, daemon=True).start()

    def _propagate_worker(self):
        def ui_progress(cur, total):
            self.root.after(0, self._update_prop_progress, cur, total)

        try:
            self.backend.propagate(self.output_dir, self.colors_custom, progress_callback=ui_progress)
            err = None
        except Exception as e:
            err = str(e)

        def finalize():
            if err:
                self.prop_progress_label.config(text="Propagation Failed.")
                messagebox.showerror("Propagation Error", err)
            else:
                self.prop_progress_label.config(text="Propagation Complete!")
                messagebox.showinfo("Success", "Mask propagation complete.")
            self.prop_button.config(state=tk.NORMAL)

        self.root.after(0, finalize)

    def _update_prop_progress(self, current, total):
        self.prop_progress['value'] = current
        self.prop_progress_label.config(text=f"Processed frame {current}/{total}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MaskApp(root)
    root.mainloop()
