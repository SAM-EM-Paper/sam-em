import customtkinter as ctk

class ParticleTrackingUI:
    def __init__(self, parent, app):
        """
        parent : the CTkTab (frame) to build on
        app    : reference to the main MaskApp (so you can call its methods/attributes)
        """
        self.parent = parent
        self.app = app
        self._build_ui()

    def _build_ui(self):
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(frame, text="Particle Tracking Module").pack(pady=20)

        # Example controls
        ctk.CTkButton(
            frame, text="Run Tracking", 
            command=self.run_tracking
        ).pack(pady=10)

    def run_tracking(self):
        # Here you can access self.app.backend, self.app.video_dir, etc.
        print("Running tracking with video:", self.app.video_dir)