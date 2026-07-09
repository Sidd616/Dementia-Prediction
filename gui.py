import tkinter.messagebox as messagebox

import customtkinter as ctk

from model import INPUT_COLUMNS, load_or_train_model


class DementiaPredictionGUI:
    def __init__(self, root, model=None):
        self.root = root
        self.root.title("Dementia Prediction System")
        self.root.geometry("1000x900")  # Increased width further for larger elements
        
        # Set the color theme and appearance
        ctk.set_appearance_mode("light")  # Light mode for brighter colors
        ctk.set_default_color_theme("green")  # Changed to green theme for variety
        
        # Define custom colors
        self.colors = {
            'primary': '#2ECC71',     # Bright green
            'secondary': '#3498DB',   # Bright blue
            'accent': '#E74C3C',      # Coral red
            'background': '#ECF0F1',  # Light gray
            'text': '#2C3E50'         # Dark blue-gray
        }
        
        # Initialize the model (loads a cached model from disk if available,
        # otherwise trains a fresh one and caches it for next launch)
        if model is not None:
            self.model = model
        else:
            try:
                self.model = load_or_train_model()
                print("Model ready")
            except Exception as e:
                messagebox.showerror("Error", f"Error preparing model: {str(e)}")
                raise
        
        self.create_widgets()
    
    def create_widgets(self):
        # Create main frame with scrollbar
        main_frame = ctk.CTkFrame(self.root, fg_color=self.colors['background'])
        main_frame.pack(fill=ctk.BOTH, expand=True, padx=30, pady=30)
        
        # Add canvas and scrollbar
        canvas = ctk.CTkCanvas(main_frame, bg=self.colors['background'])
        scrollbar = ctk.CTkScrollbar(main_frame, orientation="vertical", command=canvas.yview)
        self.scrollable_frame = ctk.CTkFrame(canvas, fg_color=self.colors['background'])
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Decorative Header
        header_frame = ctk.CTkFrame(self.scrollable_frame, fg_color=self.colors['primary'], corner_radius=15)
        header_frame.grid(row=0, column=0, columnspan=2, pady=(0, 40), sticky="ew", padx=20)
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Dementia Prediction System",
            font=ctk.CTkFont(family="Helvetica", size=32, weight="bold"),
            text_color="white"
        )
        title_label.pack(pady=20)
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Advanced Neural Analysis Platform",
            font=ctk.CTkFont(family="Helvetica", size=18),
            text_color="white"
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Sections with enhanced styling
        sections = {
            "👤 Patient Demographics": [
                ('Age', None),
                ('Gender', ['Male', 'Female']),
                ('Weight', None),
                ('Dominant_Hand', ['Right', 'Left'])
            ],
            "🏥 Medical History": [
                ('Diabetic', ['0', '1']),
                ('Family_History', ['Yes', 'No']),
                ('Chronic_Health_Conditions', ['None', 'Heart Disease', 'Diabetes', 'Hypertension']),
                ('APOE_ε4', ['Positive', 'Negative'])
            ],
            "❤️ Vital Signs": [
                ('HeartRate', None),
                ('BloodOxygenLevel', None),
                ('BodyTemperature', None)
            ],
            "🌿 Lifestyle Factors": [
                ('AlcoholLevel', None),
                ('Smoking_Status', ['Never Smoked', 'Former Smoker', 'Current Smoker']),
                ('Physical_Activity', ['Sedentary', 'Mild Activity', 'Moderate Activity', 'High Activity']),
                ('Sleep_Quality', ['Good', 'Fair', 'Poor']),
                ('Nutrition_Diet', ['Balanced Diet', 'Low-Carb Diet', 'Mediterranean Diet', 'Other'])
            ],
            "🧠 Clinical Assessment": [
                ('Cognitive_Test_Scores', ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']),
                ('Depression_Status', ['Yes', 'No']),
                ('MRI_Delay', None)
            ]
        }
        
        # Input fields
        self.inputs = {}
        current_row = 1
        
        for section_name, fields in sections.items():
            # Create section frame with gradient-like effect
            section_frame = ctk.CTkFrame(
                self.scrollable_frame,
                fg_color=self.colors['secondary'],
                corner_radius=15
            )
            section_frame.grid(row=current_row, column=0, columnspan=2, padx=20, pady=(0, 30), sticky="ew")
            
            # Section header with icon
            ctk.CTkLabel(
                section_frame,
                text=section_name,
                font=ctk.CTkFont(size=24, weight="bold"),
                text_color="white"
            ).pack(pady=(15, 10), padx=15, anchor="w")
            
            # Create grid frame for inputs
            grid_frame = ctk.CTkFrame(section_frame, fg_color="white", corner_radius=10)
            grid_frame.pack(fill="x", padx=15, pady=(0, 15))
            
            # Add fields with enhanced styling
            for i, (field, values) in enumerate(fields):
                ctk.CTkLabel(
                    grid_frame,
                    text=field,
                    font=ctk.CTkFont(size=16, weight="bold"),
                    text_color=self.colors['text']
                ).grid(row=i, column=0, padx=(15, 10), pady=10, sticky="e")
                
                if values:
                    self.inputs[field] = ctk.CTkComboBox(
                        grid_frame,
                        values=values,
                        width=250,
                        height=40,
                        font=ctk.CTkFont(size=16),
                        border_width=2,
                        button_color=self.colors['primary'],
                        button_hover_color=self.colors['accent']
                    )
                else:
                    self.inputs[field] = ctk.CTkEntry(
                        grid_frame,
                        width=250,
                        height=40,
                        font=ctk.CTkFont(size=16),
                        border_width=2
                    )
                self.inputs[field].grid(row=i, column=1, padx=15, pady=10, sticky="w")
            
            current_row += 1
        
        # Predict button with enhanced styling
        predict_button = ctk.CTkButton(
            self.scrollable_frame,
            text="Generate Prediction",
            font=ctk.CTkFont(size=20, weight="bold"),
            height=50,
            corner_radius=25,
            command=self.make_prediction,
            fg_color=self.colors['accent'],
            hover_color=self.colors['primary']
        )
        predict_button.grid(row=current_row, column=0, columnspan=2, pady=30)
        current_row += 1
        
        # Results section with enhanced styling
        self.results_frame = ctk.CTkFrame(
            self.scrollable_frame,
            fg_color=self.colors['primary'],
            corner_radius=15
        )
        self.results_frame.grid(row=current_row, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        
        results_title = ctk.CTkLabel(
            self.results_frame,
            text="📊 Prediction Results",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="white"
        )
        results_title.pack(pady=20)
        
        # Labels for results with enhanced styling
        self.result_labels = {}
        models_frame = ctk.CTkFrame(self.results_frame, fg_color="white", corner_radius=10)
        models_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        for i, model_name in enumerate(['Logistic Regression', 'Random Forest', 'XGBoost']):
            model_frame = ctk.CTkFrame(models_frame, fg_color=self.colors['background'], corner_radius=10)
            model_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkLabel(
                model_frame,
                text=model_name,
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=self.colors['text']
            ).pack(pady=10)
            
            results_subframe = ctk.CTkFrame(model_frame, fg_color="transparent")
            results_subframe.pack(fill="x", padx=15, pady=(0, 10))
            
            self.result_labels[f"{model_name}_no_dementia"] = ctk.CTkLabel(
                results_subframe,
                text="No Dementia: --",
                font=ctk.CTkFont(size=16),
                text_color=self.colors['secondary']
            )
            self.result_labels[f"{model_name}_no_dementia"].pack(side="left", padx=10)
            
            self.result_labels[f"{model_name}_dementia"] = ctk.CTkLabel(
                results_subframe,
                text="Dementia: --",
                font=ctk.CTkFont(size=16),
                text_color=self.colors['accent']
            )
            self.result_labels[f"{model_name}_dementia"].pack(side="right", padx=10)
    
    def make_prediction(self):
        try:
            # Add animation effect
            self.root.config(cursor="wait")
            
            # Gather all inputs, in the same order the model expects
            input_values = []
            for field in INPUT_COLUMNS:
                value = self.inputs[field].get()
                if value == '':
                    messagebox.showerror("Error", f"Please fill in the {field} field")
                    self.root.config(cursor="")
                    return
                input_values.append(str(value))
            
            # Make prediction
            input_string = ','.join(input_values)
            predictions = self.model.predict(input_string)
            
            # Update results with animation effect
            if predictions:
                for model_name, probs in predictions.items():
                    self.result_labels[f"{model_name}_no_dementia"].configure(
                        text=f"No Dementia: {probs[0]:.1%}",
                        font=ctk.CTkFont(size=16, weight="bold")
                    )
                    self.result_labels[f"{model_name}_dementia"].configure(
                        text=f"Dementia: {probs[1]:.1%}",
                        font=ctk.CTkFont(size=16, weight="bold")
                    )
            else:
                messagebox.showerror("Error", "Prediction failed. Please check your inputs.")
            
            self.root.config(cursor="")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.root.config(cursor="")

def main():
    root = ctk.CTk()
    DementiaPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()