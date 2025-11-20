import numpy as np
import time 
import os
import datetime
import shutil
import subprocess

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as patches 
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: Matplotlib not found.")

# --- Global Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
IMAGE_OUTPUT_DIR_NAME = "nn_classification_visualizations"
IMAGE_OUTPUT_DIR = os.path.join(script_dir, IMAGE_OUTPUT_DIR_NAME)
QMD_FILENAME = "nn_classification_report.qmd"


# --- Neural Network Class ---
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, random_seed=None):
        if random_seed is not None: np.random.seed(random_seed)
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) * 0.2 - 0.1
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) * 0.2 - 0.1
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        self.hidden_layer_input_sum, self.hidden_layer_activation = None, None
        self.output_layer_input_sum, self.output_layer_activation = None, None
        self.loss_history_ = []
        print(f"NN Initialized: Input({input_size})->Hidden({hidden_size})->Output({output_size})")

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _sigmoid_derivative(self, x_act): return x_act * (1 - x_act)

    def forward_propagate(self, X_input, store_internals=True): # Modified to always store by default for viz
        if store_internals:
            self.hidden_layer_input_sum = np.dot(X_input, self.weights_input_hidden) + self.bias_hidden
            self.hidden_layer_activation = self._sigmoid(self.hidden_layer_input_sum)
            self.output_layer_input_sum = np.dot(self.hidden_layer_activation, self.weights_hidden_output) + self.bias_output
            self.output_layer_activation = self._sigmoid(self.output_layer_input_sum)
        else: # For speed during bulk prediction if internals not needed
            h_sum = np.dot(X_input, self.weights_input_hidden) + self.bias_hidden
            h_act = self._sigmoid(h_sum)
            o_sum = np.dot(h_act, self.weights_hidden_output) + self.bias_output
            return self._sigmoid(o_sum)
        return self.output_layer_activation


    def back_propagate(self, X_input, y_true, predicted_output, learning_rate):
        # These activations must have been set by a preceding forward_propagate call
        if self.hidden_layer_activation is None or self.output_layer_activation is None:
            # This might happen if predict_class was called without store_internals=True
            # For training, forward_propagate in train() method always stores them.
            self.forward_propagate(X_input, store_internals=True) 
            predicted_output = self.output_layer_activation # Update predicted_output if re-calculated

        output_error = y_true - predicted_output
        delta_output = output_error * self._sigmoid_derivative(predicted_output)
        hidden_layer_error = np.dot(delta_output, self.weights_hidden_output.T)
        delta_hidden = hidden_layer_error * self._sigmoid_derivative(self.hidden_layer_activation)
        
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_layer_activation.T, delta_output)
        self.bias_output += learning_rate * np.sum(delta_output, axis=0, keepdims=True)
        self.weights_input_hidden += learning_rate * np.dot(X_input.T, delta_hidden)
        self.bias_hidden += learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

    def train(self, X_train, y_train, epochs, learning_rate, print_loss_every=100):
        print("\n--- Training Started ---"); self.loss_history_ = []
        for epoch in range(epochs):
            predicted_output = self.forward_propagate(X_train, store_internals=True) # Ensure internals stored
            self.back_propagate(X_train, y_train, predicted_output, learning_rate)
            loss = np.mean(0.5 * (y_train - predicted_output)**2)
            self.loss_history_.append(loss)
            if (epoch+1)%print_loss_every==0 or epoch==0: print(f"E {epoch+1}/{epochs}, Loss:{loss:.6f}")
        print("--- Training Finished ---")

    def predict_raw_outputs(self, X_input): 
        return self.forward_propagate(X_input, store_internals=False) # Don't need to store for simple prediction

    def predict_class(self, X_input):
        return np.argmax(self.predict_raw_outputs(X_input), axis=1)

# --- Visualization Functions ---
def ensure_dir(directory): # ... (same)
    if not os.path.exists(directory): os.makedirs(directory)

def plot_nn_architecture(input_size, hidden_size, output_size, filename="nn_architecture.png"): # ... (same)
    if not MATPLOTLIB_AVAILABLE: return os.path.basename(filename)
    fig, ax = plt.subplots(figsize=(8,5)); ax.axis('off'); layer_x=[1,3,5]; layer_lbls=['Input','Hidden','Output']
    node_counts=[input_size,hidden_size,output_size]; max_nodes=max(node_counts); node_r=0.25; y_sp=0.8 if max_nodes<=5 else 0.6
    for i,(x,cnt,lbl) in enumerate(zip(layer_x,node_counts,layer_lbls)):
        ax.text(x,max_nodes*y_sp+1.2,lbl,ha='center',fontsize=12); start_y=(max_nodes-cnt)/2.0*y_sp+node_r+0.5
        curr_nodes=[]
        for n in range(cnt):
            y=start_y+n*y_sp; circle=patches.Circle((x,y),node_r,fc='skyblue',ec='black'); ax.add_patch(circle)
            ax.text(x,y,f"N{n+1}",ha='center',va='center',fontsize=7 if cnt<=5 else 6); curr_nodes.append((x,y))
        if i>0:
            for prev_x,prev_y in prev_nodes:
                for curr_x,curr_y in curr_nodes: ax.plot([prev_x+node_r,curr_x-node_r],[prev_y,curr_y],'gray',alpha=0.5,lw=0.8)
        prev_nodes=curr_nodes
    ax.set_ylim(0,max_nodes*y_sp+2); ax.set_xlim(0,layer_x[-1]+1); plt.title("NN Architecture",fontsize=14)
    fp=os.path.join(IMAGE_OUTPUT_DIR,filename); plt.savefig(fp,bbox_inches='tight',dpi=100); plt.close(fig)
    return os.path.basename(filename)

# --- Enhanced Forward Pass Visualization ---
def plot_forward_pass_detailed(nn_model, sample_x, filename="forward_pass_detailed.png"):
    if not MATPLOTLIB_AVAILABLE: return os.path.basename(filename)

    # Ensure forward pass is done and internal states are stored
    _ = nn_model.forward_propagate(sample_x.reshape(1, -1), store_internals=True)

    fig = plt.figure(figsize=(10, 10)) # Increased size
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 2], height_ratios=[1,1,1], wspace=0.3, hspace=0.4)

    def format_array_for_table(arr):
        return [[f"{x:.2f}" for x in row] for row in arr.reshape(arr.shape[0], -1)]

    # 1. Input X
    ax_x = fig.add_subplot(gs[0, 0])
    ax_x.axis('off')
    table_x = ax_x.table(cellText=format_array_for_table(sample_x.reshape(1, -1)),
                         rowLabels=['X (Input)'], colLabels=[f'x{i+1}' for i in range(nn_model.input_size)],
                         loc='center', cellLoc='center')
    table_x.auto_set_font_size(False); table_x.set_fontsize(9); table_x.scale(1, 1.5)
    ax_x.set_title('1. Input Sample (X)', fontsize=11)

    # 2. Z_h = X @ W_ih + b_h  and A_h = sigmoid(Z_h)
    ax_h = fig.add_subplot(gs[0, 1])
    ax_h.axis('off')
    zh_data = format_array_for_table(nn_model.hidden_layer_input_sum)
    ah_data = format_array_for_table(nn_model.hidden_layer_activation)
    table_h_content = [zh_data[0], ah_data[0]]
    table_h = ax_h.table(cellText=table_h_content,
                         rowLabels=['Z_h (Net Input)', 'A_h (Activation)'],
                         colLabels=[f'Hidden {i+1}' for i in range(nn_model.hidden_size)],
                         loc='center', cellLoc='center')
    table_h.auto_set_font_size(False); table_h.set_fontsize(9); table_h.scale(1, 1.8)
    ax_h.set_title(f'2. Hidden Layer (Z_h = XW_ih+b_h; A_h=σ(Z_h))', fontsize=11)

    # 3. W_ih and b_h (Conceptual - too large to display all values nicely)
    ax_w_ih = fig.add_subplot(gs[1,0])
    ax_w_ih.imshow(nn_model.weights_input_hidden, cmap='coolwarm', aspect='auto', vmin=-0.5, vmax=0.5)
    ax_w_ih.set_title(f'Weights W_ih ({nn_model.input_size}x{nn_model.hidden_size})', fontsize=9)
    ax_w_ih.set_xticks([]); ax_w_ih.set_yticks([])
    # Add bias_hidden as text below if space allows or as part of title
    bias_h_str = ", ".join([f"{b:.2f}" for b in nn_model.bias_hidden.flatten()])
    ax_w_ih.set_xlabel(f"Bias b_h: [{bias_h_str}]", fontsize=8)

    # 4. Z_o = A_h @ W_ho + b_o and A_o = sigmoid(Z_o)
    ax_o = fig.add_subplot(gs[1, 1])
    ax_o.axis('off')
    zo_data = format_array_for_table(nn_model.output_layer_input_sum)
    ao_data = format_array_for_table(nn_model.output_layer_activation)
    table_o_content = [zo_data[0], ao_data[0]]
    table_o = ax_o.table(cellText=table_o_content,
                         rowLabels=['Z_o (Net Input)', 'A_o (Activation)'],
                         colLabels=[f'Output {i+1}' for i in range(nn_model.output_size)],
                         loc='center', cellLoc='center')
    table_o.auto_set_font_size(False); table_o.set_fontsize(9); table_o.scale(1, 1.8)
    ax_o.set_title(f'3. Output Layer (Z_o = A_hW_ho+b_o; A_o=σ(Z_o))', fontsize=11)

    # 5. W_ho and b_o
    ax_w_ho = fig.add_subplot(gs[2,0])
    ax_w_ho.imshow(nn_model.weights_hidden_output, cmap='coolwarm', aspect='auto', vmin=-0.5, vmax=0.5)
    ax_w_ho.set_title(f'Weights W_ho ({nn_model.hidden_size}x{nn_model.output_size})', fontsize=9)
    ax_w_ho.set_xticks([]); ax_w_ho.set_yticks([])
    bias_o_str = ", ".join([f"{b:.2f}" for b in nn_model.bias_output.flatten()])
    ax_w_ho.set_xlabel(f"Bias b_o: [{bias_o_str}]", fontsize=8)

    # 6. Final Prediction
    ax_pred = fig.add_subplot(gs[2,1])
    ax_pred.axis('off')
    final_pred_idx = np.argmax(nn_model.output_layer_activation)
    pred_text = f"Final Predicted Class Index: {final_pred_idx}"
    ax_pred.text(0.5, 0.5, pred_text, ha='center', va='center', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange"))
    ax_pred.set_title('4. Prediction', fontsize=11)
    
    # Arrows (conceptual)
    arrow_props = dict(arrowstyle="->", connectionstyle="arc3,rad=.1", color='gray', lw=1.0)
    # X to Hidden calculations
    con1 = patches.ConnectionPatch(xyA=(1.0, 0.5), coordsA=ax_x.transAxes, xyB=(0, 0.5), coordsB=ax_h.transAxes, **arrow_props)
    fig.add_artist(con1)
    # Hidden to Output calculations
    con2 = patches.ConnectionPatch(xyA=(1.0, 0.5), coordsA=ax_h.transAxes, xyB=(0, 0.5), coordsB=ax_o.transAxes, **arrow_props) # from ax_h (Hidden Activations) to ax_o (Output Calcs)
    fig.add_artist(con2)
     # Output Calcs to Prediction
    con3 = patches.ConnectionPatch(xyA=(1.0, 0.5), coordsA=ax_o.transAxes, xyB=(0, 0.5), coordsB=ax_pred.transAxes, **arrow_props)
    fig.add_artist(con3)


    plt.suptitle("Detailed Forward Pass for One Sample", fontsize=16, y=0.97)
    # plt.tight_layout(rect=[0,0,1,0.95]) # May need adjustment

    filepath = os.path.join(IMAGE_OUTPUT_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=120)
    plt.close(fig)
    return os.path.basename(filename)


# --- Conceptual Backpropagation Visualization ---
def plot_backpropagation_concept(nn_model, filename="backpropagation_concept.png"):
    if not MATPLOTLIB_AVAILABLE: return os.path.basename(filename)

    fig, ax = plt.subplots(figsize=(12, 8)) # Slightly wider for more text
    ax.axis('off')
    plt.title("Conceptual Backpropagation Flow", fontsize=16)

    # Properties for the text boxes (nodes)
    node_bbox_props = dict(boxstyle="round,pad=0.5", fc="lightblue", ec="blue", lw=1)
    
    # Properties for annotation arrows
    arrow_props = dict(arrowstyle="Simple,tail_width=0.5,head_width=4,head_length=8", 
                       facecolor="salmon", edgecolor="darkred", lw=0.5,
                       connectionstyle="arc3,rad=0.2") # Added connectionstyle

    # Properties for the text within annotations (if any)
    annotation_text_bbox_props = dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="gray")
    
    # Properties for gradient text
    grad_text_style = dict(ha="center", va="center", fontsize=8, color="purple",
                           bbox=dict(boxstyle="round,pad=0.1", fc="lavenderblush", alpha=0.7, ec="mediumpurple"))


    # Layer positions
    y_out, y_hid, y_in = 1.5, 4, 6.5 # Adjusted y positions for more space
    x_mid = 5
    x_text_offset = 3.0 # Offset for annotation text boxes

    # Nodes
    ax.text(x_mid, y_out, "Output Layer\n(A_o, Z_o)", ha="center", va="center", bbox=node_bbox_props, fontsize=10)
    ax.text(x_mid, y_hid, "Hidden Layer\n(A_h, Z_h)", ha="center", va="center", bbox=node_bbox_props, fontsize=10)
    ax.text(x_mid, y_in, "Input Layer (X)", ha="center", va="center", bbox=node_bbox_props, fontsize=10)

    # Backward Arrows and Text
    # Output Error & Delta
    ax.annotate("1. Output Error\nE_o = y_true - A_o", 
                xy=(x_mid, y_out + 0.6), xytext=(x_mid + x_text_offset, y_out + 1.0),
                arrowprops=arrow_props, ha="center", va="center", fontsize=9, bbox=annotation_text_bbox_props)
    ax.annotate("2. Output Delta\nδ_o = E_o * σ'(Z_o)", 
                xy=(x_mid, y_hid - 0.6), xytext=(x_mid - x_text_offset, y_out + 1.0), # Pointing from delta calc towards hidden
                arrowprops=arrow_props, ha="center", va="center", fontsize=9, bbox=annotation_text_bbox_props)


    # Hidden Error & Delta
    ax.annotate("3. Hidden Error (Propagated)\nE_h = δ_o @ W_ho^T", 
                xy=(x_mid, y_hid + 0.6), xytext=(x_mid + x_text_offset, y_hid + 1.0),
                arrowprops=arrow_props, ha="center", va="center", fontsize=9, bbox=annotation_text_bbox_props)
    ax.annotate("4. Hidden Delta\nδ_h = E_h * σ'(Z_h)", 
                xy=(x_mid, y_in - 0.6), xytext=(x_mid - x_text_offset, y_hid + 1.0), # Pointing from delta calc towards input
                arrowprops=arrow_props, ha="center", va="center", fontsize=9, bbox=annotation_text_bbox_props)

    # Gradient Updates (conceptually linked to deltas)
    # Text boxes for gradients, slightly offset from the main nodes
    ax.text(x_mid + 1.5, y_out - 0.8, "Update W_ho, b_o\nusing A_h & δ_o", **grad_text_style)
    ax.text(x_mid + 1.5, y_hid - 0.8, "Update W_ih, b_ih\nusing X & δ_h", **grad_text_style)
    
    # Connecting Deltas to Gradient Calculations (Conceptual lines)
    # From δ_o calculation area to W_ho update text
    ax.plot([x_mid - x_text_offset*0.8, x_mid + 1.5], [y_out + 0.8, y_out - 0.6], 
            linestyle=":", color="gray", lw=1.0)
    # From δ_h calculation area to W_ih update text
    ax.plot([x_mid - x_text_offset*0.8, x_mid + 1.5], [y_hid + 0.8, y_hid - 0.6], 
            linestyle=":", color="gray", lw=1.0)


    ax.set_xlim(0, x_mid + x_text_offset + 1.5)
    ax.set_ylim(0, y_in + 1.5)
    
    filepath = os.path.join(IMAGE_OUTPUT_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=100)
    plt.close(fig)
    return os.path.basename(filename)

def plot_loss_curve(loss_history, filename="loss_curve.png"): # ... (same)
    if not MATPLOTLIB_AVAILABLE: return os.path.basename(filename)
    fig, ax=plt.subplots(figsize=(8,5)); ax.plot(loss_history,label="Training Loss (MSE)",color="purple")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss"); ax.set_title("Training Loss"); ax.legend(); ax.grid(True,ls=':')
    plt.ylim(bottom=0,top=max(0.1,min(1.0,max(loss_history)*1.1)) if loss_history else 0.1)
    fp=os.path.join(IMAGE_OUTPUT_DIR,filename); plt.savefig(fp,bbox_inches='tight',dpi=100); plt.close(fig)
    return os.path.basename(filename)

def plot_decision_boundaries(nn_model,X_data,y_true_onehot,class_names,filename="decision_boundaries.png"): # ... (same)
    if not MATPLOTLIB_AVAILABLE: return os.path.basename(filename)
    y_true_idx=np.argmax(y_true_onehot,axis=1); h=.02
    x_min,x_max=X_data[:,0].min()-0.2,X_data[:,0].max()+0.2; y_min,y_max=X_data[:,1].min()-0.2,X_data[:,1].max()+0.2
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    mesh_pts=np.c_[xx.ravel(),yy.ravel()]; Z=nn_model.predict_class(mesh_pts); Z=Z.reshape(xx.shape)
    fig,ax=plt.subplots(figsize=(8,6)); cmap_light=plt.cm.get_cmap(name='Pastel1',lut=len(class_names))
    ax.contourf(xx,yy,Z,cmap=cmap_light,alpha=0.8); cmap_dark=plt.cm.get_cmap(name='Set1',lut=len(class_names))
    scatter=ax.scatter(X_data[:,0],X_data[:,1],c=y_true_idx,cmap=cmap_dark,edgecolor='k',s=50)
    handles,_=scatter.legend_elements(); legend_lbls=[class_names[i] for i in sorted(np.unique(y_true_idx))]
    ax.legend(handles,legend_lbls,title="True Classes"); ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    ax.set_title("Decision Boundaries"); ax.set_xlim(xx.min(),xx.max()); ax.set_ylim(yy.min(),yy.max())
    fp=os.path.join(IMAGE_OUTPUT_DIR,filename); plt.savefig(fp,bbox_inches='tight',dpi=100); plt.close(fig)
    return os.path.basename(filename)

# --- QMD Report Generation (Updated to include new figures) ---
def generate_qmd_report(
    nn_params, class_names, X_sample_for_fwd_pass, y_sample_for_fwd_pass_onehot,
    arch_fn, fwd_pass_detailed_fn, backprop_concept_fn, # New filenames
    loss_curve_fn, decision_bnd_fn,
    final_loss, accuracy,
    example_predictions
):
    qmd_content = f"""---
title: "Simple Neural Network Classification Report"
author: "Automated via Python Script"
date: "{datetime.date.today().isoformat()}"
format:
  html:
    toc: true
    code-fold: true
    self-contained: true 
jupyter: python3
---

## 1. Introduction 
{'(Same as before)'}

## 2. Dataset
{'(Same as before, ensure X_sample_for_fwd_pass and y_sample_for_fwd_pass_onehot are used)'}

## 3. Neural Network Architecture
{'(Same as before, refers to arch_fn)'}

## 4. Training Process

### 4.1. Forward Propagation
During the forward pass, input data is processed layer by layer to produce an output.
1.  **Input to Hidden Layer:** `Z_h = X @ W_ih + b_h`, followed by `A_h = sigmoid(Z_h)`.
2.  **Hidden to Output Layer:** `Z_o = A_h @ W_ho + b_o`, followed by `A_o = sigmoid(Z_o)`.

The diagram below provides a more detailed breakdown of the calculations involved in a forward pass for a single sample, showing the matrices and vectors at each stage:

![Detailed Illustration of a Forward Pass for a Single Sample]({IMAGE_OUTPUT_DIR_NAME}/{fwd_pass_detailed_fn})
*Figure 2: Detailed Forward Propagation. This figure illustrates: 1. The input sample vector (X). 2. The calculations for the hidden layer, including the net input sum (Z_h) and the activations (A_h). The weights W_ih and bias b_h used in this step are shown conceptually. 3. The calculations for the output layer, including its net input sum (Z_o) and final activations (A_o). The weights W_ho and bias b_o are also shown conceptually. 4. The final class prediction derived from A_o.*

### 4.2. Loss Function
{'(Same as before)'}

### 4.3. Backpropagation
The backpropagation algorithm is key to training the network. It involves calculating the error at the output and then propagating this error backward to adjust the weights and biases throughout the network. The core idea is to determine how much each weight and bias contributed to the overall error and then update them in a direction that reduces this error. This is achieved using the chain rule of calculus to compute gradients.

The conceptual flow of backpropagation is illustrated below:

![Conceptual Flow of the Backpropagation Algorithm]({IMAGE_OUTPUT_DIR_NAME}/{backprop_concept_fn})
*Figure 3: Backpropagation Concept. This diagram shows: 1. Calculation of the output error (E_o). 2. Calculation of the output delta (δ_o), which scales the error by the derivative of the output activation. 3. Propagation of the error to the hidden layer (E_h) using output deltas and weights W_ho. 4. Calculation of the hidden delta (δ_h). These deltas are then used to compute the gradients (∇W, ∇b) for updating the respective weights and biases.*

The learning progress, driven by repeated forward and backward passes, is shown by the training loss curve:

![Training Loss Curve Over Epochs]({IMAGE_OUTPUT_DIR_NAME}/{loss_curve_fn})
*Figure 4: Training Loss Curve. (Same explanation as before for loss_curve_fn)*

## 5. Results
{'(Same as before, nn_params should include epochs and learning_rate for this text)'}

### 5.1. Classification Visualization and Decision Boundaries
{'(Same as before, refers to decision_bnd_fn)'}

### 5.2. Example Predictions on Training Data
{'(Same as before)'}

## 6. Conclusion
{'(Same as before)'}
"""
    qmd_file_path = os.path.join(script_dir, QMD_FILENAME)
    try:
        with open(qmd_file_path, 'w', encoding='utf-8') as f: f.write(qmd_content)
        print(f"\nQuarto Markdown report generated: {qmd_file_path}"); return qmd_file_path
    except Exception as e: print(f"Error writing QMD file: {e}"); return None

# --- Main Script ---
if __name__ == "__main__":
    if not MATPLOTLIB_AVAILABLE: print("Matplotlib essential. Install it."); exit()
    ensure_dir(IMAGE_OUTPUT_DIR)
    print("--- Simple NN for Object Classification ---")

    X_data = np.array([[0.1,0.2],[0.2,0.1],[0.8,0.3],[0.7,0.25],[0.3,0.8],[0.25,0.7],[0.15,0.9],[0.9,0.85],[0.05,0.1],[0.95,0.15]])
    y_data_onehot = np.array([[1,0,0],[1,0,0],[0,1,0],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[0,0,1]])
    class_names = ["Small & Dim", "Bright OR Large", "Small & Bright"]
    y_data_indices = np.argmax(y_data_onehot, axis=1)

    nn_params = {
        'input_size': X_data.shape[1], 'hidden_size': 5, 'output_size': y_data_onehot.shape[1],
        'learning_rate': 0.1, 'epochs': 20000 # Added for QMD text
    }
    
    nn_classifier = SimpleNeuralNetwork(nn_params['input_size'], nn_params['hidden_size'], nn_params['output_size'], random_seed=45)
    arch_fn = plot_nn_architecture(nn_params['input_size'], nn_params['hidden_size'], nn_params['output_size'])
    
    # For detailed forward pass and backprop concept, we need a trained model (or at least initialized)
    # We'll generate backprop concept once, forward pass for one sample after training
    backprop_concept_filename = plot_backpropagation_concept(nn_classifier) # Uses initial weights if called before train

    nn_classifier.train(X_data, y_data_onehot, nn_params['epochs'], nn_params['learning_rate'], print_loss_every=nn_params['epochs'] // 10)

    sample_idx_for_fwd_pass = 2 
    fwd_pass_detailed_filename = plot_forward_pass_detailed(
        nn_classifier, X_data[sample_idx_for_fwd_pass]
    )
    loss_curve_filename = plot_loss_curve(nn_classifier.loss_history_)
    decision_boundaries_filename = plot_decision_boundaries(nn_classifier, X_data, y_data_onehot, class_names)

    final_loss = nn_classifier.loss_history_[-1] if nn_classifier.loss_history_ else float('inf')
    predicted_classes_indices = nn_classifier.predict_class(X_data)
    raw_preds_final = nn_classifier.predict_raw_outputs(X_data)
    correct_count = np.sum(predicted_classes_indices == y_data_indices)
    accuracy = (correct_count / len(X_data)) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}%")

    example_preds_for_qmd = []
    for i in range(min(5, len(X_data))):
        example_preds_for_qmd.append({
            'input': X_data[i], 'true_idx': y_data_indices[i], 'true_name': class_names[y_data_indices[i]],
            'pred_idx': predicted_classes_indices[i], 'pred_name': class_names[predicted_classes_indices[i]],
            'raw_output': raw_preds_final[i]
        })
    
    qmd_file_path = generate_qmd_report(
        nn_params, class_names, X_data[sample_idx_for_fwd_pass], y_data_onehot[sample_idx_for_fwd_pass],
        arch_fn, fwd_pass_detailed_filename, backprop_concept_filename, 
        loss_curve_filename, decision_boundaries_filename,
        final_loss, accuracy, example_preds_for_qmd
    )

    if qmd_file_path and os.path.exists(qmd_file_path):
        quarto_executable = shutil.which("quarto")
        if quarto_executable:
            print(f"\nRendering Quarto: {qmd_file_path}")
            try:
                cmd = [quarto_executable, "render", qmd_file_path]
                res = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=script_dir, timeout=120)
                if res.returncode == 0: print(f"Quarto rendered to {os.path.splitext(qmd_file_path)[0]}.html")
                else: print("Quarto Error:\nStdout:", res.stdout, "\nStderr:", res.stderr)
            except Exception as e: print(f"Quarto rendering error: {e}")
        else: print("Quarto not found. Render manually.")
    print(f"\nCheck '{IMAGE_OUTPUT_DIR}' for PNGs and '{script_dir}' for QMD/HTML.")