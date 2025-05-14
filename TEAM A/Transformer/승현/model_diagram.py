import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_transformer_diagram():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Colors
    colors = {
        'input': '#FFB6C1',  # Light pink
        'output': '#98FB98',  # Light green
        'attention': '#87CEEB',  # Sky blue
        'ffn': '#DDA0DD',  # Plum
        'norm': '#F0E68C',  # Khaki
        'embed': '#E6E6FA',  # Lavender
    }
    
    # Box dimensions
    box_width = 2.0
    box_height = 0.8
    spacing = 0.5
    
    # Draw input embeddings
    input_embed = patches.Rectangle((1, 8), box_width, box_height, 
                                  facecolor=colors['embed'], edgecolor='black')
    ax.add_patch(input_embed)
    ax.text(2, 8.4, 'Input Embedding', ha='center', va='center')
    
    # Draw positional encoding
    pos_encoding = patches.Rectangle((1, 7), box_width, box_height,
                                   facecolor=colors['embed'], edgecolor='black')
    ax.add_patch(pos_encoding)
    ax.text(2, 7.4, 'Positional Encoding', ha='center', va='center')
    
    # Draw encoder layers
    for i in range(6):
        y_pos = 6 - i * 1.5
        
        # Self-attention
        self_attn = patches.Rectangle((1, y_pos), box_width, box_height,
                                    facecolor=colors['attention'], edgecolor='black')
        ax.add_patch(self_attn)
        ax.text(2, y_pos + 0.4, 'Self-Attention', ha='center', va='center')
        
        # Add norm
        norm1 = patches.Rectangle((1, y_pos - 0.7), box_width, box_height,
                                facecolor=colors['norm'], edgecolor='black')
        ax.add_patch(norm1)
        ax.text(2, y_pos - 0.3, 'Layer Norm', ha='center', va='center')
        
        # Feed-forward
        ffn = patches.Rectangle((1, y_pos - 1.4), box_width, box_height,
                              facecolor=colors['ffn'], edgecolor='black')
        ax.add_patch(ffn)
        ax.text(2, y_pos - 1.0, 'Feed Forward', ha='center', va='center')
        
        # Add norm
        norm2 = patches.Rectangle((1, y_pos - 2.1), box_width, box_height,
                                facecolor=colors['norm'], edgecolor='black')
        ax.add_patch(norm2)
        ax.text(2, y_pos - 1.7, 'Layer Norm', ha='center', va='center')
    
    # Draw decoder layers
    for i in range(6):
        y_pos = -1 - i * 1.5
        
        # Self-attention
        self_attn = patches.Rectangle((4, y_pos), box_width, box_height,
                                    facecolor=colors['attention'], edgecolor='black')
        ax.add_patch(self_attn)
        ax.text(5, y_pos + 0.4, 'Self-Attention', ha='center', va='center')
        
        # Add norm
        norm1 = patches.Rectangle((4, y_pos - 0.7), box_width, box_height,
                                facecolor=colors['norm'], edgecolor='black')
        ax.add_patch(norm1)
        ax.text(5, y_pos - 0.3, 'Layer Norm', ha='center', va='center')
        
        # Cross-attention
        cross_attn = patches.Rectangle((4, y_pos - 1.4), box_width, box_height,
                                     facecolor=colors['attention'], edgecolor='black')
        ax.add_patch(cross_attn)
        ax.text(5, y_pos - 1.0, 'Cross-Attention', ha='center', va='center')
        
        # Add norm
        norm2 = patches.Rectangle((4, y_pos - 2.1), box_width, box_height,
                                facecolor=colors['norm'], edgecolor='black')
        ax.add_patch(norm2)
        ax.text(5, y_pos - 1.7, 'Layer Norm', ha='center', va='center')
        
        # Feed-forward
        ffn = patches.Rectangle((4, y_pos - 2.8), box_width, box_height,
                              facecolor=colors['ffn'], edgecolor='black')
        ax.add_patch(ffn)
        ax.text(5, y_pos - 2.4, 'Feed Forward', ha='center', va='center')
        
        # Add norm
        norm3 = patches.Rectangle((4, y_pos - 3.5), box_width, box_height,
                                facecolor=colors['norm'], edgecolor='black')
        ax.add_patch(norm3)
        ax.text(5, y_pos - 3.1, 'Layer Norm', ha='center', va='center')
    
    # Draw output layer
    output_layer = patches.Rectangle((4, -10), box_width, box_height,
                                   facecolor=colors['output'], edgecolor='black')
    ax.add_patch(output_layer)
    ax.text(5, -9.6, 'Output Layer', ha='center', va='center')
    
    # Add arrows for data flow
    # Input to encoder
    ax.arrow(2, 7.5, 0, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Encoder to decoder
    ax.arrow(2, -1, 2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Decoder to output
    ax.arrow(5, -10.5, 0, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Add residual connections (simplified)
    for i in range(6):
        y_pos = 6 - i * 1.5
        ax.plot([1, 3], [y_pos, y_pos], 'k--', alpha=0.3)
        ax.plot([1, 3], [y_pos - 2.1, y_pos - 2.1], 'k--', alpha=0.3)
    
    # Set axis properties
    ax.set_xlim(0, 7)
    ax.set_ylim(-12, 9)
    ax.axis('off')
    
    # Add title
    plt.title('Transformer Architecture', pad=20, size=14)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=colors['embed'], edgecolor='black', label='Embedding'),
        patches.Patch(facecolor=colors['attention'], edgecolor='black', label='Attention'),
        patches.Patch(facecolor=colors['ffn'], edgecolor='black', label='Feed Forward'),
        patches.Patch(facecolor=colors['norm'], edgecolor='black', label='Layer Norm'),
        patches.Patch(facecolor=colors['output'], edgecolor='black', label='Output')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
             ncol=5, frameon=False)
    
    # Save the diagram
    plt.savefig('transformer_architecture.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    create_transformer_diagram()
    print("Transformer architecture diagram has been saved as 'transformer_architecture.png'") 