import torch
import torch.nn as nn

class MinRNNPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, rnn_type='minGRU', batch_first=True):
        """
        input_size: Size of each input vector x_t
        hidden_size: Size of the hidden state h_t
        output_size: Size of the final output at each time step
        n_layers: Number of stacked RNN layers
        rnn_type: 'minGRU' or 'minLSTM'
        batch_first: If True, the input and output tensors are provided as (batch, seq, feature)
        """
        super(MinRNNPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.n_layers = n_layers
        self.rnn_type = rnn_type.lower()
        
        # Create a list to hold RNN layers
        self.rnn_layers = nn.ModuleList()
        
        # First layer input size
        rnn_input_size = input_size
        
        # Create RNN layers
        for i in range(n_layers):
            if self.rnn_type == 'mingru':
                rnn_layer = MinGRU(rnn_input_size, hidden_size)
            elif self.rnn_type == 'minlstm':
                rnn_layer = MinLSTM(rnn_input_size, hidden_size)
            else:
                raise ValueError("rnn_type must be 'minGRU' or 'minLSTM'")
            self.rnn_layers.append(rnn_layer)
            # After first layer, input size is hidden_size
            rnn_input_size = hidden_size
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize h0 for each layer
        self.h0 = nn.ParameterList([nn.Parameter(torch.zeros(1, hidden_size), requires_grad=False) for _ in range(n_layers)])
    
    def forward(self, x):
        """
        x: Input sequence of shape (batch_size, seq_length, input_size) if batch_first=True
           or (seq_length, batch_size, input_size) if batch_first=False
        """
        if not self.batch_first:
            # If input is (seq_length, batch_size, input_size), transpose to (batch_size, seq_length, input_size)
            x = x.transpose(0, 1)
        
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Process through RNN layers
        output = x
        for layer_idx, rnn_layer in enumerate(self.rnn_layers):
            # Expand h0 to match batch size
            h0 = self.h0[layer_idx].expand(batch_size, -1)  # Shape: (batch_size, hidden_size)
            output = rnn_layer(output, h0)  # Shape: (batch_size, seq_length, hidden_size)
        
        # Pass each time step's hidden state through the fully connected layer
        output = self.fc(output)  # Shape: (batch_size, seq_length, output_size)
        
        return output  # Final prediction at each time step

# Minimal GRU implementation
class MinGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinGRU, self).__init__()
        self.hidden_size = hidden_size
        self.linear_z = nn.Linear(input_size, hidden_size)
        self.linear_h_tilde = nn.Linear(input_size, hidden_size)
                
    def forward(self, x, h_0):
        batch_size, seq_length, _ = x.size()
        z_t = torch.sigmoid(self.linear_z(x))
        h_tilde = self.linear_h_tilde(x)
        
        # Initialize hidden states tensor
        h_t = torch.zeros(batch_size, seq_length, self.hidden_size, device=x.device)
        
        # Initialize h_prev
        h_prev = h_0  # Shape: (batch_size, hidden_size)
    
        # Compute hidden states
        for t in range(seq_length):
            h_prev = (1 - z_t[:, t, :]) * h_prev + z_t[:, t, :] * h_tilde[:, t, :]
            h_t[:, t, :] = h_prev
            
        return h_t  # Shape: (batch_size, seq_length, hidden_size)

# Minimal LSTM implementation
class MinLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.linear_f = nn.Linear(input_size, hidden_size)
        self.linear_i = nn.Linear(input_size, hidden_size)
        self.linear_h_tilde = nn.Linear(input_size, hidden_size)
                
    def forward(self, x, h_0):
        batch_size, seq_length, _ = x.size()
        f_t = torch.sigmoid(self.linear_f(x))
        i_t = torch.sigmoid(self.linear_i(x))
        h_tilde = self.linear_h_tilde(x)
        
        # Normalize gates so that f_t + i_t = 1
        gate_sum = f_t + i_t + 1e-7  # Small epsilon to avoid division by zero
        f_t_norm = f_t / gate_sum
        i_t_norm = i_t / gate_sum
        
        # Initialize hidden states tensor
        h_t = torch.zeros(batch_size, seq_length, self.hidden_size, device=x.device)
        
        # Initialize h_prev
        h_prev = h_0  # Shape: (batch_size, hidden_size)
    
        # Compute hidden states
        for t in range(seq_length):
            h_prev = f_t_norm[:, t, :] * h_prev + i_t_norm[:, t, :] * h_tilde[:, t, :]
            h_t[:, t, :] = h_prev
            
        return h_t  # Shape: (batch_size, seq_length, hidden_size)
