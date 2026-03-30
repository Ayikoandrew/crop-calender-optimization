
import torch
import torch.nn as nn
import numpy as np


class RainfallLSTM(nn.Module):
    """LSTM-based rainfall forecasting model."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, 
                num_regions=5, region_embed_dim=8,
                forecast_horizon=14, dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        self.region_embedding = nn.Embedding(num_regions, region_embed_dim)
        
        self.lstm = nn.LSTM(
            input_size=input_dim + region_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.fc_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon)
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon)
        )
        
    def forward(self, x, region_ids):
        batch_size, seq_len, _ = x.shape
        region_emb = self.region_embedding(region_ids)
        region_emb = region_emb.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([x, region_emb], dim=-1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        mean = self.fc_mean(last_hidden)
        logvar = self.fc_logvar(last_hidden)
        return mean, logvar
    
    def predict(self, x, region_ids, num_samples=100):
        self.eval()
        with torch.no_grad():
            mean, logvar = self.forward(x, region_ids)
            std = torch.exp(0.5 * logvar)
            samples = torch.randn(num_samples, *mean.shape, device=mean.device)
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * samples
            return {'mean': mean, 'std': std, 'samples': samples}


class RainfallWorldModel:
    """Wrapper for using LSTM as World Model in RL."""
    
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        self.scaler_params = checkpoint['scaler_params']
        self.model = RainfallLSTM(**self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        target_idx = self.scaler_params['features'].index('Mean')
        self.target_mean = self.scaler_params['mean'][target_idx]
        self.target_scale = self.scaler_params['scale'][target_idx]
        
    def predict(self, sequence, region_id, return_samples=False, num_samples=100):
        if isinstance(sequence, np.ndarray):
            sequence = torch.tensor(sequence, dtype=torch.float32)
        sequence = sequence.unsqueeze(0).to(self.device)
        region_tensor = torch.tensor([region_id], dtype=torch.long).to(self.device)
        with torch.no_grad():
            result = self.model.predict(sequence, region_tensor, num_samples)
            mean = result['mean'].squeeze(0).cpu().numpy()
            std = result['std'].squeeze(0).cpu().numpy()
            mean = mean * self.target_scale + self.target_mean
            std = std * self.target_scale
            output = {'mean': mean, 'std': std}
            if return_samples:
                samples = result['samples'].squeeze(1).cpu().numpy()
                samples = samples * self.target_scale + self.target_mean
                output['samples'] = samples
            return output
    
    def step(self, sequence, region_id):
        result = self.predict(sequence, region_id, return_samples=False)
        mean = result['mean'][0]
        std = result['std'][0]
        rainfall = max(0, np.random.normal(mean, std))
        return rainfall, {'mean': mean, 'std': std}
