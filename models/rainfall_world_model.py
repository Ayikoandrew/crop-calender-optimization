
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


class RainfallEncoder(nn.Module):
    """Encode seasonal rainfall to latent space."""
    
    def __init__(self, seq_length=150, latent_dim=8):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.flat_size = 128 * 19
        
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
    def forward(self, x):
        h = torch.nn.functional.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        h = torch.nn.functional.leaky_relu(self.bn2(self.conv2(h)), 0.2)
        h = torch.nn.functional.leaky_relu(self.bn3(self.conv3(h)), 0.2)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class RainfallDecoder(nn.Module):
    """Decode latent space to seasonal rainfall."""
    
    def __init__(self, seq_length=150, latent_dim=8):
        super().__init__()
        
        self.seq_length = seq_length
        self.flat_size = 128 * 19
        
        self.fc = nn.Linear(latent_dim, self.flat_size)
        
        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=0)
        self.deconv3 = nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        
    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 128, 19)
        h = torch.nn.functional.leaky_relu(self.bn1(self.deconv1(h)), 0.2)
        h = torch.nn.functional.leaky_relu(self.bn2(self.deconv2(h)), 0.2)
        h = torch.sigmoid(self.deconv3(h))
        if h.size(2) != self.seq_length:
            h = torch.nn.functional.interpolate(h, size=self.seq_length, mode='linear', align_corners=False)
        return h


class RainfallVAE(nn.Module):
    """Variational Autoencoder for seasonal rainfall patterns."""
    
    def __init__(self, seq_length=150, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = RainfallEncoder(seq_length, latent_dim)
        self.decoder = RainfallDecoder(seq_length, latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z):
        return self.decoder(z)


class RainfallVAEGenerator:
    """
    Wrapper for using trained VAE in RL environment.
    Uses perturbation-based sampling with scaling correction.
    """
    
    def __init__(self, model_path, noise_scale=0.3, device='cpu'):
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.latent_dim = checkpoint['latent_dim']
        self.seq_length = checkpoint['seq_length']
        self.norm_min = checkpoint['norm_params']['min']
        self.norm_max = checkpoint['norm_params']['max']
        self.noise_scale = noise_scale
        self.scale_factor = checkpoint.get('scale_factor', 1.0)
        
        self.latent_vectors = checkpoint['latent_vectors']
        self.latent_std = checkpoint['latent_std']
        self.latent_mean = checkpoint['latent_mean']
        
        self.model = RainfallVAE(self.seq_length, self.latent_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def denormalize(self, x_norm):
        x_log = x_norm * (self.norm_max - self.norm_min) + self.norm_min
        return np.expm1(x_log) * self.scale_factor
    
    def generate_random(self, n_seasons=1):
        """Generate random seasons by perturbing real encoded latent vectors."""
        base_indices = np.random.choice(len(self.latent_vectors), size=n_seasons, replace=True)
        z_list = []
        
        for idx in base_indices:
            noise = np.random.randn(self.latent_dim) * self.noise_scale * self.latent_std
            z_list.append(self.latent_vectors[idx] + noise)
        
        z = torch.tensor(np.array(z_list), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            samples = self.model.decode(z).cpu().numpy().squeeze()
        
        if n_seasons == 1:
            samples = samples.reshape(1, -1)
        return self.denormalize(samples)
    
    def generate_shifted(self, shift=0.0, dim=None):
        """Generate season with shifted climate (from mean latent)."""
        z = torch.tensor(self.latent_mean, dtype=torch.float32).unsqueeze(0).to(self.device)
        if dim is not None:
            z[0, dim] += shift * self.latent_std[dim]
        
        with torch.no_grad():
            sample = self.model.decode(z).cpu().numpy().squeeze()
        return self.denormalize(sample)
    
    def interpolate(self, z1, z2, alpha):
        """Interpolate between two latent vectors."""
        z = (1 - alpha) * np.array(z1) + alpha * np.array(z2)
        z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            sample = self.model.decode(z_tensor).cpu().numpy().squeeze()
        return self.denormalize(sample)
    
    def decode_latent(self, z):
        """Decode a latent vector to rainfall sequence."""
        z_tensor = torch.tensor(z, dtype=torch.float32).to(self.device)
        if z_tensor.dim() == 1:
            z_tensor = z_tensor.unsqueeze(0)
        
        with torch.no_grad():
            sample = self.model.decode(z_tensor).cpu().numpy().squeeze()
        return self.denormalize(sample)



class ConditionalScenarioGenerator:
    """
    Generates rainfall scenarios conditioned on LSTM short-term forecasts.
    
    Combines:
    - LSTM: Accurate short-term forecasts (14 days) with uncertainty
    - VAE: Realistic seasonal patterns (150 days)
    
    Strategies:
    1. Latent conditioning: Select VAE patterns consistent with LSTM forecast
    2. Hybrid generation: LSTM for early season, VAE for full structure
    3. Uncertainty scaling: Use LSTM uncertainty to adjust VAE variability
    """
    
    def __init__(self, lstm_path, vae_path, device='cpu'):
        """
        Initialize with trained LSTM and VAE models.
        
        Args:
            lstm_path: Path to trained LSTM checkpoint
            vae_path: Path to trained VAE checkpoint
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        
        # Load LSTM
        self.lstm = RainfallWorldModel(lstm_path, device=device)
        
        # Load VAE
        self.vae = RainfallVAEGenerator(vae_path, device=device)
        
        # Precompute decoded patterns for all stored latent vectors
        self._precompute_decoded_patterns()
    
    def _precompute_decoded_patterns(self):
        """Decode all stored latent vectors for efficient matching."""
        self.decoded_patterns = []
        
        for z in self.vae.latent_vectors:
            pattern = self.vae.decode_latent(z)
            self.decoded_patterns.append(pattern)
        
        self.decoded_patterns = np.array(self.decoded_patterns)
    
    def generate_conditioned(self, sequence, region_id, n_scenarios=100, 
                            match_days=14, top_k_fraction=0.3):
        """
        Generate VAE scenarios conditioned on LSTM short-term forecast.
        
        Args:
            sequence: Input sequence for LSTM (normalized features)
            region_id: Region identifier for LSTM
            n_scenarios: Number of scenarios to generate
            match_days: Number of days to match with LSTM forecast
            top_k_fraction: Fraction of best-matching patterns to use
        
        Returns:
            Array of shape (n_scenarios, season_length) with rainfall values
        """
        # Get LSTM forecast distribution
        lstm_pred = self.lstm.predict(sequence, region_id)
        forecast_mean = lstm_pred['mean'][:match_days]
        forecast_std = lstm_pred['std'][:match_days]
        
        # Score each decoded pattern by similarity to LSTM forecast
        similarities = []
        for i, pattern in enumerate(self.decoded_patterns):
            pattern_early = pattern[:match_days]
            
            # Weighted difference (less penalty where LSTM is uncertain)
            weights = 1.0 / (forecast_std + 1e-6)
            weights = weights / weights.sum()
            
            diff = np.abs(pattern_early - forecast_mean)
            weighted_diff = (diff * weights).sum()
            
            similarities.append((i, weighted_diff))
        
        # Select best matching patterns
        similarities.sort(key=lambda x: x[1])
        top_k = max(1, int(len(similarities) * top_k_fraction))
        best_indices = [s[0] for s in similarities[:top_k]]
        
        # Generate scenarios by perturbing matched latents
        scenarios = []
        scenarios_per_match = max(1, n_scenarios // len(best_indices))
        
        for idx in best_indices:
            for _ in range(scenarios_per_match):
                if len(scenarios) >= n_scenarios:
                    break
                
                # Perturb the matched latent vector
                noise = np.random.randn(self.vae.latent_dim) * 0.2 * self.vae.latent_std
                z_perturbed = self.vae.latent_vectors[idx] + noise
                
                scenario = self.vae.decode_latent(z_perturbed)
                scenarios.append(scenario)
        
        return np.array(scenarios[:n_scenarios])
    
    def generate_hybrid(self, sequence, region_id, n_scenarios=100,
                    lstm_days=14, blend_days=7):
        """
        Generate hybrid scenarios: LSTM short-term + VAE seasonal structure.
        
        Args:
            sequence: Input sequence for LSTM
            region_id: Region identifier
            n_scenarios: Number of scenarios
            lstm_days: Days to use pure LSTM sampling
            blend_days: Days to blend LSTM and VAE
        
        Returns:
            Array of shape (n_scenarios, season_length)
        """
        season_length = self.vae.seq_length
        scenarios = []
        
        for _ in range(n_scenarios):
            # Generate LSTM trajectory for first lstm_days
            lstm_part = []
            current_seq = sequence.copy()
            
            for day in range(lstm_days):
                pred = self.lstm.predict(current_seq, region_id)
                rain = max(0, np.random.normal(pred['mean'][0], pred['std'][0]))
                lstm_part.append(rain)
                
                # Simplified sequence update (shift and add new value)
                current_seq = np.roll(current_seq, -1, axis=0)
                # Note: In practice, should update all features properly
            
            # Generate matching VAE scenario
            vae_scenario = self.generate_conditioned(
                sequence, region_id, n_scenarios=1, match_days=lstm_days
            )[0]
            
            # Create blended scenario
            full_scenario = np.zeros(season_length)
            
            # Pure LSTM region
            full_scenario[:lstm_days - blend_days] = lstm_part[:lstm_days - blend_days]
            
            # Blend region (smooth transition)
            for i in range(blend_days):
                day_idx = lstm_days - blend_days + i
                alpha = i / blend_days  # 0 to 1
                full_scenario[day_idx] = (1 - alpha) * lstm_part[day_idx] + alpha * vae_scenario[day_idx]
            
            # Pure VAE region
            full_scenario[lstm_days:] = vae_scenario[lstm_days:]
            
            scenarios.append(full_scenario)
        
        return np.array(scenarios)
    
    def generate_uncertainty_scaled(self, sequence, region_id, n_scenarios=100,
                                base_noise=0.2, uncertainty_weight=0.5):
        """
        Generate VAE scenarios with noise scaled by LSTM forecast uncertainty.
        
        Higher LSTM uncertainty → more VAE variation.
        
        Args:
            sequence: Input sequence for LSTM
            region_id: Region identifier
            n_scenarios: Number of scenarios
            base_noise: Base noise scale for VAE perturbation
            uncertainty_weight: How much LSTM uncertainty affects noise scale
        
        Returns:
            Array of shape (n_scenarios, season_length)
        """
        # Get LSTM forecast uncertainty
        pred = self.lstm.predict(sequence, region_id)
        avg_uncertainty = pred['std'].mean()
        
        # Scale noise based on uncertainty (higher uncertainty → more variation)
        # Normalize uncertainty to typical range
        typical_std = 5.0  # Approximate typical rainfall std
        normalized_uncertainty = avg_uncertainty / typical_std
        
        noise_scale = base_noise + uncertainty_weight * normalized_uncertainty
        noise_scale = np.clip(noise_scale, 0.1, 0.8)
        
        # Temporarily adjust VAE noise scale
        original_noise = self.vae.noise_scale
        self.vae.noise_scale = noise_scale
        
        scenarios = self.vae.generate_random(n_scenarios)
        
        # Restore original noise scale
        self.vae.noise_scale = original_noise
        
        return scenarios
    
    def generate_ensemble(self, sequence, region_id, n_scenarios=100):
        """
        Generate ensemble combining all strategies for robust scenario set.
        
        Returns scenarios from:
        - Conditioned generation (40%)
        - Hybrid generation (30%)
        - Uncertainty-scaled generation (30%)
        """
        n_conditioned = int(n_scenarios * 0.4)
        n_hybrid = int(n_scenarios * 0.3)
        n_uncertainty = n_scenarios - n_conditioned - n_hybrid
        
        conditioned = self.generate_conditioned(sequence, region_id, n_conditioned)
        hybrid = self.generate_hybrid(sequence, region_id, n_hybrid)
        uncertainty = self.generate_uncertainty_scaled(sequence, region_id, n_uncertainty)
        
        ensemble = np.vstack([conditioned, hybrid, uncertainty])
        
        # Shuffle to mix strategies
        np.random.shuffle(ensemble)
        
        return ensemble
    
    def get_scenario_statistics(self, scenarios):
        """
        Compute summary statistics for generated scenarios.
        
        Args:
            scenarios: Array of shape (n_scenarios, season_length)
        
        Returns:
            Dict with statistics
        """
        totals = scenarios.sum(axis=1)
        rainy_days = (scenarios > 1).sum(axis=1)
        max_daily = scenarios.max(axis=1)
        
        return {
            'total_rainfall': {
                'mean': totals.mean(),
                'std': totals.std(),
                'min': totals.min(),
                'max': totals.max(),
                'percentiles': np.percentile(totals, [10, 25, 50, 75, 90])
            },
            'rainy_days': {
                'mean': rainy_days.mean(),
                'std': rainy_days.std()
            },
            'max_daily_rainfall': {
                'mean': max_daily.mean(),
                'std': max_daily.std()
            },
            'n_scenarios': len(scenarios)
        }
