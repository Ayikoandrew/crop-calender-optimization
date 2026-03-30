
"""Crop Calendar Optimization RL Environment."""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

import gymnasium as gym
from gymnasium import spaces


@dataclass
class CropVariety:
    """Crop variety parameters."""
    name: str
    cycle_length: int
    water_requirement: float
    potential_yield: float
    drought_tolerance: float
    emergence_frac: float = 0.1
    vegetative_frac: float = 0.3
    flowering_frac: float = 0.2
    grain_filling_frac: float = 0.3
    maturity_frac: float = 0.1
    stage_sensitivity: Dict[str, float] = field(default_factory=lambda: {
        'emergence': 0.8, 'vegetative': 0.6, 'flowering': 1.0,
        'grain_filling': 0.9, 'maturity': 0.3
    })


CROP_VARIETIES = {
    'short': CropVariety(
        name='Early Maturing', cycle_length=90,
        water_requirement=4.5, potential_yield=3.5, drought_tolerance=0.7
    ),
    'medium': CropVariety(
        name='Medium Maturing', cycle_length=110,
        water_requirement=5.0, potential_yield=5.0, drought_tolerance=0.5
    ),
    'long': CropVariety(
        name='Late Maturing', cycle_length=130,
        water_requirement=5.5, potential_yield=7.0, drought_tolerance=0.3
    )
}


class CropGrowthModel:
    """Simplified crop growth model."""
    
    def __init__(self, variety: CropVariety, 
                soil_water_capacity: float = 150.0,
                initial_soil_water: float = 50.0):
        self.variety = variety
        self.soil_water_capacity = soil_water_capacity
        self.initial_soil_water = initial_soil_water
        self.reset()
    
    def reset(self):
        self.soil_water = self.initial_soil_water
        self.days_since_planting = 0
        self.is_planted = False
        self.is_mature = False
        self.cumulative_stress = 0.0
        self.stress_by_stage = {}
        self.daily_stress_history = []
        self.daily_soil_water_history = []
        
    def plant(self):
        self.is_planted = True
        self.days_since_planting = 0
        
    def get_current_stage(self) -> str:
        if not self.is_planted:
            return 'not_planted'
        frac = self.days_since_planting / self.variety.cycle_length
        cum = 0
        for stage, stage_frac in [
            ('emergence', self.variety.emergence_frac),
            ('vegetative', self.variety.vegetative_frac),
            ('flowering', self.variety.flowering_frac),
            ('grain_filling', self.variety.grain_filling_frac),
            ('maturity', self.variety.maturity_frac)
        ]:
            cum += stage_frac
            if frac <= cum:
                return stage
        return 'mature'
    
    def compute_et(self, temperature: float = 25.0) -> float:
        if not self.is_planted:
            return 2.0
        stage = self.get_current_stage()
        kc = {'emergence': 0.4, 'vegetative': 0.8, 'flowering': 1.2,
            'grain_filling': 1.0, 'maturity': 0.6}.get(stage, 0.5)
        et0 = 0.0023 * (temperature + 17.8) * 15 * 0.5
        return kc * et0
    
    def compute_water_stress(self, et_demand: float) -> float:
        if self.soil_water <= 0:
            return 1.0
        available_frac = self.soil_water / self.soil_water_capacity
        critical_frac = 0.5
        if available_frac >= critical_frac:
            stress = 0.0
        else:
            stress = 1 - (available_frac / critical_frac)
        stress *= (1 - self.variety.drought_tolerance * 0.5)
        return min(1.0, max(0.0, stress))
    
    def step(self, rainfall: float, temperature: float = 25.0) -> Dict[str, float]:
        self.soil_water += rainfall
        self.soil_water = min(self.soil_water, self.soil_water_capacity)
        et = self.compute_et(temperature)
        stress = self.compute_water_stress(et) if self.is_planted else 0.0
        actual_et = et * (1 - stress * 0.5)
        self.soil_water = max(0, self.soil_water - actual_et)
        
        if self.is_planted and not self.is_mature:
            stage = self.get_current_stage()
            sensitivity = self.variety.stage_sensitivity.get(stage, 0.5)
            self.cumulative_stress += stress * sensitivity
            self.daily_stress_history.append(stress)
            self.days_since_planting += 1
            if self.days_since_planting >= self.variety.cycle_length:
                self.is_mature = True
        
        self.daily_soil_water_history.append(self.soil_water)
        return {'rainfall': rainfall, 'et': actual_et, 'soil_water': self.soil_water,
                'stress': stress, 'stage': self.get_current_stage(),
                'days_since_planting': self.days_since_planting}
    
    def compute_yield(self) -> float:
        if not self.is_planted:
            return 0.0
        avg_stress = self.cumulative_stress / max(1, self.days_since_planting)
        yield_factor = np.exp(-2.0 * avg_stress)
        maturity_factor = min(1.0, self.days_since_planting / self.variety.cycle_length)
        return self.variety.potential_yield * yield_factor * maturity_factor


class CropCalendarEnv(gym.Env):
    """Gymnasium environment for crop planting calendar optimization."""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, rainfall_data: np.ndarray,
                region: str = 'Region1',
                season_start_doy: int = 80,
                season_duration: int = 150,
                max_planting_window: int = 60,
                render_mode: Optional[str] = None):
        super().__init__()
        
        self.rainfall_data = rainfall_data
        self.region = region
        self.season_start_doy = season_start_doy
        self.season_duration = season_duration
        self.max_planting_window = max_planting_window
        self.render_mode = render_mode
        self.varieties = CROP_VARIETIES
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        
        self.current_day = 0
        self.crop = None
        self.episode_reward = 0
        self.history = []
        
    def _get_obs(self) -> np.ndarray:
        doy = self.season_start_doy + self.current_day
        doy_sin = np.sin(2 * np.pi * doy / 365)
        doy_cos = np.cos(2 * np.pi * doy / 365)
        
        rain_7d = self.rainfall_data[max(0, self.current_day-7):self.current_day].sum() / 50.0
        rain_14d = self.rainfall_data[max(0, self.current_day-14):self.current_day].sum() / 100.0
        rain_30d = self.rainfall_data[max(0, self.current_day-30):self.current_day].sum() / 200.0
        
        soil_moisture = self.crop.soil_water / self.crop.soil_water_capacity
        
        future_start = self.current_day
        future_end = min(self.current_day + 7, len(self.rainfall_data))
        forecast = self.rainfall_data[future_start:future_end].mean() / 10.0 if future_end > future_start else 0.0
        
        is_planted = float(self.crop.is_planted)
        days_since_planting = self.crop.days_since_planting / 130.0
        
        stage_map = {'not_planted': 0.0, 'emergence': 0.2, 'vegetative': 0.4,
                    'flowering': 0.6, 'grain_filling': 0.8, 'maturity': 1.0, 'mature': 1.0}
        growth_stage = stage_map.get(self.crop.get_current_stage(), 0.0)
        season_progress = self.current_day / self.season_duration
        
        return np.array([doy_sin, doy_cos, self.current_day / self.season_duration,
                        rain_7d, rain_14d, rain_30d, soil_moisture, forecast,
                        is_planted, days_since_planting, growth_stage, season_progress],
                    dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        return {'day': self.current_day, 'is_planted': self.crop.is_planted,
                'days_since_planting': self.crop.days_since_planting,
                'soil_water': self.crop.soil_water,
                'cumulative_stress': self.crop.cumulative_stress,
                'current_yield': self.crop.compute_yield() if self.crop.is_planted else 0.0}
    
    def reset(self, *, seed: Optional[int] = None,
            options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_day = 0
        self.crop = CropGrowthModel(self.varieties['medium'])
        self.episode_reward = 0
        self.history = []
        return self._get_obs(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        reward = 0.0
        terminated = False
        
        if not self.crop.is_planted:
            if action == 0:
                if self.crop.soil_water > 50:
                    reward -= 0.01
            elif action in [1, 2, 3]:
                variety_key = {1: 'short', 2: 'medium', 3: 'long'}[action]
                self.crop = CropGrowthModel(self.varieties[variety_key])
                self.crop.plant()
                if self.crop.soil_water > 60:
                    reward += 0.1
                days_remaining = self.season_duration - self.current_day
                if days_remaining < self.crop.variety.cycle_length:
                    reward -= 0.5
        
        daily_rain = self.rainfall_data[self.current_day] if self.current_day < len(self.rainfall_data) else 0
        self.crop.step(daily_rain)
        self.current_day += 1
        
        if self.crop.is_mature:
            reward += self.crop.compute_yield()
            terminated = True
        elif self.current_day >= self.max_planting_window and not self.crop.is_planted:
            reward -= 1.0
            terminated = True
        elif self.current_day >= self.season_duration:
            if self.crop.is_planted:
                reward += self.crop.compute_yield() * 0.5
            terminated = True
        
        self.episode_reward += reward
        return self._get_obs(), reward, terminated, False, self._get_info()
