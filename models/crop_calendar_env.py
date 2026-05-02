
"""
Crop Calendar Optimization RL Environment
==========================================

A Gymnasium-compatible reinforcement learning environment for optimizing 
crop planting decisions under rainfall uncertainty. The agent learns to 
select appropriate crop varieties and planting timing based on weather 
conditions and seasonal forecasts.

Environment Design
------------------
This environment models a single growing season where an agricultural agent
must decide:
1. WHEN to plant (timing within a planting window)
2. WHAT to plant (short, medium, or long-cycle crop variety)

The environment incorporates:
- Historical rainfall data with stochastic sampling
- Simplified water-balance crop growth model
- Phenological stage-dependent water stress sensitivity
- Domain randomization (rainfall scaling, initial soil moisture)

Observation Space (13 dimensions)
---------------------------------
- Day of year (sin/cos encoded): 2 dims
- Season progress: 1 dim
- Recent rainfall (7d, 14d, 30d sums, normalized): 3 dims
- Soil moisture fraction: 1 dim
- 7-day rainfall forecast (normalized): 1 dim
- Crop status (planted flag, days since planting, growth stage): 3 dims
- Season progress duplicate: 1 dim
- Rainfall scale indicator (dry/wet year): 1 dim

Action Space (Discrete, 4 actions)
----------------------------------
- 0: Wait (do not plant)
- 1: Plant short-cycle variety (90 days, drought-tolerant)
- 2: Plant medium-cycle variety (110 days, balanced)
- 3: Plant long-cycle variety (130 days, high-yield but drought-sensitive)

Reward Structure
----------------
- Planting: Weather-condition scoring + variety-weather matching bonus
- During growth: Minimal waiting penalties (only when season runs out)
- Harvest: Efficiency-based reward (yield / potential_yield × 5.0)
- Failure penalties: -2.0 for not planting or incomplete season

References
----------
- FAO Crop Water Requirements (FAO Irrigation and Drainage Paper 56)
- Allen et al. (1998) for ET0 estimation
- Steduto et al. (2012) for crop yield response to water (AquaCrop)

Author: [Your Name]
Date: April 2026
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

import gymnasium as gym
from gymnasium import spaces


@dataclass
class CropVariety:
    """
    Crop variety parameterization for growth simulation.
    
    This dataclass encapsulates the agronomic characteristics of a crop
    variety, including its growth cycle, water requirements, yield potential,
    and drought tolerance. The phenological stage fractions must sum to 1.0.
    
    Attributes
    ----------
    name : str
        Human-readable variety name (e.g., "Early Maturing").
    cycle_length : int
        Total growing period from planting to maturity in days.
    water_requirement : float
        Cumulative water requirement over the growing season (mm/day average).
        Used conceptually; actual water use computed via ET model.
    potential_yield : float
        Maximum achievable yield under optimal conditions (t/ha).
    drought_tolerance : float
        Tolerance to water stress, range [0, 1]. Higher values indicate
        greater resilience. Affects stress multiplier and yield loss curves.
        - 0.9 (high): Short-cycle, drought-adapted varieties
        - 0.6 (medium): Balanced varieties
        - 0.15 (low): Long-cycle, water-demanding varieties
    emergence_frac : float
        Fraction of cycle spent in emergence stage (default: 0.1).
    vegetative_frac : float
        Fraction of cycle spent in vegetative stage (default: 0.3).
    flowering_frac : float
        Fraction of cycle spent in flowering stage (default: 0.2).
    grain_filling_frac : float
        Fraction of cycle spent in grain filling (default: 0.3).
    maturity_frac : float
        Fraction of cycle spent in maturation (default: 0.1).
    stage_sensitivity : Dict[str, float]
        Water stress sensitivity by phenological stage. Values in [0, 1]
        where 1.0 indicates maximum sensitivity (e.g., flowering).
        
    Notes
    -----
    The stage fractions are based on generalized maize phenology but can
    be adjusted for other crops. The stage sensitivities follow the
    critical period concept where flowering is most sensitive to drought.
    """
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


# =============================================================================
# CROP VARIETY DEFINITIONS
# =============================================================================
# Three varieties representing the risk-return tradeoff in crop selection:
#
# | Variety | Cycle | Yield (t/ha) | Drought Tol. | Strategy               |
# |---------|-------|--------------|--------------|------------------------|
# | Short   | 90d   | 4.5          | 0.90 (high)  | Safe, dry years        |
# | Medium  | 110d  | 5.5          | 0.60 (med)   | Balanced, normal years |
# | Long    | 130d  | 7.0          | 0.40 (mod)   | High-yield, wet years  |
#
# The drought tolerance dramatically affects yield under water stress through
# a nonlinear tolerance multiplier in the stress calculation.
# =============================================================================

CROP_VARIETIES = {
    'short': CropVariety(
        name='Early Maturing', cycle_length=90,
        water_requirement=3.5, potential_yield=4.5, drought_tolerance=0.9  
    ),
    'medium': CropVariety(
        name='Medium Maturing', cycle_length=110,
        water_requirement=5.0, potential_yield=5.5, drought_tolerance=0.6  
    ),
    'long': CropVariety(
        name='Late Maturing', cycle_length=130,
        water_requirement=7.0, potential_yield=7.0, drought_tolerance=0.40  
    )
}


class CropGrowthModel:
    """
    Simplified daily water-balance crop growth model.
    
    This model simulates crop growth using a soil water balance approach,
    tracking daily water inputs (rainfall) and outputs (evapotranspiration).
    Water stress accumulates based on soil moisture deficit and affects
    final yield through phenological stage-weighted stress factors.
    
    The model is inspired by FAO's AquaCrop but significantly simplified
    for computational efficiency in RL training. Key simplifications:
    - Single-layer soil water balance
    - Temperature assumed constant (25°C)
    - No nutrient or pest stress
    - Instantaneous planting (no seedling establishment period)
    
    Attributes
    ----------
    variety : CropVariety
        The planted crop variety parameterization.
    soil_water_capacity : float
        Maximum soil water holding capacity (mm). Default: 150 mm,
        typical for medium-textured soils at 1m depth.
    initial_soil_water : float
        Starting soil moisture level (mm).
    soil_water : float
        Current soil water content (mm).
    cumulative_stress : float
        Accumulated water stress weighted by stage sensitivity.
    daily_stress_history : list
        Record of daily stress values for analysis.
        
    Methods
    -------
    reset()
        Reset model state for new season.
    plant()
        Initialize planting, setting days_since_planting to 0.
    get_current_stage() -> str
        Return current phenological stage.
    step(rainfall, temperature) -> dict
        Advance one day, update water balance and stress.
    compute_yield() -> float
        Calculate final yield based on accumulated stress.
        
    Notes
    -----
    The water stress calculation includes a tolerance multiplier that
    makes low-tolerance varieties (e.g., long-cycle) extremely sensitive:
    
        tolerance_multiplier = 3.0 × (1 - drought_tolerance)
        
    For long variety (tolerance=0.15): multiplier = 2.55 (catastrophic)
    For short variety (tolerance=0.90): multiplier = 0.30 (resilient)
    
    This creates the risk-return tradeoff where high-yield varieties
    fail catastrophically in drought conditions.
    """
    
    def __init__(self, variety: CropVariety, 
                soil_water_capacity: float = 150.0,
                initial_soil_water: float = 50.0):
        """
        Initialize crop growth model.
        
        Parameters
        ----------
        variety : CropVariety
            Crop variety to simulate.
        soil_water_capacity : float, optional
            Maximum soil water holding capacity in mm.
        initial_soil_water : float, optional
            Initial soil moisture in mm.
        """
        self.variety = variety
        self.soil_water_capacity = soil_water_capacity
        self.initial_soil_water = initial_soil_water
        self.reset()
    
    def reset(self):
        """Reset model state for a new growing season."""
        self.soil_water = self.initial_soil_water
        self.days_since_planting = 0
        self.is_planted = False
        self.is_mature = False
        self.cumulative_stress = 0.0
        self.stress_by_stage = {}
        self.daily_stress_history = []
        self.daily_soil_water_history = []
        
    def plant(self):
        """Mark crop as planted and initialize day counter."""
        self.is_planted = True
        self.days_since_planting = 0
        
    def get_current_stage(self) -> str:
        """
        Determine current phenological stage based on days since planting.
        
        Returns
        -------
        str
            One of: 'not_planted', 'emergence', 'vegetative', 'flowering',
            'grain_filling', 'maturity', or 'mature'.
        """
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
        """
        Compute daily evapotranspiration using simplified Hargreaves method.
        
        The ET is scaled by a crop coefficient (Kc) that varies by 
        phenological stage, following FAO-56 guidelines.
        
        Parameters
        ----------
        temperature : float
            Daily mean temperature in °C (default: 25°C).
            
        Returns
        -------
        float
            Daily ET in mm/day.
            
        Notes
        -----
        Kc values by stage:
        - Emergence: 0.4 (small canopy)
        - Vegetative: 0.8 (canopy developing)
        - Flowering: 1.2 (peak water demand)
        - Grain filling: 1.0 (high but declining)
        - Maturity: 0.6 (senescence)
        """
        if not self.is_planted:
            return 2.0  # Bare soil evaporation
        stage = self.get_current_stage()
        kc = {'emergence': 0.4, 'vegetative': 0.8, 'flowering': 1.2,
            'grain_filling': 1.0, 'maturity': 0.6}.get(stage, 0.5)
        # Simplified Hargreaves ET0 (assumes radiation proxy)
        et0 = 0.0023 * (temperature + 17.8) * 15 * 0.5
        return kc * et0
    
    def compute_water_stress(self, et_demand: float) -> float:
        """
        Compute water stress index based on soil moisture deficit.
        
        Stress increases when available soil water drops below a critical
        fraction (50% of capacity). The stress is amplified for varieties
        with low drought tolerance through a tolerance multiplier.
        
        Parameters
        ----------
        et_demand : float
            Daily ET demand in mm (not used in current implementation
            but available for future extensions).
            
        Returns
        -------
        float
            Water stress index in [0, 1], where 0 = no stress, 1 = maximum.
            
        Notes
        -----
        The tolerance multiplier creates dramatic differences:
        - Long variety (0.15 tolerance): 3.0 × 0.85 = 2.55× stress amplification
        - Short variety (0.90 tolerance): 3.0 × 0.10 = 0.30× stress reduction
        """
        if self.soil_water <= 0:
            return 1.0
        available_frac = self.soil_water / self.soil_water_capacity
        critical_frac = 0.5
        if available_frac >= critical_frac:
            stress = 0.0
        else:
            stress = 1 - (available_frac / critical_frac)
        # EXTREME drought sensitivity for low tolerance varieties
        # Long (0.15 tolerance) -> multiplier = 2.55 (catastrophic)
        # Short (0.9 tolerance) -> multiplier = 0.3 (resilient)
        tolerance_multiplier = 3.0 * (1 - self.variety.drought_tolerance)
        stress *= tolerance_multiplier
        return min(1.0, max(0.0, stress))
    
    def step(self, rainfall: float, temperature: float = 25.0) -> Dict[str, float]:
        """
        Advance simulation by one day.
        
        Updates soil water balance, computes stress, and advances
        phenological development.
        
        Parameters
        ----------
        rainfall : float
            Daily rainfall in mm.
        temperature : float
            Daily mean temperature in °C.
            
        Returns
        -------
        dict
            Daily simulation outputs: rainfall, et, soil_water, stress,
            stage, days_since_planting.
        """
        # Water balance: add rainfall (capped at capacity)
        self.soil_water += rainfall
        self.soil_water = min(self.soil_water, self.soil_water_capacity)
        
        # Compute ET and stress
        et = self.compute_et(temperature)
        stress = self.compute_water_stress(et) if self.is_planted else 0.0
        
        # Actual ET reduced under stress
        actual_et = et * (1 - stress * 0.5)
        self.soil_water = max(0, self.soil_water - actual_et)
        
        # Accumulate stress if planted
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
        """
        Calculate final yield based on accumulated water stress.
        
        Yield is computed as potential yield × stress factor × maturity factor.
        The stress factor uses a nonlinear exponential relationship where
        the exponent depends on drought tolerance.
        
        Returns
        -------
        float
            Final yield in t/ha.
            
        Notes
        -----
        The yield equation:
        
            yield = potential_yield × exp(-k × avg_stress) × maturity_frac
            
        Where k (stress exponent) varies by drought tolerance:
        - Long (0.15): k = 4.7 → exp(-4.7 × 0.3) = 0.24 (76% loss at 30% avg stress)
        - Short (0.9): k = 3.2 → stress values are already lower due to tolerance
        
        A germination penalty applies if first 14 days had >40% stress,
        reducing yield by up to 90% (crop establishment failure).
        """
        if not self.is_planted:
            return 0.0
        avg_stress = self.cumulative_stress / max(1, self.days_since_planting)
        
        # CATASTROPHIC yield loss for high stress on low-tolerance varieties
        # This makes Long variety fail badly in dry years
        stress_exponent = 3.0 + 2.0 * (1 - self.variety.drought_tolerance)
        # Long (0.15): exponent = 4.7 -> exp(-4.7 * 0.3) = 0.24 (76% loss!)
        # Short (0.9): exponent = 3.2 -> exp(-3.2 * 0.3) = 0.38 (but stress is lower)
        yield_factor = np.exp(-stress_exponent * avg_stress)
        maturity_factor = min(1.0, self.days_since_planting / self.variety.cycle_length)
        
        # Germination penalty: if first 14 days had high stress, apply extra penalty
        if len(self.daily_stress_history) >= 14:
            early_stress = np.mean(self.daily_stress_history[:14])
            if early_stress > 0.4:  # Lower threshold for germination failure
                germination_penalty = 1 - (early_stress - 0.4) * 1.5  # Up to 90% penalty
                yield_factor *= max(0.1, germination_penalty)
        
        return self.variety.potential_yield * yield_factor * maturity_factor


class CropCalendarEnv(gym.Env):
    """
    Gymnasium environment for crop planting calendar optimization.
    
    This environment simulates the decision problem faced by a farmer at
    the start of a growing season: when to plant and which crop variety
    to select, given uncertain rainfall conditions.
    
    The environment uses domain randomization to expose the agent to
    diverse conditions during training:
    - Random starting position in historical rainfall data
    - Random initial soil moisture (20-90 mm)
    - Random rainfall scaling (0.5-1.5×) simulating dry/wet years
    
    Episode Dynamics
    ----------------
    1. Agent observes weather conditions and forecasts
    2. Agent can wait (action=0) or plant a variety (action 1/2/3)
    3. Once planted, crop grows daily with water stress accumulation
    4. Episode ends when: crop matures, season ends, or planting window closes
    
    Reward Design
    -------------
    The reward structure encourages weather-adaptive variety selection:
    
    **Planting Rewards:**
    - Dry years (scale < 0.75): Short +1.0, Medium -0.3, Long -0.8
    - Normal (0.75-1.25): Medium +0.8, Short +0.2, Long +0.3
    - Wet years (scale > 1.25): Long +1.0, Medium -0.2, Short -0.5
    
    **Harvest Reward:**
    - Efficiency-based: (yield / potential_yield) × 5.0
    - Bonus for >80% efficiency: additional (ratio - 0.8) × 3.0
    - Small raw yield bonus: yield × 0.1
    
    **Penalties:**
    - Failed to plant: -2.0
    - Incomplete season: partial yield with -1.0 penalty
    
    Example
    -------
    >>> import gymnasium as gym
    >>> from crop_calendar_env import CropCalendarEnv
    >>> 
    >>> # Load rainfall data
    >>> rainfall_data = np.random.uniform(0, 20, 365*10)  # 10 years
    >>> 
    >>> env = gym.make('CropCalendar-v0', rainfall_data=rainfall_data)
    >>> obs, info = env.reset()
    >>> 
    >>> # Agent decides to wait
    >>> obs, reward, done, truncated, info = env.step(0)
    >>> 
    >>> # Agent decides to plant long-cycle variety
    >>> obs, reward, done, truncated, info = env.step(3)
    
    Attributes
    ----------
    rainfall_data : np.ndarray
        Historical daily rainfall time series (mm/day).
    season_start_doy : int
        Day of year when growing season begins (default: 80, ~March 21).
    season_duration : int
        Total season length in days (default: 150).
    max_planting_window : int
        Days within which planting must occur (default: 60).
    randomize_start : bool
        Whether to randomize starting conditions per episode.
    rainfall_scale : float
        Multiplier for rainfall (0.5=drought, 1.5=wet), randomized.
    
    See Also
    --------
    CropGrowthModel : The underlying crop simulation model.
    CropVariety : Variety parameterization dataclass.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(self, rainfall_data: np.ndarray,
                region: str = 'Region1',
                season_start_doy: int = 80,
                season_duration: int = 150,
                max_planting_window: int = 60,
                render_mode: Optional[str] = None,
                randomize_start: bool = True):
        """
        Initialize crop calendar optimization environment.
        
        Parameters
        ----------
        rainfall_data : np.ndarray
            Historical daily rainfall time series in mm/day. Should contain
            multiple years of data for stochastic episode sampling.
        region : str, optional
            Region identifier for logging (default: 'Region1').
        season_start_doy : int, optional
            Day of year when growing season begins (default: 80 ≈ March 21).
        season_duration : int, optional
            Total season length in days (default: 150).
        max_planting_window : int, optional
            Maximum days to wait before planting (default: 60).
        render_mode : str, optional
            Gymnasium render mode ('human' or None).
        randomize_start : bool, optional
            If True, randomize rainfall offset, soil moisture, and rainfall
            scale each episode for domain randomization (default: True).
        """
        super().__init__()
        
        self.rainfall_data = rainfall_data
        self.region = region
        self.season_start_doy = season_start_doy
        self.season_duration = season_duration
        self.max_planting_window = max_planting_window
        self.render_mode = render_mode
        self.randomize_start = randomize_start
        self.varieties = CROP_VARIETIES
        
        # Ensure we have enough data for random slicing
        self.min_data_length = season_duration + 30  # Buffer for lookback
        
        # =====================================================================
        # OBSERVATION SPACE (13 dimensions)
        # =====================================================================
        # Idx 0-1:  Day of year (sin/cos encoded) - seasonal position
        # Idx 2:    Season progress (0-1) - time remaining
        # Idx 3-5:  Recent rainfall (7d, 14d, 30d) - antecedent conditions
        # Idx 6:    Soil moisture fraction (0-1) - current water status
        # Idx 7:    7-day rainfall forecast - upcoming conditions
        # Idx 8:    Is planted flag (0/1)
        # Idx 9:    Days since planting (normalized by 130d)
        # Idx 10:   Growth stage (0-1 categorical encoding)
        # Idx 11:   Season progress (duplicate for policy convenience)
        # Idx 12:   Rainfall scale indicator (-0.5 to +0.5) - year type
        # =====================================================================
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        
        # =====================================================================
        # ACTION SPACE (4 discrete actions)
        # =====================================================================
        # 0: Wait (do not plant this day)
        # 1: Plant short-cycle variety (90 days, drought-tolerant)
        # 2: Plant medium-cycle variety (110 days, balanced)
        # 3: Plant long-cycle variety (130 days, high-yield, drought-sensitive)
        # =====================================================================
        self.action_space = spaces.Discrete(4)
        
        # Episode state
        self.current_day = 0
        self.crop = None
        self.episode_reward = 0
        self.history = []
        self.rainfall_offset = 0  # Starting offset in rainfall data
        self.initial_soil_water = 50.0  # Will be randomized
        self.rainfall_scale = 1.0  # Dry/wet year scaling (0.5-1.5)
        
        # =====================================================================
        # RENDERING STATE (pygame)
        # =====================================================================
        self.screen = None
        self.clock = None
        self.screen_width = 800
        self.screen_height = 600
        self.font = None
        self.font_small = None
        
    def _get_rainfall(self, day_idx: int) -> float:
        """
        Get rainfall for a given day index with offset and scaling applied.
        
        Parameters
        ----------
        day_idx : int
            Day index relative to season start.
            
        Returns
        -------
        float
            Scaled rainfall in mm for the given day.
        """
        abs_idx = self.rainfall_offset + day_idx
        if 0 <= abs_idx < len(self.rainfall_data):
            return self.rainfall_data[abs_idx] * self.rainfall_scale
        return 0.0
    
    def _get_rainfall_sum(self, start_day: int, end_day: int) -> float:
        """
        Get cumulative rainfall over a day range.
        
        Parameters
        ----------
        start_day : int
            Start day index (inclusive).
        end_day : int
            End day index (exclusive).
            
        Returns
        -------
        float
            Total rainfall in mm over the period.
        """
        total = 0.0
        for d in range(max(0, start_day), end_day):
            total += self._get_rainfall(d)
        return total
    
    def _get_obs(self) -> np.ndarray:
        """
        Construct observation vector from current environment state.
        
        Returns
        -------
        np.ndarray
            13-dimensional observation vector (see __init__ for details).
        """
        doy = self.season_start_doy + self.current_day
        doy_sin = np.sin(2 * np.pi * doy / 365)
        doy_cos = np.cos(2 * np.pi * doy / 365)
        
        rain_7d = self._get_rainfall_sum(self.current_day - 7, self.current_day) / 50.0
        rain_14d = self._get_rainfall_sum(self.current_day - 14, self.current_day) / 100.0
        rain_30d = self._get_rainfall_sum(self.current_day - 30, self.current_day) / 200.0
        
        soil_moisture = self.crop.soil_water / self.crop.soil_water_capacity
        
        # 7-day forecast
        forecast_sum = self._get_rainfall_sum(self.current_day, self.current_day + 7)
        forecast = forecast_sum / 70.0  # Normalized (expecting ~10mm/day max)
        
        is_planted = float(self.crop.is_planted)
        days_since_planting = self.crop.days_since_planting / 130.0
        
        stage_map = {'not_planted': 0.0, 'emergence': 0.2, 'vegetative': 0.4,
                    'flowering': 0.6, 'grain_filling': 0.8, 'maturity': 1.0, 'mature': 1.0}
        growth_stage = stage_map.get(self.crop.get_current_stage(), 0.0)
        season_progress = self.current_day / self.season_duration
        
        return np.array([doy_sin, doy_cos, self.current_day / self.season_duration,
                        rain_7d, rain_14d, rain_30d, soil_moisture, forecast,
                        is_planted, days_since_planting, growth_stage, season_progress,
                        (self.rainfall_scale - 1.0)],  # Normalized: -0.5 (dry) to +0.5 (wet)
                    dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Generate info dictionary for current environment state.
        
        Returns
        -------
        dict
            Contains: day, is_planted, days_since_planting, soil_water,
            cumulative_stress, current_yield.
        """
        return {'day': self.current_day, 'is_planted': self.crop.is_planted,
                'days_since_planting': self.crop.days_since_planting,
                'soil_water': self.crop.soil_water,
                'cumulative_stress': self.crop.cumulative_stress,
                'current_yield': self.crop.compute_yield() if self.crop.is_planted else 0.0}
    
    def reset(self, *, seed: Optional[int] = None,
            options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state with optional domain randomization.
        
        When randomize_start=True (default), each episode samples:
        - Random time slice from rainfall_data (different "weather year")
        - Random initial soil moisture (20-90 mm)
        - Random rainfall scaling (0.5-1.5) for dry/wet year simulation
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Additional reset options (not currently used).
            
        Returns
        -------
        observation : np.ndarray
            Initial 13-dimensional observation.
        info : dict
            Initial environment state information.
        """
        super().reset(seed=seed)
        self.current_day = 0
        self.episode_reward = 0
        self.history = []
        
        # Domain randomization for training robustness
        if self.randomize_start and self.np_random is not None:
            # Random offset into rainfall data (different "year" each episode)
            max_offset = len(self.rainfall_data) - self.season_duration - 30
            if max_offset > 0:
                self.rainfall_offset = self.np_random.integers(0, max_offset)
            else:
                self.rainfall_offset = 0
            
            # Random initial soil moisture (20-90 mm) - wider range
            self.initial_soil_water = self.np_random.uniform(20.0, 90.0)
            
            # Random rainfall scaling (dry years vs wet years)
            # 0.5 = severe drought, 1.0 = normal, 1.5 = very wet
            self.rainfall_scale = self.np_random.uniform(0.5, 1.5)
        else:
            self.rainfall_offset = 0
            self.initial_soil_water = 50.0
            self.rainfall_scale = 1.0
        
        # Initialize crop model without a variety (not planted yet)
        self.crop = CropGrowthModel(
            self.varieties['medium'],  # Default, will be replaced on planting
            initial_soil_water=self.initial_soil_water
        )
        
        return self._get_obs(), self._get_info()
    
    def _compute_planting_conditions(self) -> Dict[str, float]:
        """
        Evaluate current agrometeorological conditions for planting.
        
        Computes a planting favorability score based on:
        - Current soil moisture (optimal: 40-80% capacity)
        - 7-day rainfall forecast (optimal: 20-50mm)
        - 14-day sustained moisture outlook
        
        Returns
        -------
        dict
            Contains: score (float, favorability index),
            rain_7d, forecast_7d, forecast_14d, soil_frac.
        """
        # Recent rainfall (soil preparation)
        rain_7d = self._get_rainfall_sum(self.current_day - 7, self.current_day)
        
        # Upcoming rainfall forecast (critical for germination)
        forecast_7d = self._get_rainfall_sum(self.current_day, self.current_day + 7)
        forecast_14d = self._get_rainfall_sum(self.current_day, self.current_day + 14)
        
        # Soil moisture
        soil_frac = self.crop.soil_water / self.crop.soil_water_capacity
        
        # Planting score: higher is better
        # Good conditions: recent rain (wet soil), good forecast, not too wet
        score = 0.0
        
        # Soil moisture in sweet spot (40-80% capacity)
        if 0.4 <= soil_frac <= 0.8:
            score += 0.3
        elif soil_frac < 0.3:  # Too dry
            score -= 0.3
        elif soil_frac > 0.9:  # Too wet (waterlogging risk)
            score -= 0.2
        
        # Good forecast for germination (20-50mm in next 7 days)
        if 20 <= forecast_7d <= 50:
            score += 0.4
        elif forecast_7d < 10:  # Dry spell ahead
            score -= 0.3
        elif forecast_7d > 80:  # Flood risk
            score -= 0.2
        
        # Sustained moisture in next 2 weeks
        if forecast_14d >= 40:
            score += 0.2
        
        return {
            'score': score,
            'rain_7d': rain_7d,
            'forecast_7d': forecast_7d,
            'forecast_14d': forecast_14d,
            'soil_frac': soil_frac
        }
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Processes the agent's action (wait or plant a variety), advances
        the daily simulation, and computes rewards.
        
        Parameters
        ----------
        action : int
            0 = wait, 1 = plant short, 2 = plant medium, 3 = plant long.
            
        Returns
        -------
        observation : np.ndarray
            New 13-dimensional observation.
        reward : float
            Reward for this transition.
        terminated : bool
            True if episode ended (harvest, timeout, or failure).
        truncated : bool
            Always False (no truncation in this env).
        info : dict
            Environment state information.
            
        Notes
        -----
        Reward structure:
        - Wait: minimal penalty, increases near max_planting_window
        - Plant: condition-based + variety-weather matching bonuses
        - Harvest: (yield/potential) × 5.0 + efficiency bonus + yield bonus
        - Failure: -2.0 penalty
        """
        reward = 0.0
        terminated = False
        
        if not self.crop.is_planted:
            if action == 0:  # Wait
                # MINIMAL waiting penalty - allow timing exploration
                # Only penalize if season is running out
                days_left = self.max_planting_window - self.current_day
                if days_left < 10:
                    reward -= 0.01 * (10 - days_left)  # Increasing urgency
                # No penalty otherwise - let agent explore waiting
                    
            elif action in [1, 2, 3]:  # Plant a variety
                variety_key = {1: 'short', 2: 'medium', 3: 'long'}[action]
                conditions = self._compute_planting_conditions()
                
                # Preserve soil water when creating new crop model
                current_soil = self.crop.soil_water
                self.crop = CropGrowthModel(
                    self.varieties[variety_key],
                    initial_soil_water=current_soil
                )
                self.crop.plant()
                
                # Weather-aware planting reward
                reward += conditions['score'] * 0.5
                
                # =============================================================
                # VARIETY-WEATHER MATCHING REWARD TABLE
                # =============================================================
                # Design: each variety must DECISIVELY win in its target zone.
                # Normal covers 50% of episodes (scale 0.75-1.25), so dry/wet
                # bonuses must be large enough to overcome the frequency disadvantage.
                # Medium's harvest efficiency advantage requires smaller normal bonus.
                #
                # | Condition         | Short  | Medium | Long   | Gap (best vs 2nd)  |
                # |-------------------|--------|--------|--------|---------------------|
                # | Dry (<0.75)       | +2.0   | -1.0   | -3.0   | 3.0 (S vs M)       |
                # | Normal (0.75-1.25)| -0.5   | +1.5   | -1.0   | 2.0 (M vs S)       |
                # | Wet (>1.25)       | -2.0   | -1.5   | +3.0   | 4.5 (L vs M)       |
                #
                # Expected learned policy:
                # - Dry years → Short variety (risk mitigation)
                # - Normal years → Medium variety (balanced)
                # - Wet years → Long variety (maximize yield)
                # =============================================================
                if self.rainfall_scale < 0.75:  # Dry year
                    if variety_key == 'short':
                        reward += 2.0  # Decisive bonus for drought-tolerant choice
                    elif variety_key == 'medium':
                        reward -= 1.0  # Suboptimal — Short is clearly better
                    elif variety_key == 'long':
                        reward -= 3.0  # Heavy penalty — Long fails in drought
                        
                elif self.rainfall_scale > 1.25:  # Wet year
                    if variety_key == 'long':
                        reward += 3.0  # Decisive bonus for high-yield in wet
                    elif variety_key == 'medium':
                        reward -= 1.5  # Missed opportunity — Long is far better
                    elif variety_key == 'short':
                        reward -= 2.0  # Significant missed opportunity
                        
                else:  # NORMAL CONDITIONS (0.75 - 1.25) - Medium's sweet spot
                    if variety_key == 'medium':
                        reward += 1.5  # Clear bonus but not extreme
                    elif variety_key == 'short':
                        reward -= 0.5  # Safe but underperforming
                    elif variety_key == 'long':
                        reward -= 1.0  # Risky — Long often fails in normal
                
                # Bonus for matching variety to remaining season
                days_remaining = self.season_duration - self.current_day
                cycle_fit = days_remaining - self.crop.variety.cycle_length
                
                if cycle_fit >= 10:
                    reward += 0.2
                elif cycle_fit >= 0:
                    reward += 0.1
                else:
                    reward -= 0.3 * (abs(cycle_fit) / 30)
        
        # Advance simulation
        daily_rain = self._get_rainfall(self.current_day)
        self.crop.step(daily_rain)
        self.current_day += 1
        
        # =================================================================
        # TERMINAL CONDITIONS AND HARVEST REWARD
        # =================================================================
        if self.crop.is_mature:
            final_yield = self.crop.compute_yield()
            
            # EFFICIENCY-BASED REWARD (normalizes across varieties)
            # Short (4.5 t/ha potential) and Long (7.0 t/ha) can achieve
            # similar rewards if both reach high efficiency (~80%+)
            yield_ratio = final_yield / self.crop.variety.potential_yield
            
            # Base reward: efficiency × 5.0
            # 100% efficiency = 5.0 reward, 50% = 2.5 reward
            efficiency_reward = yield_ratio * 5.0
            
            # Bonus for exceeding 80% efficiency (reward excellence)
            if yield_ratio > 0.8:
                efficiency_reward += (yield_ratio - 0.8) * 3.0
            
            # Small raw yield bonus (0.1 per t/ha)
            # Provides slight preference for high-yield varieties in good conditions
            raw_yield_bonus = final_yield * 0.1
            
            reward += efficiency_reward + raw_yield_bonus
            terminated = True
            
        elif self.current_day >= self.max_planting_window and not self.crop.is_planted:
            # Failed to plant in time
            reward -= 2.0
            terminated = True
            
        elif self.current_day >= self.season_duration:
            if self.crop.is_planted:
                # Partial yield for incomplete harvest - significant penalty
                final_yield = self.crop.compute_yield()
                maturity_frac = self.crop.days_since_planting / self.crop.variety.cycle_length
                # Harsh penalty for not completing the cycle
                reward += final_yield * maturity_frac * 0.3 - 1.0
            else:
                reward -= 2.0
            terminated = True
        
        self.episode_reward += reward
        return self._get_obs(), reward, terminated, False, self._get_info()
    
    def render(self):
        """
        Render the environment using pygame.
        
        Displays:
        - Farm field with crop growth visualization
        - Soil moisture gauge
        - Rainfall bar chart (past 7 days + forecast)
        - Season timeline with planting window
        - Weather condition indicator (dry/normal/wet)
        - Crop variety and growth stage info
        
        Returns
        -------
        np.ndarray or None
            RGB array if render_mode='rgb_array', else None.
        """
        if self.render_mode is None:
            return None
            
        try:
            import pygame
        except ImportError:
            raise ImportError(
                "pygame is required for rendering. Install with: pip install pygame"
            )
        
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption("🌾 Crop Calendar Environment")
            else:  # rgb_array
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 48)
        
        # Colors
        SKY_BLUE = (135, 206, 235)
        SOIL_BROWN = (139, 90, 43)
        SOIL_DRY = (194, 154, 108)
        SOIL_WET = (101, 67, 33)
        PLANT_GREEN = (34, 139, 34)
        PLANT_YELLOW = (218, 165, 32)
        PLANT_BROWN = (139, 119, 101)
        WATER_BLUE = (30, 144, 255)
        RAIN_BLUE = (70, 130, 180)
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GRAY = (128, 128, 128)
        RED = (220, 20, 60)
        ORANGE = (255, 140, 0)
        GREEN = (50, 205, 50)
        
        # Clear screen with sky gradient
        self.screen.fill(SKY_BLUE)
        
        # Draw weather condition indicator (top)
        weather_rect = pygame.Rect(10, 10, 200, 40)
        if self.rainfall_scale < 0.75:
            weather_color = ORANGE
            weather_text = "☀️ DRY YEAR"
        elif self.rainfall_scale > 1.25:
            weather_color = WATER_BLUE
            weather_text = "🌧️ WET YEAR"
        else:
            weather_color = GREEN
            weather_text = "🌤️ NORMAL YEAR"
        pygame.draw.rect(self.screen, weather_color, weather_rect, border_radius=10)
        pygame.draw.rect(self.screen, BLACK, weather_rect, 2, border_radius=10)
        text = self.font_small.render(weather_text, True, WHITE)
        self.screen.blit(text, (weather_rect.x + 15, weather_rect.y + 10))
        
        # Draw rainfall scale value
        scale_text = self.font_small.render(f"Scale: {self.rainfall_scale:.2f}", True, BLACK)
        self.screen.blit(scale_text, (220, 20))
        
        # Draw day counter (top right)
        day_text = self.font.render(f"Day {self.current_day} / {self.season_duration}", True, BLACK)
        self.screen.blit(day_text, (self.screen_width - 200, 15))
        
        # =====================================================================
        # FIELD AREA (center)
        # =====================================================================
        field_rect = pygame.Rect(50, 80, 400, 250)
        
        # Soil color based on moisture
        soil_frac = self.crop.soil_water / self.crop.soil_water_capacity if self.crop else 0.5
        soil_color = (
            int(SOIL_DRY[0] * (1 - soil_frac) + SOIL_WET[0] * soil_frac),
            int(SOIL_DRY[1] * (1 - soil_frac) + SOIL_WET[1] * soil_frac),
            int(SOIL_DRY[2] * (1 - soil_frac) + SOIL_WET[2] * soil_frac)
        )
        pygame.draw.rect(self.screen, soil_color, field_rect)
        pygame.draw.rect(self.screen, BLACK, field_rect, 3)
        
        # Draw furrows
        for i in range(5):
            y = field_rect.y + 40 + i * 45
            pygame.draw.line(self.screen, SOIL_BROWN, 
                        (field_rect.x + 20, y), 
                        (field_rect.x + field_rect.width - 20, y), 3)
        
        # Draw crops if planted
        if self.crop and self.crop.is_planted:
            stage = self.crop.get_current_stage()
            progress = self.crop.days_since_planting / self.crop.variety.cycle_length
            
            # Crop appearance by stage
            if stage == 'emergence':
                plant_height = 15
                plant_color = PLANT_GREEN
            elif stage == 'vegetative':
                plant_height = int(15 + 60 * (progress - 0.1) / 0.3)
                plant_color = PLANT_GREEN
            elif stage == 'flowering':
                plant_height = 75
                plant_color = PLANT_YELLOW
            elif stage == 'grain_filling':
                plant_height = 75
                plant_color = (180, 160, 80)  # Yellowish
            elif stage in ['maturity', 'mature']:
                plant_height = 70
                plant_color = PLANT_BROWN
            else:
                plant_height = 0
                plant_color = PLANT_GREEN
            
            # Draw plants in furrows
            for i in range(5):
                y = field_rect.y + 40 + i * 45
                for j in range(8):
                    x = field_rect.x + 40 + j * 45
                    if plant_height > 0:
                        # Stem
                        pygame.draw.line(self.screen, plant_color,
                                    (x, y), (x, y - plant_height), 3)
                        # Leaves (if vegetative or beyond)
                        if plant_height > 20:
                            pygame.draw.line(self.screen, plant_color,
                                        (x, y - plant_height//2),
                                        (x - 10, y - plant_height//2 + 10), 2)
                            pygame.draw.line(self.screen, plant_color,
                                        (x, y - plant_height//2),
                                        (x + 10, y - plant_height//2 + 10), 2)
                        # Flower/grain head
                        if stage in ['flowering', 'grain_filling', 'maturity', 'mature']:
                            pygame.draw.circle(self.screen, PLANT_YELLOW,
                                            (x, y - plant_height - 5), 5)
        
        # =====================================================================
        # SOIL MOISTURE GAUGE (right of field)
        # =====================================================================
        gauge_x = 480
        gauge_y = 80
        gauge_width = 40
        gauge_height = 250
        
        pygame.draw.rect(self.screen, WHITE, (gauge_x, gauge_y, gauge_width, gauge_height))
        pygame.draw.rect(self.screen, BLACK, (gauge_x, gauge_y, gauge_width, gauge_height), 2)
        
        # Fill level
        fill_height = int(gauge_height * soil_frac)
        fill_rect = pygame.Rect(gauge_x + 2, gauge_y + gauge_height - fill_height, gauge_width - 4, fill_height)
        pygame.draw.rect(self.screen, WATER_BLUE, fill_rect)
        
        # Labels
        text = self.font_small.render("SOIL", True, BLACK)
        self.screen.blit(text, (gauge_x - 5, gauge_y - 25))
        text = self.font_small.render(f"{soil_frac*100:.0f}%", True, BLACK)
        self.screen.blit(text, (gauge_x + 3, gauge_y + gauge_height + 5))
        
        # =====================================================================
        # RAINFALL CHART (bottom left)
        # =====================================================================
        chart_x = 50
        chart_y = 370
        chart_width = 300
        chart_height = 100
        
        pygame.draw.rect(self.screen, WHITE, (chart_x, chart_y, chart_width, chart_height))
        pygame.draw.rect(self.screen, BLACK, (chart_x, chart_y, chart_width, chart_height), 2)
        
        # Past 7 days + forecast 7 days
        bar_width = chart_width // 14
        max_rain = 30  # mm scale
        
        for i in range(-7, 7):
            day_rain = self._get_rainfall(self.current_day + i)
            bar_height = min(int((day_rain / max_rain) * chart_height), chart_height - 5)
            bar_x = chart_x + (i + 7) * bar_width
            bar_y = chart_y + chart_height - bar_height
            
            # Past (darker) vs forecast (lighter)
            if i < 0:
                color = RAIN_BLUE
            else:
                color = (150, 190, 220)
            
            if bar_height > 0:
                pygame.draw.rect(self.screen, color, (bar_x + 2, bar_y, bar_width - 4, bar_height))
        
        # Dividing line (today)
        today_x = chart_x + 7 * bar_width
        pygame.draw.line(self.screen, RED, (today_x, chart_y), (today_x, chart_y + chart_height), 2)
        
        # Labels
        text = self.font_small.render("RAINFALL (mm)", True, BLACK)
        self.screen.blit(text, (chart_x, chart_y - 25))
        text = self.font_small.render("← Past | Future →", True, GRAY)
        self.screen.blit(text, (chart_x + 70, chart_y + chart_height + 5))
        
        # =====================================================================
        # SEASON TIMELINE (bottom center)
        # =====================================================================
        timeline_x = 380
        timeline_y = 400
        timeline_width = 380
        timeline_height = 30
        
        pygame.draw.rect(self.screen, WHITE, (timeline_x, timeline_y, timeline_width, timeline_height))
        pygame.draw.rect(self.screen, BLACK, (timeline_x, timeline_y, timeline_width, timeline_height), 2)
        
        # Planting window (green zone)
        plant_window_width = int(timeline_width * self.max_planting_window / self.season_duration)
        pygame.draw.rect(self.screen, (200, 255, 200), 
                        (timeline_x, timeline_y, plant_window_width, timeline_height))
        
        # Current day marker
        day_pos = int(timeline_width * self.current_day / self.season_duration)
        pygame.draw.rect(self.screen, RED, 
                        (timeline_x + day_pos - 2, timeline_y - 5, 4, timeline_height + 10))
        
        # Harvest date marker (if planted)
        if self.crop and self.crop.is_planted:
            harvest_day = self.current_day - self.crop.days_since_planting + self.crop.variety.cycle_length
            if harvest_day <= self.season_duration:
                harvest_pos = int(timeline_width * harvest_day / self.season_duration)
                pygame.draw.rect(self.screen, PLANT_BROWN, (timeline_x + harvest_pos - 2, timeline_y - 5, 4, timeline_height + 10))
        
        text = self.font_small.render("SEASON TIMELINE", True, BLACK)
        self.screen.blit(text, (timeline_x, timeline_y - 25))
        text = self.font_small.render("Planting Window", True, GREEN)
        self.screen.blit(text, (timeline_x, timeline_y + 35))
        
        # =====================================================================
        # CROP INFO PANEL (right side)
        # =====================================================================
        info_x = 550
        info_y = 80
        
        pygame.draw.rect(self.screen, (245, 245, 245), (info_x, info_y, 230, 180))
        pygame.draw.rect(self.screen, BLACK, (info_x, info_y, 230, 180), 2)
        
        text = self.font.render("CROP STATUS", True, BLACK)
        self.screen.blit(text, (info_x + 50, info_y + 10))
        
        if self.crop and self.crop.is_planted:
            variety_names = {'short': 'Short (90d)', 'medium': 'Medium (110d)', 'long': 'Long (130d)'}
            var_name = variety_names.get(self.crop.variety.name.lower().split()[0], self.crop.variety.name)
            # Find variety key from name
            for vkey, vdata in self.varieties.items():
                if vdata.name == self.crop.variety.name:
                    var_name = f"{vkey.capitalize()} ({vdata.cycle_length}d)"
                    break
            
            lines = [
                f"Variety: {var_name}",
                f"Stage: {self.crop.get_current_stage().replace('_', ' ').title()}",
                f"Day: {self.crop.days_since_planting} / {self.crop.variety.cycle_length}",
                f"Stress: {self.crop.cumulative_stress / max(1, self.crop.days_since_planting):.2f}",
                f"Est. Yield: {self.crop.compute_yield():.2f} t/ha"
            ]
        else:
            lines = [
                "Status: Not Planted",
                "",
                "Actions:",
                "  0: Wait",
                "  1: Plant Short",
                "  2: Plant Medium", 
                "  3: Plant Long"
            ]
        
        for i, line in enumerate(lines):
            text = self.font_small.render(line, True, BLACK)
            self.screen.blit(text, (info_x + 10, info_y + 45 + i * 22))
        
        # =====================================================================
        # REWARD INFO (bottom right)
        # =====================================================================
        text = self.font_small.render(f"Episode Reward: {self.episode_reward:.2f}", True, BLACK)
        self.screen.blit(text, (550, 550))
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Clean up pygame resources."""
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
