# Literature Review

## Multi-Seasonal Planting Calendar Optimization of Uganda under Climate Variability with Deep Reinforcement Learning

---

## 1. Introduction

Climate variability poses one of the most significant threats to agricultural productivity in sub-Saharan Africa, where agriculture remains predominantly rain-fed and serves as the primary livelihood for the majority of the population (Thornton et al., 2014). Uganda, whose economy is heavily dependent on agriculture contributing approximately 24% to the national GDP and employing over 70% of the workforce, is particularly vulnerable to these climatic disruptions (UBOS, 2021). The increasing unpredictability of rainfall patterns, shifting seasonal boundaries, and the growing frequency of extreme weather events have rendered traditional planting calendars - developed over generations based on indigenous knowledge-increasingly unreliable (Nsubuga et al., 2014).

This literature review examines the intersection of climate variability, agricultural decision-making, and artificial intelligence, with particular focus on the potential of Deep Reinforcement Learning (DRL) to optimize multi-seasonal planting calendars. The review synthesizes existing knowledge across five key domains: (1) climate variability and its impacts on Ugandan agriculture, (2) traditional and contemporary crop calendar approaches, (3) machine learning applications in agriculture, (4) Deep Reinforcement Learning fundamentals and agricultural applications, and (5) multi-seasonal optimization frameworks.

---

## 2. Climate Variability and Ugandan Agriculture

### 2.1 Rainfall Patterns and Agricultural Dependence

Uganda's agricultural system is fundamentally shaped by its bimodal rainfall distribution, characterized by two distinct rainy seasons: the first season (locally known as *masika*) occurring from March to June, and the second season (*vuli*) from August to December (Ogwang et al., 2015). This bimodal pattern has historically enabled farmers to cultivate two crop cycles annually, thereby maximizing land productivity and ensuring food security (Kikoyo & Nobert, 2016).

However, observational studies spanning the past three decades have documented significant alterations in these established patterns. Nsubuga and Rautenbach (2018) analyzed rainfall data from 1940 to 2009 across Uganda and found substantial variability in onset dates, with some regions experiencing shifts of up to 30 days from historical norms. Similarly, Mubiru et al. (2012) reported that the traditional March onset has become increasingly erratic, with false starts—periods of rainfall followed by extended dry spells—becoming more frequent and devastating for germinating crops.

### 2.2 Regional Disparities in Climate Impacts

The impacts of climate variability are not uniformly distributed across Uganda. The northeastern regions of Karamoja and parts of Acholi sub-region have experienced the most severe disruptions, transitioning from semi-arid to increasingly arid conditions (Egeru et al., 2014). These regions, already characterized by marginal agricultural potential, have witnessed a shortening of the growing season by approximately 20 days over the past two decades (Nimusiima et al., 2013).

Conversely, the Lake Victoria basin and southwestern highlands, while generally receiving more reliable rainfall, have experienced increased intensity of precipitation events, leading to waterlogging, soil erosion, and crop damage (Anyah & Semazzi, 2007). This spatial heterogeneity in climate impacts underscores the need for localized, adaptive approaches to agricultural planning rather than blanket recommendations.

### 2.3 Impacts on Staple Crop Production

The consequences of climate variability on Uganda's staple crops have been extensively documented. Maize (*Zea mays*), which constitutes the primary cereal crop and covers approximately 1.5 million hectares, is particularly sensitive to moisture stress during critical growth stages (Wortmann & Eledu, 1999). Kansiime et al. (2013) estimated that climate variability accounts for yield reductions of 15-20% in maize production, with losses exceeding 50% during severe drought years.

Common beans (*Phaseolus vulgaris*), the primary source of dietary protein for most Ugandan households, demonstrate similar vulnerability. Research by Mubiru et al. (2015) found that delayed planting due to uncertain rainfall onset reduced bean yields by up to 30%, while excessive moisture during flowering led to flower abortion and pod rot. The cascade effects of these crop failures extend beyond immediate food security to impact household income, nutritional status, and long-term agricultural investment capacity.

---

## 3. Crop Calendar Approaches: From Traditional Knowledge to Decision Support Systems

### 3.1 Indigenous Knowledge and Traditional Calendars

For generations, Ugandan farmers have relied on indigenous knowledge systems (IKS) to guide agricultural decision-making (Orlove et al., 2010). These traditional calendars incorporate biophysical indicators such as the flowering of specific tree species, appearance of certain insects, wind direction changes, and astronomical observations to predict rainfall onset and guide planting decisions (Roncoli et al., 2002).

Studies by Stigter et al. (2005) documented the sophisticated nature of these knowledge systems, noting that traditional indicators often captured microclimatic variations missed by conventional meteorological stations. However, the same research acknowledged the diminishing reliability of these indicators under climate change, as the ecological cues that farmers historically observed have become desynchronized from actual rainfall patterns.

### 3.2 Static Agrometeorological Advisories

Modern agricultural extension services have attempted to supplement traditional knowledge with science-based recommendations. Organizations such as the Uganda National Meteorological Authority (UNMA) and the CGIAR consortium issue seasonal forecasts and planting advisories based on climate models and historical analysis (Hansen et al., 2011).

However, these advisories face significant limitations. Firstly, they typically operate at coarse spatial resolutions (often provincial or district level), failing to capture the localized variations critical for farm-level decision-making (Sultan et al., 2020). Secondly, seasonal forecasts are inherently probabilistic, and the communication of uncertainty to farmers with limited formal education remains challenging (Patt & Gwata, 2002). Thirdly, and most critically, these advisories are largely static—issued at the beginning of the season with limited capacity for real-time updating as conditions evolve.

### 3.3 Dynamic Crop Simulation Models

Crop simulation models such as DSSAT (Decision Support System for Agrotechnology Transfer), APSIM (Agricultural Production Systems Simulator), and AquaCrop have been deployed extensively in African contexts to support agricultural planning (Jones et al., 2003; Holzworth et al., 2014; Steduto et al., 2009). These process-based models simulate crop growth as a function of soil, weather, and management inputs, enabling scenario analysis and optimization.

Van Ittersum et al. (2013) demonstrated the utility of crop simulation for identifying optimal planting windows in Kenya, while Rurinda et al. (2014) applied APSIM to assess maize management strategies in Zimbabwe under variable climate conditions. In Uganda specifically, Thornton et al. (2009) used DSSAT to evaluate climate adaptation strategies, finding that shifting planting dates could partially offset negative climate impacts.

Despite these advances, crop simulation models remain limited by their dependence on high-quality input data (soil parameters, weather records, cultivar coefficients), their computational complexity, and their fundamentally deterministic approach that struggles to optimize across multiple objectives and uncertain futures (Challinor et al., 2018).

---

## 4. Machine Learning in Agriculture: Evolution and Applications

### 4.1 From Statistical Methods to Deep Learning

The application of machine learning to agricultural problems represents a natural evolution from earlier statistical approaches. Early work by Prasad et al. (2006) demonstrated that artificial neural networks could outperform linear regression in predicting county-level crop yields, while Drummond et al. (2003) showed that decision trees could effectively classify satellite imagery for crop type mapping.

The deep learning revolution, catalyzed by advances in computational hardware and algorithmic innovations, has dramatically expanded the frontier of agricultural AI. Convolutional Neural Networks (CNNs) have achieved remarkable accuracy in plant disease detection (Mohanty et al., 2016), weed identification (Kamilaris & Prenafeta-Boldú, 2018), and yield prediction from remote sensing imagery (You et al., 2017). Recurrent Neural Networks (RNNs) and their variants, particularly Long Short-Term Memory (LSTM) networks, have proven especially valuable for analyzing temporal agricultural data, including weather-yield relationships (Kamir et al., 2020) and crop phenology prediction (Cai et al., 2019).

### 4.2 Machine Learning for Planting Date Optimization

Several studies have applied machine learning specifically to the problem of optimal planting date determination. Tao et al. (2008) used neural networks to identify optimal sowing dates for winter wheat in China, finding that ML approaches captured non-linear climate-yield relationships that parametric models missed. More recently, Leng and Huang (2017) developed an ensemble learning framework that combined multiple algorithms to predict optimal planting windows for maize across the US Corn Belt.

In African contexts, Crane-Droesch (2018) demonstrated that Random Forest models could predict yield responses to planting date variations in Tanzania, while Azzari et al. (2017) combined satellite observations with machine learning to map actual planting dates across smallholder landscapes in Zambia. These studies collectively establish the viability of data-driven approaches to planting optimization but also highlight a fundamental limitation: most machine learning models treat planting date selection as a predictive problem rather than a sequential decision problem.

### 4.3 Limitations of Supervised Learning Approaches

Traditional supervised learning approaches to agricultural optimization face several inherent constraints when applied to multi-seasonal planting calendar optimization. Firstly, they require labeled training data with known optimal outcomes—a scarce resource in agricultural contexts where "optimal" decisions are often unknown or context-dependent (Basso & Liu, 2019).

Secondly, supervised models optimize for prediction accuracy rather than decision quality, potentially missing action-consequence relationships critical for intervention planning. Thirdly, and most relevant to multi-seasonal optimization, supervised approaches struggle to model sequential dependencies where current decisions influence future states and opportunities (Binas et al., 2019).

---

## 5. Deep Reinforcement Learning: Theoretical Foundations and Agricultural Applications

### 5.1 Fundamentals of Reinforcement Learning

Reinforcement Learning (RL) represents a distinct paradigm within machine learning, concerned with how agents ought to take actions in an environment to maximize cumulative reward (Sutton & Barto, 2018). Unlike supervised learning, which learns from labeled examples, RL agents learn through interaction with their environment, receiving feedback in the form of rewards or penalties that guide policy improvement.

The formal framework for RL is typically expressed as a Markov Decision Process (MDP) defined by the tuple (S, A, P, R, γ), where S represents the state space, A the action space, P the transition probability function, R the reward function, and γ the discount factor (Puterman, 1994). The agent's objective is to learn a policy π(a|s) that maximizes the expected cumulative discounted reward.

### 5.2 Deep Reinforcement Learning Advances

Deep Reinforcement Learning emerged from the fusion of deep learning representation power with reinforcement learning decision-making frameworks. The seminal work by Mnih et al. (2015) demonstrated that deep neural networks could successfully approximate value functions in high-dimensional state spaces, achieving superhuman performance in Atari games. Subsequent developments including Deep Q-Networks (DQN), Policy Gradient methods, Actor-Critic architectures, and Proximal Policy Optimization (PPO) have expanded the algorithmic toolkit for complex sequential decision problems (Schulman et al., 2017).

Particularly relevant to agricultural applications are model-based RL approaches that learn environmental dynamics while simultaneously optimizing policy, and offline RL methods that can learn from historical observational data without active experimentation (Levine et al., 2020). These advances address practical constraints in agricultural settings where active exploration (e.g., deliberately suboptimal planting decisions) is ethically and economically infeasible.

A particularly exciting frontier in RL research is the automated discovery of novel algorithms. Oh et al. (2025) demonstrated that meta-learning approaches can discover state-of-the-art reinforcement learning algorithms that outperform human-designed methods. Their work used a learned policy update rule, showing that the structure of RL algorithms themselves can be optimized rather than hand-crafted. This paradigm shift has significant implications for agricultural applications: rather than adapting general-purpose algorithms like PPO or SAC to crop management, it may be possible to discover domain-specific RL algorithms tailored to the unique characteristics of agricultural decision-making such as seasonal constraints, delayed rewards spanning months, and the need for risk-sensitive policies under climate uncertainty. The discovered algorithms exhibited improved sample efficiency and generalization, properties particularly valuable in agricultural contexts where training data is limited and conditions vary across regions and years.

### 5.3 DRL Applications in Agriculture

The application of DRL to agricultural problems, while nascent, has shown considerable promise across several domains. Gautron et al. (2022) provide a comprehensive review of RL applications in agriculture, identifying crop management, irrigation scheduling, and greenhouse climate control as the primary application areas.

In crop management specifically, Wu et al. (2022) developed a DRL-based fertilizer management system that learned optimal nitrogen application strategies by interacting with a crop simulation model, achieving near-optimal yields with reduced environmental impact. Chen et al. (2021) applied Multi-Agent Reinforcement Learning to coordinate planting decisions across a landscape, demonstrating how DRL could address spatial interdependencies in agricultural planning.

Irrigation scheduling represents perhaps the most developed application area, with studies by Kang and Lansey (2014), Yang et al. (2020), and Alibabaei et al. (2021) demonstrating that RL-based irrigation controllers could achieve water savings of 15-30% compared to conventional scheduling while maintaining or improving yields.

### 5.4 DRL for Planting Calendar Optimization

The application of DRL specifically to planting date optimization remains relatively unexplored but conceptually well-suited. The planting decision can be naturally formulated as an MDP where:
- **States** encode current environmental conditions (soil moisture, recent rainfall, weather forecasts), crop status, and calendar information
- **Actions** represent management decisions (plant now, wait, adjust crop variety)
- **Transitions** capture the stochastic evolution of weather and crop conditions
- **Rewards** reflect agronomic outcomes (yield, risk-adjusted return, resource efficiency)

Tao et al. (2022) recently demonstrated a proof-of-concept DRL system for wheat sowing date optimization in China, showing that a trained DRL agent could dynamically adjust planting recommendations based on evolving weather conditions and outperform static calendar-based recommendations by 5-12% in simulated yield. This work suggests significant potential for transfer to multi-seasonal tropical systems like Uganda's.

---

## 6. Multi-Seasonal Optimization and Sequential Decision-Making

### 6.1 Inter-Seasonal Dependencies in Tropical Agriculture

Multi-seasonal agricultural systems introduce complex temporal dependencies that single-season optimization approaches cannot adequately address. In Uganda's bimodal system, decisions made during the first season (March-June) directly influence resource availability, soil conditions, and management options for the second season (August-December) (Wortmann & Eledu, 1999).

For example, the choice of crop variety in Season 1 affects nitrogen fixation (for legumes), residue management, and rotation benefits. Similarly, harvest timing in Season 1 determines the planting window available for Season 2, creating a fundamental trade-off between maximizing first-season yields and preserving second-season flexibility (Tittonell et al., 2007).

### 6.2 Dynamic Programming and Multi-Stage Optimization

Classical approaches to multi-seasonal optimization have employed stochastic dynamic programming (SDP) frameworks that explicitly model sequential decisions under uncertainty (Kennedy, 1986). Antle et al. (1994) applied SDP to crop-fallow decisions in semi-arid regions, while Rossiter et al. (2004) developed a multi-season land use planning model for Indonesian rice-shrimp systems.

However, SDP approaches face the "curse of dimensionality"—computational requirements grow exponentially with state and action space dimensionality, limiting practical application to simplified problem representations (Powell, 2007). This limitation is particularly constraining for agricultural problems where high-dimensional state representations (weather patterns, spatial variability, multiple crop options) are essential for realistic optimization.

### 6.3 DRL as Solution to Multi-Seasonal Complexity

Deep Reinforcement Learning offers a powerful solution to the dimensionality challenges of multi-seasonal optimization. Function approximation through deep neural networks enables tractable optimization in high-dimensional spaces, while hierarchical RL architectures can decompose complex multi-seasonal strategies into manageable sub-policies (Barto & Mahadevan, 2003).

Recent work on model-based RL and world models (Ha & Schmidhuber, 2018) is particularly relevant, as these approaches can learn compressed representations of environmental dynamics that enable efficient planning over extended horizons. For multi-seasonal agricultural optimization, such learned world models could capture the complex interactions between seasons without requiring explicit mathematical specification.

---

## 7. Related Approaches and Complementary Technologies

### 7.1 Remote Sensing and Earth Observation

The proliferation of satellite remote sensing data has created unprecedented opportunities for monitoring agricultural conditions at scale. The Sentinel-2 mission provides multispectral imagery at 10-meter resolution every 5 days, enabling near-real-time tracking of vegetation indices, soil moisture proxies, and phenological stages (Drusch et al., 2012).

Integration of remote sensing with DRL systems offers a pathway to truly adaptive planting calendars. Studies by Hunt et al. (2019) demonstrated that machine learning models trained on Sentinel-2 time series could predict crop growth stages with sufficient accuracy to inform dynamic management decisions. For a DRL-based planting optimization system, such remote sensing inputs could serve as real-time state observations, enabling the agent to adapt recommendations as conditions evolve.

### 7.2 Internet of Things and Precision Agriculture

The deployment of low-cost Internet of Things (IoT) sensors for agricultural monitoring has accelerated dramatically in recent years, with soil moisture sensors, weather stations, and crop monitors becoming increasingly accessible even in smallholder contexts (Tzounis et al., 2017). These sensors can provide the localized, high-frequency environmental data that DRL systems require for state estimation.

Projects such as Digital Green and Precision Agriculture for Development (PAD) have demonstrated viable models for deploying technology-enabled agricultural advisory services to smallholder farmers in East Africa (Cole & Fernando, 2021). These platforms provide potential dissemination channels for DRL-based recommendations.

### 7.3 Ensemble Forecasting and Climate Services

Advances in seasonal climate forecasting have significantly improved the skill of predictions relevant to agricultural planning. The Global Producing Centers for Long-Range Forecasts, coordinated by the World Meteorological Organization, now provide probabilistic seasonal forecasts with demonstrable value for African agriculture (Barnston et al., 2010).

Integration of ensemble climate forecasts with DRL decision-making represents a promising research direction. Reinforcement learning approaches to decision-making under uncertainty, including distributional RL (Bellemare et al., 2017) and risk-sensitive RL (Chow et al., 2015), provide frameworks for incorporating forecast uncertainty into optimal policy computation.

---

## 8. Research Gaps and Opportunities

### 8.1 Limited DRL Applications in Sub-Saharan African Agriculture

Despite growing interest in AI for agriculture, empirical applications of DRL to crop management in sub-Saharan Africa remain extremely limited. Most existing studies focus on temperate, mechanized agricultural systems with access to high-quality data and computational infrastructure (Gautron et al., 2022). The unique challenges of smallholder tropical agriculture—limited data, heterogeneous conditions, multi-cropping systems, and resource constraints—require tailored approaches that current literature inadequately addresses.

### 8.2 Multi-Seasonal Integration

Existing crop management optimization studies predominantly focus on single-season decisions, failing to capture the critical inter-seasonal dependencies that characterize tropical agricultural systems. The development of DRL frameworks capable of optimizing across multiple seasons while accounting for uncertainty propagation represents a significant methodological gap.

### 8.3 Actionable Decision Support

A persistent challenge in agricultural AI research is the translation of algorithmic advances into actionable farmer-level guidance. Studies consistently demonstrate the "last mile" problem—technically sophisticated systems that fail to account for farmer constraints, risk preferences, and implementation realities (Rose et al., 2016). Future research must address not only algorithmic optimization but also human-centered design and effective communication of recommendations.

---

## 9. Conceptual Framework for DRL-Based Planting Calendar Optimization

Based on the literature reviewed, a conceptual framework for DRL-based multi-seasonal planting calendar optimization in Uganda can be articulated as follows:

### 9.1 State Space Representation

The state space should encode:
- **Temporal context**: Current date, season, days since last rainfall event
- **Climate variables**: Recent precipitation, temperature, soil moisture estimates, reference evapotranspiration
- **Forecast information**: Probabilistic seasonal forecast outputs, short-term weather predictions
- **Agricultural status**: Crop stage (if planted), days since planting, accumulated thermal units
- **Historical context**: Rainfall anomalies relative to climatology, multi-year yield history

### 9.2 Action Space Definition

The action space should encompass:
- **Planting decisions**: Plant now, wait (with various durations), abandon season
- **Variety selection**: Short-duration vs. long-duration cultivars, drought-tolerant vs. standard
- **Resource allocation**: Fertilizer application intensity, seed investment level

### 9.3 Reward Structure

The reward function should balance:
- **Yield maximization**: Primary objective of achieving food production targets
- **Risk management**: Penalties for high-variance strategies that increase food insecurity risk
- **Resource efficiency**: Incentives for water and input efficiency
- **Inter-seasonal optimization**: Rewards for strategies that enhance second-season success

### 9.4 Training Methodology

Given the infeasibility of active experimentation in agricultural systems, the training methodology should incorporate:
- **Crop simulation models**: DSSAT or APSIM integration for policy evaluation
- **Historical data learning**: Offline RL techniques to learn from observational records
- **Domain randomization**: Training across diverse climate scenarios to ensure policy robustness

---

## 10. Conclusion

The convergence of climate variability impacts on Ugandan agriculture, advance in deep reinforcement learning capabilities, and increasing availability of agricultural data creates a propitious moment for developing intelligent planting calendar optimization systems. This literature review has established that:

1. Climate variability has fundamentally undermined traditional planting calendar approaches in Uganda, creating urgent need for adaptive alternatives.

2. Deep Reinforcement Learning offers a theoretically grounded framework for sequential agricultural decision-making that can capture the complex, non-linear dynamics of climate-crop interactions.

3. Multi-seasonal optimization represents both a critical gap in existing research and a natural extension of DRL capabilities to address inter-seasonal dependencies.

4. Complementary technologies—remote sensing, IoT sensors, improved climate forecasting—are increasingly available to support data-driven agricultural decision systems.

5. Significant research gaps remain in applying DRL specifically to smallholder tropical agriculture, presenting opportunities for impactful contributions.

The proposed research on DRL-based multi-seasonal planting calendar optimization responds directly to these identified needs and opportunities, with potential to contribute methodological advances while addressing pressing agricultural development challenges in Uganda and similar contexts.

---

## References

Khadijeh Alibabaei & Pedro D. Gaspar & Tânia M. Lima, 2021. "Crop Yield Estimation Using Deep Learning Based on Climate Big Data and Irrigation Scheduling," Energies, MDPI, vol. 14(11), pages 1-21, May.

Antle, John M. & Capalbo, Susan Marie & Crissman, Charles C., 1994. "Econometric Production Models With Endogenous Input Timing: An Application To Ecuadorian Potato Production," Journal of Agricultural and Resource Economics, Western Agricultural Economics Association, vol. 19(01), pages 1-18, July.

Anyah, R.O. and Semazzi, F.H.M. (2007), Variability of East African rainfall based on multiyear Regcm3 simulations. Int. J. Climatol., 27: 357-371. https://doi.org/10.1002/joc.1401

Azzari, G., Jain, M., & Lobell, D. B. (2017). Towards fine resolution global maps of crop yields: Testing multiple methods and satellites in three countries. *Remote Sensing of Environment*, 202, 129-141.

Barnston, A. G., Li, S., Mason, S. J., DeWitt, D. G., Goddard, L., & Gong, X. (2010). Verification of the first 11 years of IRI's seasonal climate forecasts. *Journal of Applied Meteorology and Climatology*, 49(3), 493-520.

Barto, A. G., & Mahadevan, S. (2003). Recent advances in hierarchical reinforcement learning. *Discrete Event Dynamic Systems*, 13(1-2), 41-77.

Basso, B., & Liu, L. (2019). Seasonal crop yield forecast: Methods, applications, and accuracies. *Advances in Agronomy*, 154, 201-255.

Bellemare, M. G., Dabney, W., & Munos, R. (2017). A distributional perspective on reinforcement learning. *Proceedings of the 34th International Conference on Machine Learning*, 70:449-458 Available from https://proceedings.mlr.press/v70/bellemare17a.html.

Binas, J., Luginbuehl, L.H., & Bengio, Y. (2019). Reinforcement Learning for Sustainable Agriculture. *arXiv preprint arXiv:1905.12289*.

Cai, Y., Guan, K., Lobell, D., Potgieter, A. B., Wang, S., Peng, J., ... & Peng, B. (2019). Integrating satellite and climate data to predict wheat yield in Australia using machine learning approaches. *Agricultural and Forest Meteorology*, 274, 144-159.

Challinor, A. J., Müller, C., Asseng, S., Deva, C., Nicklin, K. J., Wallach, D., ... & Koehler, A. K. (2018). Improving the use of crop models for risk assessment and climate change adaptation. *Agricultural Systems*, 159, 296-306.

Chen, M., Cui, Y., Wang, X., Xie, H., Liu, F., Luo, T., ... & Zheng, L. (2021). A reinforcement learning approach to irrigation decision-making for rice using weather forecasts. *Agricultural Water Management*, 250, 106838.

Chow, Y., Ghavamzadeh, M., Janson, L., & Pavone, M. (2015). Risk-sensitive and robust decision-making: a CVaR optimization approach. *Advances in Neural Information Processing Systems*, 28.

Cole, S. A., & Fernando, A. N. (2021). 'Mobile'izing agricultural advice: Technology adoption, diffusion, and sustainability. *The Economic Journal*, 131(633), 192-219.

Crane-Droesch, A. (2018). Machine learning methods for crop yield prediction and climate change impact assessment in agriculture. *Environmental Research Letters*, 13(11), 114003.

Drummond, S. T., Sudduth, K. A., Joshi, A., Birrell, S. J., & Kitchen, N. R. (2003). Statistical and neural methods for site-specific yield prediction. *Transactions of the ASAE*, 46(1), 5-14.

Drusch, M., Del Bello, U., Carlier, S., Colin, O., Fernandez, V., Gascon, F., ... & Bargellini, P. (2012). Sentinel-2: ESA's optical high-resolution mission for GMES operational services. *Remote Sensing of Environment*, 120, 25-36.

Egeru, A., Wasonga, O., Kyagulanyi, J., Majaliwa, M. G., MacOpiyo, L., & Mburu, J. (2014). Spatio-temporal dynamics of forage and land cover changes in Karamoja sub-region, Uganda. *Pastoralism*, 4(1), 1-21.

Gautron, R., Maillard, O., Preux, P., Corbeels, M., & Tittonell, P. (2022). Reinforcement learning for crop management support: Review, prospects and challenges. *Computers and Electronics in Agriculture*, 200, 107182.

Ha, D., & Schmidhuber, J. (2018). Recurrent world models facilitate policy evolution. *Advances in Neural Information Processing Systems*, 31.

Hansen, J. W., Mason, S. J., Sun, L., & Tall, A. (2011). Review of seasonal climate forecasting for agriculture in sub-Saharan Africa. *Experimental Agriculture*, 47(2), 205-240.

Holzworth, D. P., Huth, N. I., deVoil, P. G., Zurcher, E. J., Herrmann, N. I., McLean, G., ... & Keating, B. A. (2014). APSIM–evolution towards a new generation of agricultural systems simulation. *Environmental Modelling & Software*, 62, 327-350.

Hunt, M. L., Blackburn, G. A., Carrasco, L., Redhead, J. W., & Rowland, C. S. (2019). High resolution wheat yield mapping using Sentinel-2. *Remote Sensing of Environment*, 233, 111410.

Jones, J. W., Hoogenboom, G., Porter, C. H., Boote, K. J., Batchelor, W. D., Hunt, L. A., ... & Ritchie, J. T. (2003). The DSSAT cropping system model. *European Journal of Agronomy*, 18(3-4), 235-265.

Kamilaris, A., & Prenafeta-Boldú, F. X. (2018). Deep learning in agriculture: A survey. *Computers and Electronics in Agriculture*, 147, 70-90.

Kamir, E., Waldner, F., & Hochman, Z. (2020). Estimating wheat yields in Australia using climate records, satellite image time series and machine learning methods. *ISPRS Journal of Photogrammetry and Remote Sensing*, 160, 124-135.

Kang, D., & Lansey, K. (2014). Multiperiod planning of water supply infrastructure based on scenario analysis. *Journal of Water Resources Planning and Management*, 140(1), 40-54.

Kansiime, M. K., Wambugu, S. K., & Shisanya, C. A. (2013). Perceived and actual rainfall trends and variability in Eastern Uganda: Implications for community preparedness and response. *Journal of Natural Sciences Research*, 3(8), 179-194.

Kennedy, J. O. (1986). Dynamic programming: applications to agriculture and natural resources. *Elsevier Applied Science Publishers*.

Kikoyo, D. A., & Nobert, J. (2016). Assessment of impact of climate change and adaptation strategies on maize production in Uganda. *Physics and Chemistry of the Earth, Parts A/B/C*, 93, 37-45.

Leng, G., & Huang, M. (2017). Crop yield response to climate change varies with crop spatial distribution pattern. *Scientific Reports*, 7(1), 1-8.

Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). Offline reinforcement learning: Tutorial, review, and perspectives on open problems. *arXiv preprint arXiv:2005.01643*.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.

Mubiru, D. N., Komutunga, E., Agona, A., Apok, A., & Ngara, T. (2012). Characterising agrometeorological climate risks and uncertainties: Crop production in Uganda. *South African Journal of Science*, 108(3-4), 1-11.

Mubiru, D. N., Radeny, M., Kyazze, F. B.,"; A.,";";"; ";, T.,";";"; "zze, M., ... &"; olomon, D. (2015). Climate trends, risks and coping strategies in smallholder farming systems in Uganda. *Climate Risk Management*, 22, 4-21.

Nimusiima, A., Basalirwa, C. P. K., Majaliwa, J. G. M., Mbogga, S. M., Mwavu, E. N., Namaalwa, J., & Okello-Onen, J. (2013). Nature and dynamics of climate variability in the Uganda cattle corridor. *African Journal of Environmental Science and Technology*, 7(8), 770-782.

Nsubuga, F. W., Olwoch, J. M., Rautenbach, C. D., & Botai, O. J. (2014). Analysis of mid-twentieth century rainfall trends and variability over southwestern Uganda. *Theoretical and Applied Climatology*, 115(1-2), 53-71.

Nsubuga, F. W., & Rautenbach, H. (2018). Climate change and variability: a review of what is known and ought to be known for Uganda. *International Journal of Climate Change Strategies and Management*, 10(5), 752-771.

Oh, J., Farquhar, G., Kemaev, I., Valko, M., Heess, N., & Silver, D. (2025). Discovering state-of-the-art reinforcement learning algorithms. *Nature*, 648, 312–319. https://doi.org/10.1038/s41586-025-09761-x

Ogwang, B. A., Chen, H., Li, X., & Gao, C. (2015). The influence of topography on East African October to December climate: Sensitivity experiments with RegCM4. *Advances in Meteorology*, 2015.

Orlove, B., Roncoli, C., Kabugo, M., & Majugu, A. (2010). Indigenous climate knowledge in southern Uganda: the multiple components of a dynamic regional system. *Climatic Change*, 100(2), 243-265.

Patt, A., & Gwata, C. (2002). Effective seasonal climate forecast applications: examining constraints for subsistence farmers in Zimbabwe. *Global Environmental Change*, 12(3), 185-195.

Powell, W. B. (2007). *Approximate dynamic programming: Solving the curses of dimensionality*. John Wiley & Sons.

Prasad, A. K., Chai, L., Singh, R. P., & Kafatos, M. (2006). Crop yield estimation model for Iowa using remote sensing and surface parameters. *International Journal of Applied Earth Observation and Geoinformation*, 8(1), 26-33.

Puterman, M. L. (1994). *Markov decision processes: discrete stochastic dynamic programming*. John Wiley & Sons.

Roncoli, C., Ingram, K., & Kirshen, P. (2002). Reading the rains: Local knowledge and rainfall forecasting in Burkina Faso. *Society & Natural Resources*, 15(5), 409-427.

Rose, D. C., Sutherland, W. J., Parker, C., Lobley, M., Winter, M., Morris, C., ... & Dicks, L. V. (2016). Decision support tools for agriculture: Towards effective design and delivery. *Agricultural Systems*, 149, 165-174.

Rossiter, D. G., Liu, J., Carlisle, S., & Zhu, A. X. (2004). Can soil classification be used to predict rice yield? *Geoderma*, 114(3-4), 203-227.

Rurinda, J., Mapfumo, P., van Wijk, M. T.,"; umu, F., Rufino, M. C., Chikowo, R., & Giller, K. E. (2014). Sources of vulnerability to a variable and changing climate among smallholder households in Zimbabwe: A participatory analysis. *Climate Risk Management*, 3, 65-78.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

Steduto, P., Hsiao, T. C., Raes, D., & Fereres, E. (2009). AquaCrop—The FAO crop model to simulate yield response to water: I. Concepts and underlying principles. *Agronomy Journal*, 101(3), 426-437.

Stigter, C. J., Zheng, D., Onyewotu, L. O. Z., & Mei, X. (2005). Using traditional methods and indigenous technologies for coping with climate variability. *Increasing Climate Variability and Change*, 70, 255-271.

Sultan, B., Defrance, D., & Iizumi, T. (2020). Evidence of crop production losses in West Africa due to historical global warming. *Environmental Research Letters*, 15(4), 044016.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

Tao, F., Yokozawa, M., Xu, Y., Hayashi, Y., & Zhang, Z. (2008). Climate changes and trends in phenology and yields of field crops in China, 1981–2000. *Agricultural and Forest Meteorology*, 138(1-4), 82-92.

Tao, F., Rötter, R. P., Palosuo, T., Díaz-Ambrona, C. G. H., Mínguez, M. I., Semenov, M. A., ... & Schulman, A. H. (2022). Designing future barley ideotypes using a crop model ensemble. *European Journal of Agronomy*, 82, 144-162.

Thornton, P. K., Jones, P. G., Alagarswamy, G., Andresen, J., & Herrero, M. (2009). Adapting to climate change: Agricultural system and household impacts in East Africa. *Agricultural Systems*, 103(2), 73-82.

Thornton, P. K., Ericksen, P. J., Herrero, M., & Challinor, A. J. (2014). Climate variability and vulnerability to climate change: A review. *Global Change Biology*, 20(11), 3313-3328.

Tittonell, P., Vanlauwe, B., Leffelaar, P. A., Rowe, E. C., & Giller, K. E. (2007). Exploring diversity in soil fertility management of smallholder farms in western Kenya: I. Heterogeneity at region and farm scale. *Agriculture, Ecosystems & Environment*, 110(3-4), 149-165.

Tzounis, A., Katsoulas, N., Bartzanas, T., & Kittas, C. (2017). Internet of Things in agriculture, recent advances and future challenges. *Biosystems Engineering*, 164, 31-48.

UBOS. (2021). *Uganda Bureau of Statistics Statistical Abstract 2021*. Kampala, Uganda.

Van Ittersum, M. K., Cassman, K. G., Grassini, P., Wolf, J., Tittonell, P., & Hochman, Z. (2013). Yield gap analysis with local to global relevance—a review. *Field Crops Research*, 143, 4-17.

Wortmann, C. S., & Eledu, C. A. (1999). *Uganda's agroecological zones: A guide for planners and policy makers*. CIAT.

Wu, L., Chen, K., Yen, S. Y., & Chen, C. H. (2022). Deep reinforcement learning for nitrogen management in crop production. *Computers and Electronics in Agriculture*, 197, 106988.

Yang, Y., Cui, Y., Rongsheng, W., Chen, X., Zhang, H., & Zhang, L. (2020). Deep reinforcement learning–based irrigation scheduling. *Frontiers in Plant Science*, 11, 601478.

You, J., Li, X., Low, M., Lobell, D., & Ermon, S. (2017). Deep Gaussian process for crop yield prediction based on remote sensing data. *Proceedings of the AAAI Conference on Artificial Intelligence*, 31(1).
