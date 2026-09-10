# Screening Windows and Observation Horizons

## 1. Longitudinal Time Horizon in the NASA Dataset

The dataset provides 11 discrete inspection milestones over a total duration of 194 hours:
$$T = [0, 24, 47, 71, 94, 116, 139, 149, 161, 171, 194] \text{ hours}$$

To avoid arbitrary post-hoc window tuning (cherry-picking), screening observation windows are defined systematically based on proportional test duration milestones:

| Screening Window | Cutoff Time ($t_{\text{screen}}$) | Available Pre-Screening Steps | Fraction of Stress Life | Physical State at Cutoff | Experimental Purpose |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Window 1 (Ultra-Early)** | **$t = 24\,\text{h}$** | Steps 0, 1 (2 points) | 12.4% | $\Delta C \in [0.44\%, 0.83\%]$, $\Delta\text{ESR} \in [8.97\%, 9.89\%]$ | Evaluates whether a single initial burn-in cycle provides sufficient discriminatory signal. |
| **Window 2 (PRIMARY)** | **$t = 47\,\text{h}$** | Steps 0, 1, 2 (3 points) | **24.2%** | $\Delta C \in [1.24\%, 1.86\%]$, $\Delta\text{ESR} \in [16.89\%, 17.46\%]$ | **Recommended Primary Benchmark**: Allows calculation of rate-of-change ($\frac{dC}{dt}$) while remaining 10x below static spec limits. |
| **Window 3 (Standard ESS)** | **$t = 71\,\text{h}$** | Steps 0, 1, 2, 3 (4 points) | 36.6% | $\Delta C \in [1.60\%, 2.48\%]$, $\Delta\text{ESR} \in [24.68\%, 25.24\%]$ | Standard burn-in duration; establishes multi-step trajectory slope and acceleration. |
| **Window 4 (Mid-Stress)** | **$t = 94\,\text{h}$** | Steps 0, 1, 2, 3, 4 (5 points) | 48.5% | $\Delta C \in [2.95\%, 3.89\%]$, $\Delta\text{ESR} \in [28.98\%, 30.78\%]$ | Mid-life diagnostic cutoff where accelerated degrader C4 begins sharp upward inflection. |

---

## 2. Scientific Rationale for Primary Window ($t_{\text{screen}} = 47\,\text{h}$)

1. **Physical Incubation Criteria:**
   At $t = 47\,\text{h}$, the maximum recorded capacitance drop across the entire lot is $1.86\%$ (Unit `C4`), which is less than one-tenth of the MIL-PRF-62F $20\%$ static limit. Traditional specification screening is completely blind to failure risks at this point (100% PASS rate).
2. **Kinetic Derivative Availability:**
   Having 3 chronological inspection points ($0\,\text{h}$, $24\,\text{h}$, $47\,\text{h}$) allows the calculation of first-order drift velocities:
   $$\left.\frac{d\,\Delta C}{dt}\right|_{t=47\text{h}} = \frac{\Delta C(47\text{h}) - \Delta C(24\text{h})}{23\text{h}}$$
   This provides the minimal temporal depth required to test whether dynamic statistical and AI models can detect subtle velocity divergence.
3. **Realistic Aerospace ESS Analogy:**
   In spacecraft flight lot acceptance testing, burn-in typically consumes 15% to 25% of expected operational life to precipitate latent infant mortality without excessively eroding the remaining useful life of flight-bound components.

---

## 3. Fixed Window Evaluation Protocol
All comparative screening models (Traditional, Dynamic Statistical, AI/ML) will be evaluated across the exact same windows ($t_{\text{screen}} \in \{24\text{h}, 47\text{h}, 71\text{h}\}$) to assess how model discrimination scales with screening duration.
