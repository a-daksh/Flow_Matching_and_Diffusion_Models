# Flow Matching and Diffusion Models: Ablation Study Report

## Overview

The report presents a ablation study to understand the key parameters that affect Flow Matching and Diffusion models using MNIST generation. In total we play with 4 parameters to answer the following questions:

1. **Data Fraction:** How much training data is needed for effective generation?
2. **Guidance scale:** What is the effect of guidance on sample quality?
3. **Inference Timesteps:** How many timesteps does it take to generate good quality images?
4. **Sigma:** How much does sigma affect novel generation?

**Note on Image Layout:** The image blocks shown throughout this report follow a consistent **11×10 grid structure**, representing 10 generations for each of 11 classes. The 11 classes correspond to object IDs 0-9 plus one additional row for unconditional generation (no guidance). Each row shows 10 different samples from the same class/condition, allowing us to assess both within-class diversity and cross-class differences as we vary different parameters below.

---

## Study A: Data Fraction Effect

**Configuration:**
- Training data fractions: 0.1, 0.25, 0.5, 0.75, 1.0
- Inference: Flow (ODE), guidance=3.0, steps=100

### Results

| Data Fraction | Training Samples | Generated Samples |
|---------------|------------------|-------------------|
| 0.1 | ~6,000 | ![frac_0.1](../outputs/study_a_data_fraction/inference_ode_frac_0.1.png) |
| 0.25 | ~15,000 | ![frac_0.25](../outputs/study_a_data_fraction/inference_ode_frac_0.25.png) |
| 0.5 | ~30,000 | ![frac_0.5](../outputs/study_a_data_fraction/inference_ode_frac_0.5.png) |
| 0.75 | ~45,000 | ![frac_0.75](../outputs/study_a_data_fraction/inference_ode_frac_0.75.png) |
| 1.0 | ~60,000 | ![frac_1.0](../outputs/study_a_data_fraction/inference_ode_frac_1.0.png) |

### Observations

- The overall visual quality of generations remains pretty much consistent across all data fractions.
- In terms of diversity, it highly depends on the object. For example, trousers are nearly identical whether we use 10% or all data, but t-shirts show much more diversity with the full datase. we see stripes and camouflage patterns in generations that don't appear with smaller fractions.
- Regarding clarity, the objects we're dealing with here are relatively simple, but looking at the handbags in the generations, we can examine their straps as an example. This is relatively more complex to generate since rendering straps requires some understanding of how they flow down from the bag, and the learning signal through loss will be small because they're represented by very few pixels. Therefore, the generation with full data fraction shows much better strap definition than the 10% version.

### Conclusion
I feel that data fraction should be as high as possible irrespective of the specific object, but I would like to further validate this intuition by adding numerical metrics using some distance function and comparing potential train-test splits to quantify the differences more rigorously.

---

## Study B: Flow Parameters (Guidance Scale + Inference Steps)

**Configuration:**
- Model: `checkpoint_frac_1.0` (best from Study A)
- Guidance scales: 0.5, 1.0, 2.0, 3.0, 5.0
- Inference steps: 10, 25, 50, 100, 200

### Results

Results organized by guidance scale (rows) and inference steps (columns):

| Guidance \ Steps | 10 | 25 | 50 | 100 | 200 |
|------------------|----|----|----|-----|-----|
| **w = 0.5** | ![w0.5_s10](../outputs/study_b_flow_params/inference_ode_w0.5_steps10_frac_1.0.png) | ![w0.5_s25](../outputs/study_b_flow_params/inference_ode_w0.5_steps25_frac_1.0.png) | ![w0.5_s50](../outputs/study_b_flow_params/inference_ode_w0.5_steps50_frac_1.0.png) | ![w0.5_s100](../outputs/study_b_flow_params/inference_ode_w0.5_steps100_frac_1.0.png) | ![w0.5_s200](../outputs/study_b_flow_params/inference_ode_w0.5_steps200_frac_1.0.png) |
| **w = 1.0** | ![w1.0_s10](../outputs/study_b_flow_params/inference_ode_w1.0_steps10_frac_1.0.png) | ![w1.0_s25](../outputs/study_b_flow_params/inference_ode_w1.0_steps25_frac_1.0.png) | ![w1.0_s50](../outputs/study_b_flow_params/inference_ode_w1.0_steps50_frac_1.0.png) | ![w1.0_s100](../outputs/study_b_flow_params/inference_ode_w1.0_steps100_frac_1.0.png) | ![w1.0_s200](../outputs/study_b_flow_params/inference_ode_w1.0_steps200_frac_1.0.png) |
| **w = 2.0** | ![w2.0_s10](../outputs/study_b_flow_params/inference_ode_w2.0_steps10_frac_1.0.png) | ![w2.0_s25](../outputs/study_b_flow_params/inference_ode_w2.0_steps25_frac_1.0.png) | ![w2.0_s50](../outputs/study_b_flow_params/inference_ode_w2.0_steps50_frac_1.0.png) | ![w2.0_s100](../outputs/study_b_flow_params/inference_ode_w2.0_steps100_frac_1.0.png) | ![w2.0_s200](../outputs/study_b_flow_params/inference_ode_w2.0_steps200_frac_1.0.png) |
| **w = 3.0** | ![w3.0_s10](../outputs/study_b_flow_params/inference_ode_w3.0_steps10_frac_1.0.png) | ![w3.0_s25](../outputs/study_b_flow_params/inference_ode_w3.0_steps25_frac_1.0.png) | ![w3.0_s50](../outputs/study_b_flow_params/inference_ode_w3.0_steps50_frac_1.0.png) | ![w3.0_s100](../outputs/study_b_flow_params/inference_ode_w3.0_steps100_frac_1.0.png) | ![w3.0_s200](../outputs/study_b_flow_params/inference_ode_w3.0_steps200_frac_1.0.png) |
| **w = 5.0** | ![w5.0_s10](../outputs/study_b_flow_params/inference_ode_w5.0_steps10_frac_1.0.png) | ![w5.0_s25](../outputs/study_b_flow_params/inference_ode_w5.0_steps25_frac_1.0.png) | ![w5.0_s50](../outputs/study_b_flow_params/inference_ode_w5.0_steps50_frac_1.0.png) | ![w5.0_s100](../outputs/study_b_flow_params/inference_ode_w5.0_steps100_frac_1.0.png) | ![w5.0_s200](../outputs/study_b_flow_params/inference_ode_w5.0_steps200_frac_1.0.png) |

### Observations

#### Effect of Guidance Scale
- With 0.5 as expected (theoretically) the generations do not necessarily follow the classes prompted. 
- This sort of improves as we move from 0.5 to 1 and majority diversity cases are handled , for example since the shape of heel is much different than any shirt or jacket, no wrong is generated for heel. But shirt and jackets i.e. similar objects are still confused.
- With guidance scale 2, we finally get what we ask for, all generations belong to the right class and similarly for scale 3
- But as we move to scale 5, the generated samples start losing diversity, for example, no generated tshirt has stripes. 

#### Effect of Inference Steps
- As we go from 10 to 25 steps, we do see some noticeable changes like sleeves becoming longer, higher contrast in generated images, and in general more "complete" generations. In going from 25 to 50 steps, this effect continues but is much more subtle. Generations at 50, 100, and 200 steps are virtually identical. This suggests that increasing inference steps improves clarity up to a threshold—around 50 steps—but beyond that point, additional computation does not yield meaningful visual gains.
- However, this doesn't mean we should arbitrarily set it to something like 2000 steps every time, as computational cost scales linearly with more inference steps. So it seems best to stick to around 50 inference steps, as this balances quality with efficiency.

### Conclusion
Building on both observations, I believe the sweet spot for inference lies at around **50-100 steps with a guidance scale of 2-3**. Beyond 50 steps, computational cost increases without meaningful quality improvements. This configuration allows clear generation without artifacts, provides a strong enough conditional signal to reliably generate the correct class, yet maintains variation in patterns within each class—avoiding the diversity collapse we see at higher guidance values.

---

## Study C: Diffusion Sigma Effect

**Configuration:**
- Model: `checkpoint_frac_1.0` (best from Study A)
- Guidance: 3.0 (fixed)
- Inference steps: 100 (fixed)
- Sigma values: 0, 0.1, 0.2, 0.5, 1.0

### Results

| Sigma | Mode | Generated Samples |
|-------|------|-------------------|
| 0.0 | ODE (deterministic baseline) | ![sigma0](../outputs/study_c_diffusion_sigma/inference_sigma0_frac_1.0.png) |
| 0.1 | SDE (low noise) | ![sigma0.1](../outputs/study_c_diffusion_sigma/inference_sigma0.1_frac_1.0.png) |
| 0.2 | SDE (moderate noise) | ![sigma0.2](../outputs/study_c_diffusion_sigma/inference_sigma0.2_frac_1.0.png) |
| 0.5 | SDE (high noise) | ![sigma0.5](../outputs/study_c_diffusion_sigma/inference_sigma0.5_frac_1.0.png) |
| 1.0 | SDE (very high noise) | ![sigma1.0](../outputs/study_c_diffusion_sigma/inference_sigma1.0_frac_1.0.png) |

### Observations

- The first thing I notice in these images is the one with sigma 1—it becomes very clear that here noise dominates the generation process. The final images do have the overall structure of the generated objects (a t-shirt looks like a t-shirt and so on), but all fine details and quality are lost due to this excessive noise, making the images appear grainy and unclear.
- At sigma 0.5, we do see this noise reducing but it's still present and affecting image quality. Lower values like 0.1, 0.2, and 0.3 are much cleaner and sharper. - Although we don't see anything too conclusive by just looking at one sigma value in isolation, the importance of sigma becomes clear when comparing different sigma values side by side. As we increase sigma and induce more noise, the outputs of the model start to differ more between runs. We see some diversity when comparing generations at 0.1 and 0.2, and quite a bit more variation between 0.1 and 0.3.
- This does make sense theoretically: with sigma 0, we are essentially running an ODE solver rather than an SDE, meaning the stochasticity is completely lost and we get deterministic, repeatable outputs. As we increase sigma, we reintroduce randomness into the sampling process, which can help with diversity but at the cost of clarity if set too high.
- TODO: I speculate that if we increase inference timesteps, we might be able to handle the noise better at higher sigma values—essentially giving the model more steps to denoise properly. The graininess at sigma 1 might partly be due to insufficient denoising steps rather than just the noise level itself. But that's an experiment for next time.

### Conclusion
I think sigma is something that will highly depend on the specific task and what outputs we're expecting. There's a clear trade-off here: lower sigma values (0.1-0.2) give cleaner, more consistent images but with less variation between runs, while higher values (0.5-1.0) introduce diversity at the cost of quality. For general purposes, **sigma around 0.2-0.3** seems like a reasonable starting point—it provides some stochasticity for diversity without significantly degrading image quality. However, if we need multiple diverse samples or are doing tasks like data augmentation, slightly higher values might be worth the quality trade-off.

---

