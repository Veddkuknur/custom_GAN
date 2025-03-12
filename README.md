# A Custom Generative Adversarial Network (GAN) for Multi-Class Image Generation

### Overview
This project focuses on designing and implementing a **Generative Adversarial Network (GAN)** to produce novel 32×32 pixel images from multiple classes. Building on the foundational GAN framework proposed by Goodfellow et al., the architecture comprises a **Generator** (responsible for creating new images) and a **Discriminator** (which discerns real images from generated ones). Through iterative adversarial training, the Generator becomes adept at synthesizing realistic, multi-class images, while the Discriminator refines its ability to detect fake samples.

### Key Goals
1. **Multi-Class Image Generation**: Generate 32×32 images across different classes within the dataset.  
2. **Custom Architecture**: Design a custom network rather than using a standard DCGAN structure, balancing computational efficiency and expressive power.  
3. **Quality and Diversity**: Ensure that generated samples are not only visually plausible but also diverse across all classes. Metrics like FID or Inception Score may be adapted for lower-resolution outputs to validate this.

### Technical Highlights
- **Data Preprocessing**: The dataset is normalized and possibly augmented such as flips, shift to broaden the variety of training samples.  
- **Generator Architecture**: A sequence of transposed convolutional layers grows the latent representation into a 32×32 image.  
- **Discriminator Architecture**: Convolutional layers with increasing depth help the model discriminate real images from generated ones, potentially using methods like spectral normalization for stability.  
- **Training and Optimization**: Typical optimizers include **Adam** or **RMSProp**, with careful selection of learning rate and momentum. The adversarial training loop continues until both Generator and Discriminator converge.  
- **Evaluation Metrics**: Assess realism through **visual inspection**, **Frechet Inception Distance (FID)**, and **Inception Score**. Per-class accuracy of the Discriminator also aids in gauging how well it learns distinctive features of each class.

### Results and Observations
During early training stages, generated images might appear noisy or incomplete. Over time, the Generator refines class-specific features (shapes, color distributions, textures) and produces clearer, more distinguishable results. Careful tuning of hyperparameters (learning rate, batch size, etc.) is critical for achieving stable training.
