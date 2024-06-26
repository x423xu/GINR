# GINR
This is the implementation of Generative implicit neural representation.
We plan to train functa/mnif/mod-inr as backbone representation for image, point cloud, and nerf.
The latents are generated by lnvae.

# Training
> Notes: For functa and mnif, only one latent tensor is required for each data, the mdoels use latent embedding to fit the latents size and modulated params's size. In this case, only the hierarchical vae is suitable, because they do not assume any explicit layer-wise dependency, or we say the latents blindly have fully mutual-dependences. For loe and mod, we assume directional forward dependency, so the layer vae is more suitable. However, if we ignore the directional dependency and keep the same assumption as functa and mnif, we can still train loe and mod with hierarchical vae.

```bash
# train functa on shapenet
python train_inr.py --config cfgs/train_functa_shapenet.yml --log_dir logs
# train mnif on shapenet
python train_inr.py --config cfgs/train_mnif_shapenet.yml --log_dir logs
# train loe on shapenet
python train_inr.py --config cfgs/train_loe_shapenet.yml --log_dir logs
```
## Configuration Explanation
 - `cache_latents`: Cache the latents for each epoch, do not start from the random noise.
 - `norm_latents`: Use normalization for latents embedding module. This avoids under training of certain experts. Which should be enabled for mnif and loe.

# TODO
- [x] Train functa on shapenet
- [x] Train mnif on shapenet
- [x] Train loe on shapenet
- [x] Train functa with hierarchical vae on shapenet
- [x] Train mnif with hierarchical vae on shapenet
- [x] Train loe with hierarchical vae on shapenet
- [x] Add resume training
- [x] Train functa with layer vae
- [x] Train mnif with layer vae
- [x] Train loe with layer vae
- [ ] Add mixture-of-depth module
