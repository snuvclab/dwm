---
layout: project_page
permalink: /

title: Dexterous World Models
tab_title: DWM
social_title: Dexterous World Models (DWM)
social_description: Scene-Action-Conditioned Video Diffusion for Embodied Interaction
description:  Dexterous World Models (DWM) is a scene-action-conditioned video diffusion model that simulates embodied dexterous human interactions in static 3D scenes using egocentric hand trajectories.
venue: CVPR 2026
authors:
  - name: Byungjun Kim
    affiliation: [1]
    equal_contributor: true
    homepage: https://bjkim95.github.io
  - name: Taeksoo Kim
    affiliation: [1]
    equal_contributor: true
    homepage: https://taeksuu.github.io/
  - name: Junyoung Lee
    affiliation: [1]
    homepage: https://junc0ng.github.io/
  - name: Hanbyul Joo
    affiliation: [1, 2]
    homepage: https://jhugestar.github.io/
affiliations:
  - Seoul National University
  - RLWRLD
paper: static/pdf/dwm_arxiv_reduced.pdf
# video: https://www.youtube.com/results?search_query=turing+machine
arxiv: https://arxiv.org/abs/2512.17907
code: https://github.com/snuvclab/dwm
---

<!-- Teaser video-->
<section class="hero teaser">
  <div class="container is-max-desktop">
    <h2 class="subtitle has-text-centered">
      <b>TL;DR:</b>  A scene-action-conditioned video diffusion model <br>to simulate embodied dexterous actions in a given static 3D scene
    </h2>
    <div class="hero-body">
      <video id="teaser" autoplay muted loop playsinline height="100%">
        <!-- Your video here -->
        <source src="static/videos/DWM_teaserv2_trim.mp4" type="video/mp4">
      </video>
      <style>
        #teaser {
          clip-path: inset(2px 2px 2.5px 2px); /* Crop 2px from each side */
        }
      </style>
    </div>
  </div>
</section>
<!-- End teaser video -->

<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
Recent progress in 3D reconstruction has made it easy to create realistic digital twins from everyday environments. However, current digital twins remain largely static—limited to navigation and view synthesis without embodied interactivity. To bridge this gap, we introduce Dexterous World Model (DWM), a scene-action-conditioned video diffusion framework that models how dexterous human actions induce dynamic changes in static 3D scenes. 
Given a static 3D scene rendering and an egocentric hand motion sequence, DWM generates temporally coherent videos depicting plausible human–scene interactions. Our approach conditions video generation on (1) static scene renderings following a specified camera trajectory to ensure spatial consistency, and (2) egocentric hand mesh renderings that encode both geometry and motion cues in the egocentric view to model action-conditioned dynamics directly. To train DWM, we construct a hybrid interaction video dataset: synthetic egocentric interactions provide fully aligned supervision for joint locomotion–manipulation learning, while fixed-camera real-world videos contribute diverse and realistic object dynamics.
Experiments demonstrate that DWM enables realistic, physically plausible interactions, such as grasping, opening, or moving objects, while maintaining camera and scene consistency. This framework establishes the first step toward video diffusion-based interactive digital twins, enabling embodied simulation from egocentric actions.
        </div>
    </div>
</div>

---

## Results

<!-- Video carousel - Synthetic Scenes -->
<section class="section results-section">
  <div class="results-inner">
    <h3 class="title is-4">DWM Results in Synthetic Scenes</h3>
    <div id="results-carousel-synthetic" class="carousel results-carousel">
      <div class="item item-video1">
        <video poster="static/thumbs/synthetic_1d07.jpg" id="video-synthetic-1" controls muted loop playsinline preload="metadata">
          <source src="static/videos/synthetic/results/1d07_22-09-42_00018_merged_comp.mp4" type="video/mp4">
        </video>
      </div>
      <div class="item item-video2">
        <video poster="static/thumbs/synthetic_1a8d.jpg" id="video-synthetic-2" controls muted loop playsinline preload="metadata">
          <source src="static/videos/synthetic/results/1a8d_12-12-31_00009_merged_comp.mp4" type="video/mp4">
        </video>
      </div>
      <div class="item item-video3">
        <video poster="static/thumbs/synthetic_4a87.jpg" id="video-synthetic-3" controls muted loop playsinline preload="metadata">
          <source src="static/videos/synthetic/results/4a87_16-51-13_00001_merged_comp.mp4" type="video/mp4">
        </video>
      </div>
      <div class="item item-video3">
        <video poster="static/thumbs/synthetic_1d89.jpg" id="video-synthetic-3" controls muted loop playsinline preload="metadata">
          <source src="static/videos/synthetic/results/1d89_11-54-56_00019_merged_comp.mp4" type="video/mp4">
        </video>
      </div>
    </div>
  </div>
</section>
<!-- End video carousel - Synthetic Scenes -->

<!-- Video carousel - Real-world Scenes -->
<section class="section results-section">
  <div class="results-inner">
    <h3 class="title is-4">DWM Results in Real-world Scenes</h3>
    <div id="results-carousel-realworld" class="carousel results-carousel">
      <div class="item item-video1">
        <video poster="static/thumbs/realworld_00039.jpg" id="video-realworld-1" controls muted loop playsinline preload="metadata">
          <source src="static/videos/real_world/results/00039_merged_comp.mp4" type="video/mp4">
        </video>
      </div>
      <div class="item item-video2">
        <video poster="static/thumbs/realworld_00053.jpg" id="video-realworld-2" controls muted loop playsinline preload="metadata">
          <source src="static/videos/real_world/results/00053_merged_comp.mp4" type="video/mp4">
        </video>
      </div>
      <div class="item item-video3">
        <video poster="static/thumbs/realworld_00013.jpg" id="video-realworld-3" controls muted loop playsinline preload="metadata">
          <source src="static/videos/real_world/results/00013_merged_comp.mp4" type="video/mp4">
        </video>
      </div>
      <div class="item item-video3">
        <video poster="static/thumbs/realworld_00054.jpg" id="video-realworld-3" controls muted loop playsinline preload="metadata">
          <source src="static/videos/real_world/results/00054_merged_comp.mp4" type="video/mp4">
        </video>
      </div>
    </div>
  </div>
</section>
<!-- End video carousel - Real-world Scenes -->

## Method Overview
![Method Overview](/static/image/overview.png)

We decompose embodied actions within a static 3D scene $$ \mathbf{S}_{0} $$ into an egocentric camera motion $$\mathcal{C}_{1:F}$$ and a hand manipulation trajectory $$\mathcal{H}_{1:F}$$.
Given these components, we render the static scene video and the hand-only video by following the egocentric camera motion $$\mathcal{C}_{1:F}$$.
These two rendered videos serve as conditioning inputs to our video diffusion model, enabling it to generate egocentric visual simulations of the specified action within the given static 3D scene.

### 💡 Key Insight #1: Hybrid Interaction-Static Paired Video Dataset
> **DWM learns joint locomotion-manipulation from synthetic egocentric video pairs, while absorbing diverse real-world dynamics from fixed-camera interaction videos.**

<video autoplay muted loop playsinline style="width: 100%; max-width: 800px; margin: 1rem auto; display: block;">
  <source src="static/videos/dataset/both_comp.mp4" type="video/mp4">
</video>

<details>
<summary><strong>Click to expand for details</strong></summary>
Training DWM requires paired videos consisting of
(i) an interaction video,
(ii) a corresponding static-scene video, and
(iii) an aligned hand video,
all captured under the same camera trajectory.
In real-world settings, acquiring such perfectly aligned pairs is challenging.

To address this, we first leverage the synthetic 3D human–scene interaction dataset <a href="https://jnnan.github.io/trumans/" target="_blank">TRUMANS</a>.
We obtain egocentric interaction videos in TRUMANS by placing a virtual camera between the agent's eyes, yielding consistent egocentric viewpoints across sequences.

<video autoplay muted loop playsinline preload="metadata" style="width: 100%; max-width: 800px; margin: 1rem auto; display: block;">
  <source src="static/videos/dataset/third_to_ego_comp.mp4" type="video/mp4">
</video>

Crucially, the synthetic setup allows us to disable object dynamics and re-render the scene along the identical egocentric camera trajectory, producing a clean static-scene video.
In addition, we render only the agent's hand meshes to obtain perfectly aligned egocentric hand videos.
Through this process, we construct aligned triplets of interaction videos, static scene videos, and hand videos.

<video autoplay muted loop playsinline preload="metadata" style="width: 100%; max-width: 800px; margin: 1rem auto; display: block;">
  <source src="static/videos/dataset/synthetic_comp.mp4" type="video/mp4">
</video>

However, purely synthetic data covers a limited range of interactions and lacks the rich dynamics present in real-world environments.
Since capturing fully paired egocentric videos in the real world remains impractical, we complement the synthetic data with fixed-camera real-world interaction videos from <a href="https://taste-rob.github.io/" target="_blank">Taste-Rob</a>.

In the fixed-camera setup, we treat the first frame of each video as the static scene and replicate it across all frames to form a static-scene video.
Hand videos are extracted by running <a href="https://geopavlakos.github.io/hamer/" target="_blank">HaMeR</a> on the interaction sequences.
Although camera motion is absent, this procedure yields real-world static–interaction video pairs that capture realistic object dynamics and contact behaviors.

<video autoplay muted loop playsinline preload="metadata" style="width: 100%; max-width: 800px; margin: 1rem auto; display: block;">
  <source src="static/videos/dataset/real_comp.mp4" type="video/mp4">
</video>

By combining egocentric synthetic pairs (enabling joint locomotion–manipulation learning) with fixed-camera real-world pairs (providing diverse and realistic dynamics),
our hybrid dataset design allows DWM to learn robust action-conditioned scene dynamics while sidestepping the prohibitive cost of real-world paired capture.


</details>

### 💡 Key Insight #2: Inpainting Priors for Residual Dynamics Learning

> **A full-mask inpainting diffusion model becomes an identity function with generative priors.**

<video autoplay muted loop playsinline style="width: 100%; max-width: 800px; margin: 1rem auto; display: block;">
  <source src="static/videos/inpaint/0_inpaint.mp4" type="video/mp4">
</video>

<details>
<summary><strong>Click to expand for details</strong></summary>

When an inpainting video diffusion model is given a full mask ($m = 1$), it reproduces the input video, effectively behaving as an identity mapping with a generative prior.
In our case, this allows the static scene video to serve as a valid identity input, as it already encodes egocentric navigation motion rendered along the camera trajectory $\mathcal{C}_{1:F}$.

<video autoplay muted loop playsinline style="width: 100%; max-width: 800px; margin: 1rem auto; display: block;">
  <source src="static/videos/inpaint/1_init.mp4" type="video/mp4">
</video>

Based on this observation, we initialize our model as a full-mask inpainting model, using the static scene video as a navigation-only baseline.
This initialization encourages the model to preserve the scene appearance and egocentric camera motion, providing a stable reference before introducing any manipulation-induced changes.

<video autoplay muted loop playsinline style="width: 100%; max-width: 800px; margin: 1rem auto; display: block;">
  <source src="static/videos/inpaint/2_train.mp4" type="video/mp4">
</video>

To model interaction dynamics, we condition the model on the dexterous hand trajectory $\mathcal{H}_{1:F}$.
This additional signal guides the model to focus on <strong>residual dynamics driven by manipulation</strong>, rather than re-learning navigation effects already present in the static scene video.
As a result, training becomes more stable and the learned dynamics naturally disentangle navigation from manipulation.
</details>


### Citation
```
@inproceedings{kim2026dwm,
  title={Dexterous World Models},
  author={Kim, Byungjun and Kim, Taeksoo and Lee, Junyoung and Joo, Hanbyul},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
