# Look, Listen, and Act: Towards Audio-Visual Embodied Navigation

A crucial ability of mobile intelligent agents is to integrate the evidence from multiple sensory inputs in an environment and to make a sequence of actions to reach their goals. In this paper, we attempt to approach the problem of Audio-Visual Embodied Navigation, the task of planning the shortest path from a random starting location in a scene to the sound source in an indoor environment, given only raw egocentric visual and audio sensory data. To accomplish this task, the agent is required to learn from various modalities, i.e., relating the audio signal to the visual environment. Here we describe an approach to audio-visual embodied navigation that takes advantage of both visual and audio pieces of evidence. Our solution is based on three key ideas: a visual perception mapper module that constructs its spatial memory of the environment, a sound perception module that infers the relative location of the sound source from the agent, and a dynamic path planner that plans a sequence of actions based on the audio-visual observations and the spatial memory of the environment to navigate toward the goal. Experimental results on a newly collected Visual-Audio-Room dataset using the simulated multi-modal environment demonstrate the effectiveness of our approach over several competitive baselines.

The paper and video can be found at [Paper](https://arxiv.org/pdf/1912.11684.pdf), [Video](https://www.youtube.com/watch?v=WMpddhYZ1bc). You can find more information at our [project page](http://avn.csail.mit.edu/).

## Get Started
- Clone the repo and cd into it.
  ```
  git clone https://github.com/zhang-yw/avn.git
  cd avn
  ```
  
  Download the data from [Google Drive](https://drive.google.com/file/d/1uGoJBPU8qNUzTbiwjwuPH3K2r0d44Ky8/view?usp=sharing) and unzip it. 
  
  We are using a deprecated version of ai2thor. 
  
- Build your environment(optional). 
  
  You will need to install Unity Editor version 2017.3.1f1 for OSX (Linux Editor is currently in Beta) from [Unity Download Archive](https://unity3d.com/get-unity/download/archive). Then run the following commands from the ai2thor base directory. 
  
  ```
  pip install invoke
  invoke local-build
  ```
  This will create a build beneath the directory 'unity/builds/local-build/thor-local-OSXIntel64.app'. To use this build in your code, make the following change:
  
  ```python
  controller = ai2thor.controller.Controller()
  controller.local_executable_path = "<BASE_DIR>/unity/builds/local-build/thor-local-OSXIntel64.app/Contents/MacOS/thor-local-OSXIntel64"
  controller.start()
  ```
  
  The scenes we are using are new1-7.

- Download the dataset.

  Downlad the dataset in H5py format from [Google Drive](https://drive.google.com/file/d/1uGoJBPU8qNUzTbiwjwuPH3K2r0d44Ky8/view?usp=sharing).

If you think our work is useful, please consider citing use with

```
@article{gan2019look,
  title={Look, listen, and act: Towards audio-visual embodied navigation},
  author={Gan, Chuang and Zhang, Yiwei and Wu, Jiajun and Gong, Boqing and Tenenbaum, Joshua B},
  journal={arXiv preprint arXiv:1912.11684},
  year={2019}
}
```
