# DynamicTRF

The code for [Dynamic modeling of EEG responses to natural speech reveals earlier processing of predictable words](https://doi.org/10.1101/2024.08.26.609779)

This work depends on [nnTRF](https://github.com/powerfulbean/nnTRF).


## Roadmap
 🔜 Planned | 🚧 In Progress | 🧪 Testing | ✅ Completed 

### v0.0.1
✅ Refactor the code while reproducing results in the paper

### v1.0.0
🚧 remove dependency on the old mTRFpy  
🔜 remove dependency on the old stimrespflow trainer  
🔜 easier to use method for creating torch dataset required by the model, with code examples  
🔜 switched to light-weight stimrespflow library, with light-weight trainer  
🔜 user-friendly function to start dynamic TRF analysis, with just one call


## Installation

```sh
    pip install git+https://github.com/powerfulbean/DynamicTRF.git
```


## Citing DynamicTRF
Dou, J., Anderson, A. J., White, A. S., Norman-Haignere, S. V., & Lalor, E. C. (2025). Dynamic modeling of EEG responses to natural speech reveals earlier processing of predictable words. PLOS Computational Biology, 21(4), e1013006.
```
@article{dou2025dynamic,
  title={Dynamic modeling of EEG responses to natural speech reveals earlier processing of predictable words},
  author={Dou, Jin and Anderson, Andrew J and White, Aaron S and Norman-Haignere, Samuel V and Lalor, Edmund C},
  journal={PLOS Computational Biology},
  volume={21},
  number={4},
  pages={e1013006},
  year={2025},
  publisher={Public Library of Science San Francisco, CA USA}
}
```
