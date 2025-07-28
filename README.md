# DynamicTRF

The code for [Dynamic modeling of EEG responses to natural speech reveals earlier processing of predictable words](https://doi.org/10.1101/2024.08.26.609779)

This work depends on [nnTRF](https://github.com/powerfulbean/nnTRF).


## Roadmap
🚧 In Progress | ✅ Completed | 🧪 Testing | 🔜 Planned 

### 📦 v0.0.1
✅ Refactor the code while reproducing results in the paper

### 📦 v1.0.0
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
Dou, J., Anderson, A. J., White, A. S., Norman-Haignere, S. V., & Lalor, E. C. (2024). Dynamic modeling of EEG responses to natural speech reveals earlier processing of predictable words. bioRxiv, 2024-08.
```
@article {Dou2024.08.26.609779,
	author = {Dou, Jin and Anderson, Andrew J. and White, Aaron S. and Norman-Haignere, Samuel V. and Lalor, Edmund C.},
	title = {Dynamic modeling of EEG responses to natural speech reveals earlier processing of predictable words},
	elocation-id = {2024.08.26.609779},
	year = {2024},
	doi = {10.1101/2024.08.26.609779},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/08/26/2024.08.26.609779},
	eprint = {https://www.biorxiv.org/content/early/2024/08/26/2024.08.26.609779.full.pdf},
	journal = {bioRxiv}
}
```
