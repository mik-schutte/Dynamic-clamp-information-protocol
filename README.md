# Dynamic-clamp-information-protocol 

## About
---
Uses an artificial neural network (ANN) to generate an excitatory and inhibitory conductance based on the absence or presence of a hidden state. This input can be utilized to estimate the mutual information.

The dynamic clamp input (conductance) generation method is described in:<br>
Schutte, M. and Zeldenrust, F. (2021) Increased neural information transfer for a conductance input: a dynamic clamp approach to study information flow. Msc. University of Amsterdam. Available at: https://scripties.uba.uva.nl

The mutual information estimation protocol is described in:<br>
Zeldenrust, F., de Knecht, S., Wadman, W. J., Denève, S., Gutkin, B., Knecht, S. De, Denève, S. (2017). Estimating the Information Extracted by a Single Spiking Neuron from a Continuous Input Time Series. _Frontiers in Computational Neuroscience_, 11(June), 49. [doi:10.3389/FNCOM.2017.00049](https://doi.org/10.3389/fncom.2017.00049)

Please cite these references when using this method. 


## Setup
---
The code depends on the following packages (version):<br>
- [numpy (1.20.3)](https://numpy.org/install/)
- pandas (1.2.4)
- scipy (1.6.3)
- matplotlib (3.4.2)
- [seaborn (0.11.1)](https://seaborn.pydata.org/installing.html)
- [Brian2 (2.3)](https://brian2.readthedocs.io/en/stable/introduction/install.html)
 
A more detailed installation instruction is provided at the link.


## Usage
---
To run a mutual information simulation the code has to go through 4 steps:
1. **Input generation**<br> 
  `make_dynamic_experiments` generates a binary stimulus and the respons (input theory) of the artificial neural network to this stimulus. The input theory can both be in the form of a flutuating current or a fluctuating conductance (dynamic clamp).
2. **Input scaling**<br>
`scale_input_theory` scales the input (current or conductance) with a given scaling factor. This will result in a Brian2.TimedArray with the correct unit to be injected into a (model) neuron.
3. **Model initiation & Input injection**<br>
`Barrel_PC` & `Barrel_IN` will initialize a model neuron using the Brian2 package. `Barrel_PC.run(input)` simulates the response of the model neuron to the injected input. Model fitting is described in [Sterl & Zeldenrust (2020)](https://scripties.uba.uva.nl/search?id=715234.)
4. **Mutual information estimation**
`analyze_exp` calculates the mutual information between the binary stimulus, injected input and output spike train as described in [Zeldenrust *et al* (2017)](https://doi.org/10.3389/fncom.2017.00049) 

A run through steps 1-3 are provided in the big_sim.py file. [Schutte & Zeldenrust (2021)](https://scripties.uba.uva.nl) gives a theoretical walk-through of the protocol.


## License
---
This repository has licensed under the MIT licence. Read LICENCE.txt for the full terms and conditions.
 