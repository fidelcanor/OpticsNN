# OpticsNN
Neural Networks for Optical Inverse Design

1. scatter_net_convolution_train.py is to be used to train a Neural Network with a single 1D convolution layer. This architecture works well with the three layer case.

2. scatter_net_convolution_match.py does a single feedforward propagation to return the radii that created a given spectrum using the architecture from (1). The function get_spect() has an optional parameter "singular". If set to False, it will select a random spectrum from the data set. If set to True, you must also provide a test file that contains a specific spectrum that is to be matched.

3. scatter_net_double_convolution_train.py is to be used to train a Neural Network that has two 1D convolution layers. This architecture works better for the four (and potentially more) layer cases.

4. scatter_net_double_convolution_match.py does a single feedforward propagation to return the radii that created a given spectrum using the architecture from (3). The function get_spect() has an optional parameter "singular". If set to False, it will select a random spectrum from the data set. If set to True, you must also provide a test file that contains a specific spectrum that is to be matched.

5. scatter_net_2DConvolution_train.py is to be used to train a Neural Network whose first layer is a 2D convolution layer and the second is a 1D convolution layer. The input is now a 2-by-N matrix, where N is the number of points sampled in the spectrum -- there are essentially N pairs of points in the matrix and each pair is represented as (wavelength, scattering cross section). This architecture needs to be tested more extensively.
