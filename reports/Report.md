# Report

Last time:

- we normalized the count measures into percentages, and then used Dirichlet for emission distribution.



This time:

- Play with Compositional-LV simulated data

- Set up a new model, where we use counts as observations, and Poisson as emission
- Set the dynamics in log space





## Dirichlet model

- Compositional LV model (no inputs)

  The model performs well in the following cases

  - Full observation

  - 20% missing obs

  - 40% missing obs

  - 60% missing ops

    ![epoch_200](/Users/leah/Columbia/courses/19summer/microbialdynamics/src/rslts/test_clv/D190821_152301_np_32_bs_1_lr_0.001_epoch_200_seed_2/R_square/epoch_200.png)

![rslts](/Users/leah/Columbia/courses/19summer/microbialdynamics/reports/rslts.png)

- Only one patient's data

  - 

  

- Dynamics in log space

  Implementation:

  - From observation space to hidden space: apply a log transformation.
  - From hidden space to observation space: apply an exp transformation.

  Results:

  - No big improvements

TODO:

- make a probabilistic model, add log dynamics
- 

## Poisson model

- Works well with the CLV data
- Does not work well with the real data



## Some thoughts

Data..

particle degeneracy

