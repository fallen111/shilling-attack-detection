Here's a comprehensive README for your thesis project codes:

---

# Detecting Shilling Attacks Using Recommender Systems with Random Forests

This repository contains the code and data for the thesis project "Detecting Shilling Attacks Using Recommender Systems with Random Forests" by Fatemeh Khatami, a Master's student in Information Systems at Khaje Nasir University. The project aims to detect shilling attacks in recommender systems using random forests. Some of the code for implementing random forests is adapted from [SDLib](https://github.com/Coder-Yu/SDLib).

## Overview

Shilling attacks pose a significant threat to the integrity of recommender systems. This thesis explores the use of random forests to detect such attacks. The project consists of two main stages:

1. **Experimentation with Random Forest Settings:**
   - In this stage, various settings for random forests are tested to determine the most effective configuration for detecting shilling attacks.

2. **Evaluation of Overfitting in Random Forests:**
   - This stage investigates whether random forests overfit the data using cross-validation. Two distinct settings of random forests are trained on a large injected dataset to balance and increase the complexity of the data space. These trained models are then tested on a dataset created in the first stage.



## Installation and Setup

To run the code, you'll need Python 3.8 and the following Python packages:
- NumPy
- Pandas
- Scikit-learn

You can install the required packages using pip:
```sh
pip install numpy pandas scikit-learn
```

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/detecting-shilling-attacks.git
   cd detecting-shilling-attacks
   ```

## Citation

If you use this code or dataset in your research, please cite this thesis:

```
@mastersthesis{khatami2024detectingshilling,
  title={Detecting Shilling Attacks Using Recommender Systems with Random Forests},
  author={Fatemeh Khatami},
  school={Khaje Nasir University},
  year={2024}
}
```
## Contact

For any questions or issues, please contact Fatemeh Khatami at  or open an issue in this repository.

---

Feel free to adjust the details as necessary to fit your specific project and preferences.
