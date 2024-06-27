# vito

Face &amp; Gesture recognition

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x installed on your machine.
- Necessary Python packages installed. You can install them using:

```bash
pip install -r requirements.txt
```

## Installation

Clone the repository and navigate to the project directory

```bash
    git clone https://github.com/israelmanzi/vito && cd vito
```

## Usage

1. Run `create_dataset.py` to generate the initial dataset.

```bash
python create_dataset.py
```

2. Execute `create_clusters.py` to create clusters from the dataset.

```bash
python create_clusters.py
```

3. Execute `review_clusters.py` to display images in clusters created.

```bash
python review_clusters.py
```

4. Use `rearrange_data.py` to rearrange the clustered data for training.

```bash
python rearrange_data.py
```

5. Execute `review_dataset.py` to review images before training.

```bash
python review_dataset.py
```

6. Train the model using `train_model.py`.

```bash
python train_model.py
```

7. Finally, run `make_predictions.py` to make predictions using the trained model.

```bash
python make_predictions.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
