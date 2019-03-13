# Evaluation

## Split train and test set

### Input

All Beethoven's string quartets in CSV files (or in a single CSV file).

### Parameters

- Split by phrases
- Split by movements
- Split by pieces
- Split ratio

### Output

Two list of indices for train and test set.
Each list contains tuples of start index and end index of a chord progression.

## Preprocess dataset

### Overview

This module will convert CSV file format into a text file ready to fetch into some models.

### Input

- A CSV file
- A list contains locations of chord progressions

### Parameters

- Augment data by changing global_key
- Format of chord symbols and the information they contain

### Output

A text file.
Each line represent a chord progression.
In each line, chord symbols are seperated by space.

## Model

### Overview

The model takes a chord progression as input and assigns a probability to it.