import type { DemoDefinition } from '../types';

/** Architecture entries describe the actually implemented browser model. */
export const DEMOS: DemoDefinition[] = [
  {
    id: 'xor', title: 'XOR laboratory', category: 'Classification', difficulty: 'Introductory',
    description: 'Watch hidden units learn the simplest nonlinear decision boundary.',
    architecture: [2, 8, 8, 1], tags: ['MLP', 'Binary cross-entropy', 'Adam'], accent: '#42d8ff',
    task: 'Predict whether two binary inputs differ.', inputLabels: ['Bit A', 'Bit B'], outputLabels: ['Different'],
  },
  {
    id: 'circles', title: 'Concentric circles', category: 'Classification', difficulty: 'Introductory',
    description: 'Separate inner and outer rings with a learned nonlinear representation.',
    architecture: [2, 16, 12, 1], tags: ['MLP', 'Nonlinear geometry', '2D'], accent: '#a78bfa',
    task: 'Assign points to the inner or outer ring.', inputLabels: ['x', 'y'], outputLabels: ['Outer ring'],
  },
  {
    id: 'spirals', title: 'Three intertwined spirals', category: 'Classification', difficulty: 'Advanced',
    description: 'A three-class curved boundary that challenges shallow feature maps.',
    architecture: [2, 32, 32, 3], tags: ['Softmax', 'Multiclass', 'Decision boundary'], accent: '#ffb45e',
    task: 'Identify the generating spiral for a noisy point.', outputLabels: ['Spiral A', 'Spiral B', 'Spiral C'],
  },
  {
    id: 'moons', title: 'Two moons', category: 'Classification', difficulty: 'Introductory',
    description: 'Learn a smooth separator between noisy crescent-shaped manifolds.',
    architecture: [2, 16, 8, 1], tags: ['MLP', 'Manifolds', 'Noise'], accent: '#67e8b0',
    task: 'Classify points sampled from two noisy crescents.',
  },
  {
    id: 'gaussian-clusters', title: 'Gaussian discriminant', category: 'Classification', difficulty: 'Introductory',
    description: 'Four overlapping clouds reveal calibrated categorical predictions.',
    architecture: [2, 12, 4], tags: ['Softmax', 'Four classes', 'Uncertainty'], accent: '#60a5fa',
    task: 'Predict the Gaussian component that generated each point.',
  },
  {
    id: 'parity', title: 'Four-bit parity', category: 'Classification', difficulty: 'Advanced',
    description: 'Inspect distributed representations of a combinatorial binary function.',
    architecture: [4, 24, 16, 1], tags: ['MLP', 'Logic', 'Combinatorial'], accent: '#f472b6',
    task: 'Predict whether the number of active bits is odd.',
  },
  {
    id: 'sine-regression', title: 'Sine function approximator', category: 'Regression', difficulty: 'Introductory',
    description: 'Fit a noisy periodic signal and inspect how hidden features compose.',
    architecture: [1, 24, 16, 1], tags: ['MSE', 'Function approximation', 'R²'], accent: '#42d8ff',
    task: 'Estimate sin(πx) from a scalar input.', inputLabels: ['x'], outputLabels: ['sin(πx)'],
  },
  {
    id: 'surface-regression', title: 'Wave surface', category: 'Regression', difficulty: 'Intermediate',
    description: 'Approximate a two-dimensional oscillatory surface from samples.',
    architecture: [2, 32, 24, 1], tags: ['MSE', 'Surface', 'Nonlinear'], accent: '#a78bfa',
    task: 'Approximate sin(πx) cos(πy).',
  },
  {
    id: 'linear-regression', title: 'Linear system identification', category: 'Regression', difficulty: 'Introductory',
    description: 'Recover a linear response while ignoring an irrelevant sensor channel.',
    architecture: [3, 1], tags: ['Linear model', 'MSE', 'Interventions'], accent: '#67e8b0',
    task: 'Learn y = 0.55x₁ − 0.35x₂ + 0.12 under small measurement noise.',
  },
  {
    id: 'forecasting', title: 'Signal forecasting', category: 'Regression', difficulty: 'Intermediate',
    description: 'Predict the next value of a sinusoid from an eight-step context window.',
    architecture: [8, 24, 12, 1], tags: ['Time series', 'Windowed MLP', 'MSE'], accent: '#ffb45e',
    task: 'Predict one future sample from a short history; no recurrent architecture is implied.',
  },
  {
    id: 'image-patterns', title: 'Pixel orientation classifier', category: 'Vision', difficulty: 'Introductory',
    description: 'Train a dense classifier on noisy 8×8 synthetic line images.',
    architecture: [64, 32, 16, 4], tags: ['Synthetic pixels', 'Dense MLP', 'Four classes'], accent: '#60a5fa',
    task: 'Classify vertical, horizontal, diagonal, and anti-diagonal lines.',
  },
  {
    id: 'tiny-digits', title: 'Tiny digit recognition', category: 'Vision', difficulty: 'Intermediate',
    description: 'Recognize ten hand-designed 5×5 digit patterns under pixel noise.',
    architecture: [25, 32, 24, 10], tags: ['Synthetic digits', 'Softmax', 'Dense MLP'], accent: '#42d8ff',
    task: 'Classify synthetic noisy glyphs; this dataset is not MNIST.', outputLabels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
  },
  {
    id: 'autoencoder', title: 'Bottleneck autoencoder', category: 'Representation', difficulty: 'Intermediate',
    description: 'Compress a 25-pixel glyph through a three-unit bottleneck and reconstruct it.',
    architecture: [25, 16, 3, 16, 25], tags: ['Reconstruction', 'Bottleneck', 'MSE'], accent: '#a78bfa',
    task: 'Learn a compact nonlinear representation of synthetic digits.',
  },
  {
    id: 'denoising', title: 'Denoising glyphs', category: 'Representation', difficulty: 'Intermediate',
    description: 'Remove injected pixel noise through a learned encoder-decoder.',
    architecture: [25, 20, 8, 20, 25], tags: ['Denoising', 'Autoencoder', 'Synthetic vision'], accent: '#67e8b0',
    task: 'Reconstruct clean glyphs from independently corrupted versions.',
  },
  {
    id: 'next-token', title: 'Tiny next-token model', category: 'Language', difficulty: 'Intermediate',
    description: 'Train a two-token context MLP on a small, explicit synthetic grammar.',
    architecture: [24, 24, 16, 12], tags: ['Context MLP', 'Vocabulary softmax', 'Synthetic grammar'], accent: '#f472b6',
    task: 'Predict a next word from two one-hot context tokens; this is a small statistical language demo.',
  },
  {
    id: 'attention-sentiment', title: 'Attention pooling microscope', category: 'Language', difficulty: 'Intermediate',
    description: 'Inspect computed scaled dot-product attention and train a sentiment readout.',
    architecture: [6, 12, 1], tags: ['Fixed Q/K/V attention', 'Trainable readout', 'Sentiment'], accent: '#a78bfa',
    task: 'Classify synthetic sentiment phrases after fixed attention pooling; Q/K/V weights are not trained.',
  },
  {
    id: 'anomaly-detection', title: 'Sensor anomaly classifier', category: 'Classification', difficulty: 'Intermediate',
    description: 'Distinguish nominal sensor observations from supervised synthetic anomalies.',
    architecture: [3, 16, 8, 1], tags: ['Supervised anomalies', 'Sensor data', 'Binary classifier'], accent: '#ffb45e',
    task: 'Predict known anomalous observations; this is supervised detection, not unsupervised novelty estimation.',
  },
  {
    id: 'pca', title: 'Principal-component explorer', category: 'Representation', difficulty: 'Intermediate',
    description: 'Compute two orthogonal principal axes by covariance power iteration.',
    architecture: [3, 2, 3], tags: ['PCA', 'Power iteration', 'Linear reconstruction'], accent: '#42d8ff',
    task: 'Find the dominant covariance directions and reconstruct a correlated 3D cloud.',
  },
  {
    id: 'k-means', title: 'K-means clustering', category: 'Representation', difficulty: 'Introductory',
    description: 'Observe genuine k-means++ initialization and Lloyd centroid updates.',
    architecture: [2, 4], tags: ['Clustering', 'Lloyd algorithm', 'k-means++'], accent: '#67e8b0',
    task: 'Minimize within-cluster squared distance for four synthetic point clouds.',
  },
  {
    id: 'gridworld', title: 'Gridworld Q-learning', category: 'Reinforcement', difficulty: 'Intermediate',
    description: 'Learn a route around walls with tabular temporal-difference learning.',
    architecture: [36, 4], tags: ['Q-learning', 'ε-greedy', 'Tabular'], accent: '#ffb45e',
    task: 'Reach the goal in a 6×6 grid, avoiding walls and a hazard. No neural policy is implied.',
  },
  {
    id: 'bandit', title: 'Contextual bandit', category: 'Reinforcement', difficulty: 'Intermediate',
    description: 'Learn context-dependent action values from noisy observed rewards.',
    architecture: [3, 12, 3], tags: ['Neural value function', 'ε-greedy', 'Bandit'], accent: '#f472b6',
    task: 'Choose among three actions from a three-dimensional context and update only the chosen reward estimate.',
  },
  {
    id: 'custom-data', title: 'Uploaded dataset laboratory', category: 'Classification', difficulty: 'Intermediate',
    description: 'Train a real dense model on numerical CSV features prepared in the Data workbench.',
    architecture: [4, 24, 16, 2], tags: ['CSV import', 'Train-only preprocessing', 'Custom MLP'], accent: '#67e8b0',
    task: 'Prepare training and held-out numerical arrays, select a classification or regression target, and train locally in your browser.',
  },
  {
    id: 'tiny-cnn', title: 'Convolution microscope', category: 'Vision', difficulty: 'Advanced',
    description: 'Train actual shared spatial kernels and inspect convolution/pooling backpropagation.',
    architecture: [64, 72, 18, 12, 4], tags: ['Trainable CNN', 'Analytic convolution gradients', 'Weight sharing'], accent: '#42d8ff',
    task: 'Classify noisy line orientation with 3×3 convolution, average pooling, and a dense readout.',
    outputLabels: ['Vertical', 'Horizontal', 'Diagonal', 'Anti-diagonal'],
  },
  {
    id: 'tiny-transformer', title: 'Causal Transformer microscope', category: 'Language', difficulty: 'Advanced',
    description: 'Train token embeddings, two causal attention heads, layer norms, FFN, and vocabulary readout.',
    architecture: [4, 8, 8, 16, 8, 13], tags: ['Trainable Transformer', 'Causal attention', 'Exact backpropagation'], accent: '#a78bfa',
    task: 'Learn a tiny explicit grammar from four context tokens; all Transformer parameters receive real gradients.',
  },
  {
    id: 'super-mario', title: 'Super Mario reinforcement learning', category: 'Reinforcement', difficulty: 'Advanced',
    description: 'A separate local Python PPO project with emulator setup, checkpoints, and honest evaluation.',
    architecture: [84, 32, 64, 7], tags: ['Local Python', 'PPO', 'NES emulator'], accent: '#ffb45e',
    task: 'Run the project in demos/super-mario on your computer. The NES emulator does not execute on static GitHub Pages.',
  },
];

export function getDemo(id: string): DemoDefinition {
  const demo = DEMOS.find(candidate => candidate.id === id);
  if (!demo) throw new Error(`Unknown demo: ${id}`);
  return demo;
}
