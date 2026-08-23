/**
 * Chapter 3.2 — Layers & Forward Pass.
 *
 * Trains the real TinyNet on a real toy dataset. The decision boundary the UI
 * draws is sampled from actual forward passes, and accuracy is measured on a
 * held-out split the network never trained on.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import { TinyNet, type Activation, type TrainingSample } from '@/models/tinyNetTrainer';
import { scoreLevel } from './scoringEngine';
import { generateToyDataset, splitDataset, type ToyDatasetKind, type ToyPoint } from './toyDatasets';
import { clamp } from './shared';

export interface NetworkBoundaryConfig {
  dataset: ToyDatasetKind;
  sampleCount: number;
  noise: number;
  seed: number;
  architecture: number[];
  architectureRange?: { maxHidden: number; maxUnits: number };
  activation: Activation;
  activationOptions?: Activation[];
  learningRate: number;
  learningRateRange?: [number, number];
  epochs: number;
  maxEpochs: number;
  trainSplit: number;
  parameterBudget?: number;
  showHiddenRepresentations?: boolean;
}

export interface EpochRecord {
  epoch: number;
  trainLoss: number;
  validationLoss: number;
  validationAccuracy: number;
}

export interface NetworkBoundaryState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  config: NetworkBoundaryConfig;
  architecture: number[];
  activation: Activation;
  learningRate: number;
  epochsTrained: number;
  net: TinyNet;
  trainSet: ToyPoint[];
  validationSet: ToyPoint[];
  history: EpochRecord[];
}

export type NetworkBoundaryAction =
  | { type: 'SET_HIDDEN_LAYERS'; units: number[] }
  | { type: 'SET_ACTIVATION'; value: Activation }
  | { type: 'SET_LEARNING_RATE'; value: number }
  | { type: 'TRAIN'; epochs: number }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

const BATCH_SIZE = 16;

const asSamples = (points: readonly ToyPoint[]): TrainingSample[] =>
  points.map((p) => ({ x: p.x, label: p.label }));

export function initState(
  config: NetworkBoundaryConfig,
  rules: EngineRules
): NetworkBoundaryState {
  const { train, validation } = splitDataset(
    generateToyDataset(config.dataset, config.sampleCount, config.noise, config.seed),
    config.trainSplit
  );

  return {
    rules,
    status: 'idle',
    actionCount: 0,
    config,
    architecture: [...config.architecture],
    activation: config.activation,
    learningRate: config.learningRate,
    epochsTrained: 0,
    net: new TinyNet({
      architecture: config.architecture,
      activation: config.activation,
      seed: config.seed,
    }),
    trainSet: train,
    validationSet: validation,
    history: [],
  };
}

/** Rebuilds the network from scratch — any architecture change discards training. */
function rebuild(state: NetworkBoundaryState, architecture: number[], activation: Activation) {
  return {
    ...state,
    architecture,
    activation,
    epochsTrained: 0,
    history: [],
    net: new TinyNet({ architecture, activation, seed: state.config.seed }),
  };
}

export function applyAction(
  state: NetworkBoundaryState,
  action: NetworkBoundaryAction
): NetworkBoundaryState {
  switch (action.type) {
    case 'SET_HIDDEN_LAYERS': {
      const limits = state.config.architectureRange;
      if (!limits) return state;
      const units = action.units
        .slice(0, limits.maxHidden)
        .map((u) => Math.round(clamp(u, 1, limits.maxUnits)))
        .filter((u) => Number.isFinite(u));

      const inputs = state.config.architecture[0]!;
      return {
        ...rebuild(state, [inputs, ...units, 1], state.activation),
        status: 'active',
        actionCount: state.actionCount + 1,
      };
    }

    case 'SET_ACTIVATION': {
      const options = state.config.activationOptions;
      if (options && !options.includes(action.value)) return state;
      return {
        ...rebuild(state, state.architecture, action.value),
        status: 'active',
        actionCount: state.actionCount + 1,
      };
    }

    case 'SET_LEARNING_RATE': {
      if (!Number.isFinite(action.value)) return state;
      const [min, max] = state.config.learningRateRange ?? [0, Infinity];
      return {
        ...state,
        learningRate: clamp(action.value, min, max),
        actionCount: state.actionCount + 1,
      };
    }

    case 'TRAIN': {
      const remaining = state.config.maxEpochs - state.epochsTrained;
      const epochs = Math.max(0, Math.min(action.epochs, remaining));
      if (epochs === 0) return state;

      // Clone before training so the previous state stays untouched.
      const net = state.net.clone();
      const trainSamples = asSamples(state.trainSet);
      const validationSamples = asSamples(state.validationSet);
      const history = [...state.history];

      for (let i = 0; i < epochs; i++) {
        const epoch = state.epochsTrained + i;
        const trainLoss = net.trainEpoch(trainSamples, {
          learningRate: state.learningRate,
          batchSize: BATCH_SIZE,
          epoch,
        });
        history.push({
          epoch: epoch + 1,
          trainLoss,
          validationLoss: net.loss(validationSamples),
          validationAccuracy: net.accuracy(validationSamples),
        });
      }

      return {
        ...state,
        net,
        epochsTrained: state.epochsTrained + epochs,
        history,
        status: 'active',
        actionCount: state.actionCount + 1,
      };
    }

    case 'RESET':
      return initState(state.config, state.rules);

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

/* ────────────────────────────────────────────────────────────
   Worked solution
   ──────────────────────────────────────────────────────────── */

/**
 * A configuration that really trains to this level's target, found by training.
 *
 * Nothing here is asserted from theory: each candidate is a full run of the
 * real `TinyNet` on the level's own data for its own epoch budget, scored by
 * its own metric. The search is ordered rather than exhaustive — the level's
 * shipped architecture first, then its other activations, then a handful of
 * alternative shapes — and stops the moment a candidate clears the three-star
 * band, because every extra candidate is a real training run the player is
 * waiting on.
 */
export function solve(state: NetworkBoundaryState): NetworkBoundaryState {
  const topBand = Math.max(...state.rules.starsRules.map((rule) => rule.threshold));
  const comparator = state.rules.passCriteria.comparator;
  const better = (candidate: number, incumbent: number) =>
    comparator === 'gte' ? candidate > incumbent : candidate < incumbent;
  const clears = (value: number) => (comparator === 'gte' ? value >= topBand : value <= topBand);

  let best: NetworkBoundaryState | null = null;
  let bestValue = 0;

  for (const candidate of candidateSettings(state)) {
    let run = applyAction(state, { type: 'RESET' });
    run = applyAction(run, { type: 'SET_HIDDEN_LAYERS', units: candidate.hidden });
    run = applyAction(run, { type: 'SET_ACTIVATION', value: candidate.activation });
    run = applyAction(run, { type: 'SET_LEARNING_RATE', value: candidate.learningRate });
    run = applyAction(run, { type: 'TRAIN', epochs: state.config.maxEpochs });

    const value = evaluate(run).value;
    if (!Number.isFinite(value)) continue;
    if (best === null || better(value, bestValue)) {
      best = run;
      bestValue = value;
    }
    if (clears(bestValue)) break;
  }

  return best ?? state;
}

interface BoundaryCandidate {
  hidden: number[];
  activation: Activation;
  learningRate: number;
}

/** Ordered by how likely each is to be the answer, cheapest-to-justify first. */
function candidateSettings(state: NetworkBoundaryState): BoundaryCandidate[] {
  const limits = state.config.architectureRange;
  const defaultHidden = state.config.architecture.slice(1, -1);
  const maxUnits = limits?.maxUnits ?? Math.max(...defaultHidden, 1);
  const maxHidden = limits?.maxHidden ?? defaultHidden.length;

  const shapes: number[][] = [defaultHidden];
  for (const shape of [[3], [4], [6], [8], [8, 8], [6, 6]]) {
    if (shape.length > maxHidden) continue;
    if (shape.some((units) => units > maxUnits)) continue;
    if (shapes.some((existing) => existing.join(',') === shape.join(','))) continue;
    shapes.push(shape);
  }

  const activations: Activation[] = [state.config.activation];
  for (const option of state.config.activationOptions ?? []) {
    if (!activations.includes(option)) activations.push(option);
  }

  const [rateMin, rateMax] = state.config.learningRateRange ?? [
    state.config.learningRate,
    state.config.learningRate,
  ];
  const rates: number[] = [];
  for (const rate of [state.config.learningRate, 0.1, 0.3, 0.05]) {
    if (rate >= rateMin && rate <= rateMax && !rates.includes(rate)) rates.push(rate);
  }

  // Activation varies fastest. On the spiral level tanh cannot pass at any
  // learning rate, so trying every rate before trying ReLU spends four real
  // training runs — and the whole search — on a setting that was never going
  // to work.
  const candidates: BoundaryCandidate[] = [];
  for (const hidden of shapes) {
    for (const learningRate of rates) {
      for (const activation of activations) {
        candidates.push({ hidden, activation, learningRate });
      }
    }
  }
  return candidates;
}

/** Samples the real decision surface on a grid, for the boundary plot. */
export function boundaryGrid(state: NetworkBoundaryState, resolution = 40): number[][] {
  const grid: number[][] = [];
  for (let row = 0; row < resolution; row++) {
    const y = -1.2 + (2.4 * row) / (resolution - 1);
    const line: number[] = [];
    for (let col = 0; col < resolution; col++) {
      const x = -1.2 + (2.4 * col) / (resolution - 1);
      line.push(state.net.predict([x, y]));
    }
    grid.push(line);
  }
  return grid;
}

/** Hidden-layer activations for one input — how each layer reshapes the space. */
export function hiddenRepresentation(state: NetworkBoundaryState, input: number[]): number[][] {
  const pass = state.net.forward(input);
  return pass.activations.slice(1, -1);
}

export function evaluate(state: NetworkBoundaryState): ScoreResult {
  const accuracy = state.net.accuracy(asSamples(state.validationSet));
  const params = state.net.parameterCount;
  const breakdown = {
    accuracy,
    parameters: params,
    epochs: state.epochsTrained,
    trainAccuracy: state.net.accuracy(asSamples(state.trainSet)),
  };

  if (state.rules.passCriteria.metric === 'efficiencyScore') {
    const budget = state.config.parameterBudget ?? params;
    // Staying inside the budget costs nothing; overspending scales the score
    // down in proportion, so brute force cannot buy three stars.
    const value = accuracy * (budget / Math.max(params, budget));
    return scoreLevel({
      metric: 'efficiencyScore',
      value,
      rules: state.rules,
      breakdown: { ...breakdown, budget },
    });
  }

  return scoreLevel({
    metric: 'decisionBoundaryAccuracy',
    value: accuracy,
    rules: state.rules,
    breakdown,
  });
}
