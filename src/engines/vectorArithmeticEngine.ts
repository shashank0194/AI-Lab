/**
 * Chapter 1.2 — Vector Arithmetic.
 *
 * The arithmetic is genuinely performed on live embeddings; only the candidate
 * pool is fixed. Ranking is recomputed from cosine similarity every time a term
 * changes, so nothing about the answer is stored in the level JSON beyond which
 * pool to search.
 */
import type { EngineRules, ScoreResult } from '@/types/game';
import type { Embedder } from './deps';
import { scoreLevel } from './scoringEngine';
import { addVectors, clamp, cosineSimilarity, subtractVectors } from './shared';

export type TermOp = 'add' | 'subtract';
export type VectorArithmeticMode = 'fixed-analogy' | 'free-terms' | 'estimate-similarity';

export interface ConfiguredTerm {
  word: string;
  op: TermOp;
}

export interface VectorArithmeticConfig {
  mode: VectorArithmeticMode;
  terms?: ConfiguredTerm[];
  termCount?: number;
  target?: string;
  candidatePool: string[];
  expectedNearest?: string;
  topK: number;
  maxAttempts?: number;
  rounds?: number;
  /**
   * How many words of the model's own vocabulary to index for the live
   * nearest-neighbour readout. Display only — the engine scores against
   * `candidatePool`, so changing this cannot move a threshold.
   */
  neighbourVocabularySize?: number;
}

export interface PreparedVectorData {
  vectors: Record<string, number[]>;
}

export interface ActiveTerm {
  word: string | null;
  op: TermOp;
  vector: number[] | null;
}

export interface RankedCandidate {
  word: string;
  similarity: number;
}

export interface EstimateRound {
  word: string;
  actualSimilarity: number;
  estimate: number | null;
}

export interface VectorArithmeticState {
  rules: EngineRules;
  status: 'idle' | 'active' | 'complete';
  actionCount: number;
  mode: VectorArithmeticMode;
  config: VectorArithmeticConfig;
  vectors: Record<string, number[]>;
  terms: ActiveTerm[];
  resultVector: number[] | null;
  ranked: RankedCandidate[];
  guess: string | null;
  revealed: boolean;
  attempts: number;
  bestTargetRankScore: number;
  rounds: EstimateRound[];
}

export type VectorArithmeticAction =
  | { type: 'SET_TERM'; index: number; word: string; vector: number[]; op?: TermOp }
  | { type: 'SET_OP'; index: number; op: TermOp }
  | { type: 'GUESS'; word: string }
  | { type: 'REVEAL' }
  | { type: 'COMMIT_ATTEMPT' }
  | { type: 'ESTIMATE'; roundIndex: number; value: number }
  | { type: 'RESET' }
  | { type: 'SUBMIT' };

export async function prepare(
  config: VectorArithmeticConfig,
  deps: { embedder: Embedder }
): Promise<PreparedVectorData> {
  const words = new Set<string>(config.candidatePool);
  for (const term of config.terms ?? []) words.add(term.word);
  if (config.target) words.add(config.target);

  const list = [...words];
  if (list.length === 0) return { vectors: {} };

  const embedded = await deps.embedder.embed(list);
  if (embedded.length !== list.length) {
    throw new Error(`Embedder returned ${embedded.length} vectors for ${list.length} words`);
  }

  const vectors: Record<string, number[]> = {};
  list.forEach((word, i) => {
    vectors[word] = embedded[i]!;
  });
  return { vectors };
}

/** Applies the term chain in order: the first term seeds, the rest add or subtract. */
function computeResult(terms: readonly ActiveTerm[]): number[] | null {
  const usable = terms.filter((t) => t.vector !== null);
  if (usable.length === 0 || usable.length !== terms.length) return null;

  let result = [...usable[0]!.vector!];
  if (usable[0]!.op === 'subtract') result = result.map((v) => -v);

  for (let i = 1; i < usable.length; i++) {
    const term = usable[i]!;
    result = term.op === 'add' ? addVectors(result, term.vector!) : subtractVectors(result, term.vector!);
  }
  return result;
}

function rankCandidates(
  result: number[] | null,
  pool: readonly string[],
  vectors: Record<string, number[]>
): RankedCandidate[] {
  if (result === null) return [];
  return pool
    .filter((word) => vectors[word] !== undefined)
    .map((word) => ({ word, similarity: cosineSimilarity(result, vectors[word]!) }))
    .sort((a, b) => b.similarity - a.similarity);
}

export function initState(
  config: VectorArithmeticConfig,
  rules: EngineRules,
  prepared: PreparedVectorData
): VectorArithmeticState {
  const configured = config.terms ?? [];
  const termCount = config.termCount ?? configured.length;

  const terms: ActiveTerm[] = Array.from({ length: termCount }, (_, i) => {
    const term = configured[i];
    return term
      ? { word: term.word, op: term.op, vector: prepared.vectors[term.word] ?? null }
      : { word: null, op: 'add' as TermOp, vector: null };
  });

  const resultVector = computeResult(terms);
  const ranked = rankCandidates(resultVector, config.candidatePool, prepared.vectors);

  // Estimate rounds walk the pool in order, so each round targets a different
  // real candidate and its real similarity to the computed result.
  const rounds: EstimateRound[] =
    config.mode === 'estimate-similarity' && resultVector !== null
      ? config.candidatePool.slice(0, config.rounds ?? 0).map((word) => ({
          word,
          actualSimilarity: prepared.vectors[word]
            ? cosineSimilarity(resultVector, prepared.vectors[word]!)
            : 0,
          estimate: null,
        }))
      : [];

  return {
    rules,
    status: 'idle',
    actionCount: 0,
    mode: config.mode,
    config,
    vectors: prepared.vectors,
    terms,
    resultVector,
    ranked,
    guess: null,
    revealed: false,
    attempts: 0,
    bestTargetRankScore: 0,
    rounds,
  };
}

/** Rank score for the target word: 1 at the top of the pool, 0 at the bottom. */
function targetRankScore(state: VectorArithmeticState): number {
  const target = state.config.target;
  if (!target || state.ranked.length < 2) return 0;
  const index = state.ranked.findIndex((r) => r.word === target);
  if (index === -1) return 0;
  return 1 - index / (state.ranked.length - 1);
}

export function applyAction(
  state: VectorArithmeticState,
  action: VectorArithmeticAction
): VectorArithmeticState {
  const bump = (next: Partial<VectorArithmeticState>): VectorArithmeticState => ({
    ...state,
    ...next,
    status: 'active',
    actionCount: state.actionCount + 1,
  });

  switch (action.type) {
    case 'SET_TERM':
    case 'SET_OP': {
      if (action.index < 0 || action.index >= state.terms.length) return state;
      const terms = [...state.terms];
      const current = terms[action.index]!;
      terms[action.index] =
        action.type === 'SET_TERM'
          ? { word: action.word, vector: action.vector, op: action.op ?? current.op }
          : { ...current, op: action.op };

      const resultVector = computeResult(terms);
      return bump({
        terms,
        resultVector,
        ranked: rankCandidates(resultVector, state.config.candidatePool, state.vectors),
      });
    }

    case 'GUESS': {
      // Once revealed the answer is visible, so late guesses cannot count.
      if (state.revealed) return state;
      if (!state.config.candidatePool.includes(action.word)) return state;
      return bump({ guess: action.word });
    }

    case 'REVEAL':
      return bump({ revealed: true });

    case 'COMMIT_ATTEMPT': {
      const max = state.config.maxAttempts ?? Infinity;
      if (state.attempts >= max) return state;
      if (state.resultVector === null) return state;
      return bump({
        attempts: state.attempts + 1,
        // Keep the best attempt so experimenting cannot lose earned progress.
        bestTargetRankScore: Math.max(state.bestTargetRankScore, targetRankScore(state)),
      });
    }

    case 'ESTIMATE': {
      if (action.roundIndex < 0 || action.roundIndex >= state.rounds.length) return state;
      if (!Number.isFinite(action.value)) return state;
      const rounds = [...state.rounds];
      rounds[action.roundIndex] = {
        ...rounds[action.roundIndex]!,
        estimate: clamp(action.value, -1, 1),
      };
      return bump({ rounds });
    }

    case 'RESET':
      return initState(state.config, state.rules, { vectors: state.vectors });

    case 'SUBMIT':
      return { ...state, status: 'complete', actionCount: state.actionCount + 1 };
  }
}

/* ────────────────────────────────────────────────────────────
   Worked solution
   ──────────────────────────────────────────────────────────── */

/**
 * The answer this level's own live vectors produce.
 *
 * Nothing is read from `expectedNearest` — that field is documentation of what
 * the author measured, not an answer key, and the whole point of the chapter is
 * that the ranking is recomputed from real embeddings every time. The fill
 * takes whatever the model ranks first on this device.
 */
export function solve(state: VectorArithmeticState): VectorArithmeticState {
  switch (state.mode) {
    case 'fixed-analogy':
      return solveAnalogy(state);
    case 'free-terms':
      return solveFreeTerms(state);
    case 'estimate-similarity':
      return solveEstimates(state);
  }
}

/** Guess whatever the arithmetic actually lands nearest to. */
function solveAnalogy(state: VectorArithmeticState): VectorArithmeticState {
  // A guess after the reveal is ignored on purpose, so a player who peeked
  // first needs the board rewound before the fill can land.
  const base = state.revealed ? applyAction(state, { type: 'RESET' }) : state;
  const nearest = base.ranked[0];
  if (!nearest) return base;
  return applyAction(base, { type: 'GUESS', word: nearest.word });
}

/**
 * The best three-term combination that does not simply name the target.
 *
 * Searched rather than hard-coded: every triple of words the level has real
 * vectors for is tried in every add/subtract arrangement, and the winner is the
 * one that ranks the target highest — broken, among the many combinations that
 * put it first, by which one lands *nearest* the target in absolute cosine
 * terms. Excluding the target itself is the one piece of judgement here, and it
 * is the difference between demonstrating the analogy and demonstrating that
 * `paris + x − x` is still paris.
 */
function solveFreeTerms(state: VectorArithmeticState): VectorArithmeticState {
  const winner = bestCombination(state);
  if (!winner) return state;

  // A player who has already spent every attempt cannot commit another one, and
  // the score reads from the best *committed* attempt — so rewind first.
  const exhausted = state.attempts >= (state.config.maxAttempts ?? Infinity);
  let next = exhausted ? applyAction(state, { type: 'RESET' }) : state;

  winner.forEach((term, index) => {
    next = applyAction(next, {
      type: 'SET_TERM',
      index,
      word: term.word,
      vector: state.vectors[term.word]!,
      op: term.op,
    });
  });
  return applyAction(next, { type: 'COMMIT_ATTEMPT' });
}

/**
 * Exhaustive search for the term chain that ranks the target highest.
 *
 * The second sort key is deliberately the target's own similarity rather than
 * its lead over the runner-up. Measured against the real model, ranking by lead
 * picked `france − germany − italy` for level 2: paris does come first, but at a
 * cosine of −0.20, because subtracting two countries pushes everything else
 * further away rather than landing anywhere near paris. Ranking by absolute
 * similarity picks `tokyo − japan + france` at 0.84 — the real capital-of
 * direction, transplanted, which is the thing the level exists to show.
 */
function bestCombination(state: VectorArithmeticState): ConfiguredTerm[] | null {
  const target = state.config.target;
  const termCount = state.terms.length;
  if (!target || termCount === 0) return null;

  const vocabulary = Object.keys(state.vectors).filter((word) => word !== target);
  // Bounded rather than unbounded: enough to be exhaustive over the pools these
  // levels use, and it cannot degrade into a long freeze on a future larger one.
  const budget = 60_000;
  const maxWords = Math.max(2, Math.floor((budget / 2 ** termCount) ** (1 / termCount)));
  const words = vocabulary.slice(0, maxWords);
  if (words.length === 0) return null;

  let bestTerms: ConfiguredTerm[] | null = null;
  let bestRank = -1;
  let bestSimilarity = -Infinity;

  const walk = (chosen: ConfiguredTerm[]) => {
    if (chosen.length === termCount) {
      const active: ActiveTerm[] = chosen.map((t) => ({
        word: t.word,
        op: t.op,
        vector: state.vectors[t.word] ?? null,
      }));
      const result = computeResult(active);
      if (result === null) return;

      const ranked = rankCandidates(result, state.config.candidatePool, state.vectors);
      const index = ranked.findIndex((r) => r.word === target);
      if (index === -1) return;

      const rank = ranked.length < 2 ? 0 : 1 - index / (ranked.length - 1);
      const similarity = ranked[index]?.similarity ?? 0;

      if (rank > bestRank || (rank === bestRank && similarity > bestSimilarity)) {
        bestTerms = [...chosen];
        bestRank = rank;
        bestSimilarity = similarity;
      }
      return;
    }
    for (const word of words) {
      for (const op of ['add', 'subtract'] as const) walk([...chosen, { word, op }]);
    }
  };
  walk([]);

  return bestTerms;
}

/** Each round's estimate set to the cosine similarity the model really produces. */
function solveEstimates(state: VectorArithmeticState): VectorArithmeticState {
  let next = state;
  state.rounds.forEach((round, roundIndex) => {
    next = applyAction(next, { type: 'ESTIMATE', roundIndex, value: round.actualSimilarity });
  });
  return next;
}

export function evaluate(state: VectorArithmeticState): ScoreResult {
  switch (state.mode) {
    case 'fixed-analogy': {
      const rank = state.guess === null ? -1 : state.ranked.findIndex((r) => r.word === state.guess) + 1;
      // Top-1 is worth 3, second 2, third 1, anything further nothing.
      const value = rank > 0 ? Math.max(0, 4 - rank) : 0;
      return scoreLevel({
        metric: 'analogyAccuracy',
        value,
        rules: state.rules,
        breakdown: {
          guessRank: rank,
          topSimilarity: state.ranked[0]?.similarity ?? 0,
          poolSize: state.ranked.length,
        },
      });
    }

    case 'free-terms':
      return scoreLevel({
        metric: 'targetRankScore',
        value: state.bestTargetRankScore,
        rules: state.rules,
        breakdown: { attempts: state.attempts, currentScore: targetRankScore(state) },
      });

    case 'estimate-similarity': {
      const total = state.rounds.length;
      if (total === 0) {
        return scoreLevel({
          metric: 'cosineSimilarityError',
          value: 2,
          rules: state.rules,
          breakdown: { answered: 0, total: 0 },
        });
      }

      let answered = 0;
      let sum = 0;
      for (const round of state.rounds) {
        if (round.estimate === null) {
          // Maximum possible cosine error, so skipping is never a strategy.
          sum += 2;
          continue;
        }
        answered++;
        sum += Math.abs(round.estimate - round.actualSimilarity);
      }

      return scoreLevel({
        metric: 'cosineSimilarityError',
        value: sum / total,
        rules: state.rules,
        breakdown: { answered, total },
      });
    }
  }
}
