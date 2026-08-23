import { describe, it, expect } from 'vitest';
import {
  assistedLevelsByChapter,
  chapterSolvedWith,
  countAssistedLevels,
  resolveSolvedWith,
} from '@/lib/solutionProgress';

describe('resolveSolvedWith', () => {
  it('records a first unaided pass as self', () => {
    expect(resolveSolvedWith(undefined, 'self')).toBe('self');
  });

  it('records a first autofilled pass as solution', () => {
    expect(resolveSolvedWith(undefined, 'solution')).toBe('solution');
  });

  it('keeps self when the solution is opened after already clearing the level', () => {
    // The case the feature is built around: read the explanation once you have
    // solved it yourself, and keep your mark.
    expect(resolveSolvedWith({ solvedWith: 'self' }, 'solution')).toBe('self');
  });

  it('upgrades to self when an autofilled level is later cleared unaided', () => {
    expect(resolveSolvedWith({ solvedWith: 'solution' }, 'self')).toBe('self');
  });

  it('stays solution while every recorded pass used the solution', () => {
    expect(resolveSolvedWith({ solvedWith: 'solution' }, 'solution')).toBe('solution');
  });

  it('treats a result stored before the feature existed as unaided', () => {
    // Progress persisted by an older build carries no `solvedWith` at all, and
    // there was no solution to use when it was earned.
    expect(resolveSolvedWith({}, 'solution')).toBe('self');
  });
});

describe('countAssistedLevels', () => {
  it('is zero for a chapter with no progress at all', () => {
    expect(countAssistedLevels(null)).toBe(0);
    expect(countAssistedLevels(undefined)).toBe(0);
  });

  it('counts only the levels cleared with the solution', () => {
    const chapter = {
      levels: {
        'l1': { solvedWith: 'self' as const },
        'l2': { solvedWith: 'solution' as const },
        'l3': { solvedWith: 'solution' as const },
        'l4': {},
      },
    };
    expect(countAssistedLevels(chapter)).toBe(2);
  });
});

describe('chapterSolvedWith', () => {
  it('is self when every level was cleared unaided', () => {
    expect(
      chapterSolvedWith({ levels: { l1: { solvedWith: 'self' }, l2: { solvedWith: 'self' } } })
    ).toBe('self');
  });

  it('is solution as soon as one level needed it', () => {
    expect(
      chapterSolvedWith({ levels: { l1: { solvedWith: 'self' }, l2: { solvedWith: 'solution' } } })
    ).toBe('solution');
  });
});

describe('assistedLevelsByChapter', () => {
  it('reports a count per completed chapter, including zero for unaided ones', () => {
    const record = assistedLevelsByChapter({
      '1-1-vectors': {
        completedAt: 1,
        levels: { a: { solvedWith: 'self' }, b: { solvedWith: 'solution' } },
      },
      '1-2-vector-arithmetic': { completedAt: 2, levels: { a: { solvedWith: 'self' } } },
    });
    expect(record).toEqual({ '1-1-vectors': 1, '1-2-vector-arithmetic': 0 });
  });

  it('omits chapters still in progress, which the map does not mark either', () => {
    const record = assistedLevelsByChapter({
      '1-1-vectors': { completedAt: null, levels: { a: { solvedWith: 'solution' } } },
    });
    expect(record).toEqual({});
  });
});
