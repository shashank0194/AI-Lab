/**
 * How a level was cleared, and how that rolls up to a chapter.
 *
 * Kept as a pure module rather than living on the progress store so the map,
 * the HUD and the tests all read the same rule, and so the rule itself can be
 * unit-tested without IndexedDB or a React tree in the way.
 *
 * The rule: a level is `self` the moment it is ever cleared without the worked
 * solution, and that never downgrades. Reading the solution *after* passing a
 * level on your own is exactly the behaviour this feature wants to encourage,
 * so it must not cost the player their mark. Going the other way — clearing a
 * level with autofill and later replaying it unaided — does upgrade, because
 * by then the player really has done it themselves.
 */

export type SolvedWith = 'self' | 'solution';

/** The slice of a stored level result this module needs. */
export interface SolvedRecord {
  solvedWith?: SolvedWith;
}

/** The slice of a stored chapter this module needs. */
export interface ChapterSolveRecord {
  levels: Record<string, SolvedRecord>;
  completedAt?: number | null;
}

/**
 * How to record a level that has just been passed, given whatever was stored
 * for it before. Results saved before this feature existed carry no
 * `solvedWith` at all; those were necessarily earned unaided, so they read as
 * `self`.
 */
export function resolveSolvedWith(
  previous: SolvedRecord | undefined,
  incoming: SolvedWith
): SolvedWith {
  if (previous && (previous.solvedWith ?? 'self') === 'self') return 'self';
  return incoming;
}

/** How many of a chapter's recorded levels were cleared with the solution. */
export function countAssistedLevels(chapter: ChapterSolveRecord | null | undefined): number {
  if (!chapter) return 0;
  return Object.values(chapter.levels).filter((l) => l.solvedWith === 'solution').length;
}

/** A chapter counts as solution-assisted when any one of its levels was. */
export function chapterSolvedWith(chapter: ChapterSolveRecord | null | undefined): SolvedWith {
  return countAssistedLevels(chapter) > 0 ? 'solution' : 'self';
}

/**
 * `chapterId → assisted level count`, for every chapter the player has actually
 * finished. The map only marks completed chapters, so a chapter still in
 * progress is deliberately absent rather than present with a zero.
 */
export function assistedLevelsByChapter(
  chapters: Record<string, ChapterSolveRecord>
): Record<string, number> {
  const record: Record<string, number> = {};
  for (const [chapterId, chapter] of Object.entries(chapters)) {
    if (chapter.completedAt == null) continue;
    record[chapterId] = countAssistedLevels(chapter);
  }
  return record;
}
