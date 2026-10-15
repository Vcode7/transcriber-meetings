import { describe, it, expect } from 'vitest';
import { agendaReduction, condenseSourceIndex, countWords, formatReduction, pointSources, versionReduction } from './romVersions';

const longRom = {
  agendas: [
    {
      agenda_id: 'A1',
      discussion_points: [
        { id: 's2-a', text: 'Priya asked whether the audit covers Q3.', speakers: ['Priya'] },
        { id: 's2-b', text: 'Meena reported the migration is 80% complete and on track.', speaker: 'Meena' },
      ],
    },
    { agenda_id: 'A2', discussion_points: [{ id: 's2-z', text: 'Ravi will hire two analysts.' }] },
  ],
};

const shortRom = {
  agendas: [
    {
      agenda_id: 'A1',
      condense_sources: [
        { label: 'P1', id: 's2-a', text: 'Priya asked whether the audit covers Q3.' },
        { label: 'P2', id: 's2-b', text: 'Meena reported the migration is 80% complete and on track.' },
      ],
      discussion_points: [{ id: 'A1-SHORT-1', text: 'The audit covers Q3; migration 80% complete.', source_point_ids: ['s2-b', 's2-a'] }],
    },
    // Generated before snapshots: originals come from the long agenda.
    { agenda_id: 'A2', discussion_points: [{ id: 'A2-SHORT-1', text: 'Ravi hires two analysts.', source_point_ids: ['s2-z'] }] },
    { agenda_id: 'A3', skipped: true, discussion_points: [] },
  ],
};

describe('countWords', () => {
  it('counts whitespace-separated words', () => {
    expect(countWords('  The audit   covers Q3. ')).toBe(4);
    expect(countWords('')).toBe(0);
    expect(countWords(undefined)).toBe(0);
  });
});

describe('agendaReduction / versionReduction', () => {
  it('compares the condensed points with the long points they were written from', () => {
    const a1 = agendaReduction(shortRom.agendas[0]);
    expect(a1).toMatchObject({ originalWords: 17, originalPoints: 2, generatedWords: 7, generatedPoints: 1 });
    expect(a1.reductionPct).toBeCloseTo((10 / 17) * 100);

    const total = versionReduction(shortRom, longRom);
    expect(total).toMatchObject({ originalWords: 22, originalPoints: 3, generatedWords: 11, generatedPoints: 2 });
    expect(total.reductionPct).toBeCloseTo(50);
    expect(Object.keys(total.agendas)).toEqual(['A1', 'A2']);
    expect(total.agendas.A2.originalWords).toBe(5);
  });

  it('formats reductions and growth', () => {
    expect(formatReduction(58.06)).toBe('58.1% fewer words');
    expect(formatReduction(58.06, true)).toBe('−58%');
    expect(formatReduction(-12.5)).toBe('12.5% more words');
    expect(formatReduction(null)).toBe('—');
  });
});

describe('condenseSourceIndex / pointSources', () => {
  it('maps every cited ID, including merged points, to its original point', () => {
    const index = condenseSourceIndex(shortRom, longRom);
    expect(pointSources(shortRom.agendas[0].discussion_points[0], index).map(s => s.label)).toEqual(['P2', 'P1']);
    const fallback = pointSources(shortRom.agendas[1].discussion_points[0], index);
    expect(fallback).toEqual([expect.objectContaining({ id: 's2-z', label: 'P1', text: 'Ravi will hire two analysts.' })]);
    expect(index.get('s2-b')?.label).toBe('P2');
    expect(pointSources({ source_point_ids: ['gone'] }, index)).toEqual([{ id: 'gone', text: '', missing: true }]);
  });
});
