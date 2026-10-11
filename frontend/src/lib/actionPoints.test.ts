import { describe, it, expect } from 'vitest';
import { agendaShowsActionPoints, collectAgendaActionPoints } from './actionPoints';

describe('agendaShowsActionPoints', () => {
  it('uses the per-agenda option and falls back to the global flag for older Final ROMs', () => {
    expect(agendaShowsActionPoints({ include_action_points: true }, false)).toBe(true);
    expect(agendaShowsActionPoints({ include_action_points: false }, true)).toBe(false);
    expect(agendaShowsActionPoints({}, true)).toBe(true);
    expect(agendaShowsActionPoints({ skipped: true, include_action_points: true }, true)).toBe(false);
  });
});

describe('collectAgendaActionPoints', () => {
  it('uses only the agenda-level list for agenda-generated action points', () => {
    const agenda = {
      action_points_mode: 'agenda',
      action_points: [{ task: 'Send the revised budget', assignee: 'Ravi', deadline: 'Not specified' }],
      discussion_points: [{ id: 'p1', action_items: [{ task: 'Old Stage 1 item' }] }],
    };
    expect(collectAgendaActionPoints(agenda).map(ap => ap.task)).toEqual(['Send the revised budget']);
  });

  it('merges point-level action points for older Final ROMs, skipping duplicates and plain strings', () => {
    const agenda = {
      action_points: [{ task: 'Brief the council' }],
      discussion_points: [
        { id: 'p1', action_points: [{ task: 'brief the council' }, { task: 'Draft the report' }] },
        { id: 'p2', action_items: ['legacy string item', { task: 'Book the venue' }] },
      ],
    };
    expect(collectAgendaActionPoints(agenda).map(ap => ap.task)).toEqual(['Brief the council', 'Draft the report', 'Book the venue']);
  });
});
