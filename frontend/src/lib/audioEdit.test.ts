import { describe, it, expect } from 'vitest';
import {
  appendAudioEdit,
  buildEditSelection,
  clampHandle,
  fmtTimePrecise,
  initialTrimState,
  keptDuration,
  moveHandle,
  parseTime,
  toggleCut,
  withDuration,
  type TrimState,
} from './audioEdit';

const formEntries = (form: FormData) => Object.fromEntries(Array.from(form.entries()));

describe('parseTime / fmtTimePrecise', () => {
  it('parses seconds, m:ss and h:mm:ss with decimals', () => {
    expect(parseTime('83.4')).toBe(83.4);
    expect(parseTime('1:23.4')).toBeCloseTo(83.4);
    expect(parseTime(' 1:02:03.5 ')).toBeCloseTo(3723.5);
  });

  it('rejects partial or invalid input instead of guessing', () => {
    for (const bad of ['', '1:', ':30', 'abc', '-5', '1:2:3:4', '1::2']) {
      expect(parseTime(bad)).toBeNull();
    }
  });

  it('formats to a tenth of a second and round-trips', () => {
    expect(fmtTimePrecise(83.44)).toBe('1:23.4');
    expect(fmtTimePrecise(0.5)).toBe('0:00.5');
    expect(fmtTimePrecise(3723.55)).toBe('1:02:03.6');
    for (const t of [0, 9.9, 59.95, 600.3, 3599.9]) {
      expect(parseTime(fmtTimePrecise(t))).toBeCloseTo(Math.round(t * 10) / 10);
    }
  });
});

describe('buildEditSelection', () => {
  const full = initialTrimState(600);

  it('is null when nothing was changed', () => {
    expect(buildEditSelection(full)).toBeNull();
    expect(buildEditSelection(initialTrimState(0))).toBeNull();
  });

  it('never sends an end point the user did not move', () => {
    // Regression: moving only the start used to also send end = browser duration,
    // clipping the tail whenever the browser under-estimated the length.
    const sel = buildEditSelection(moveHandle(full, 'start', 42));
    expect(sel).toEqual({ trimStartSec: 42, sourceDurationSec: 600 });
  });

  it('a cut alone sends no trim boundaries', () => {
    const s = toggleCut(full);
    expect(buildEditSelection(s)).toEqual({ cutStartSec: 210, cutEndSec: 390, sourceDurationSec: 600 });
  });

  it('dragging a handle back to the edge counts as untouched', () => {
    const s = moveHandle(moveHandle(full, 'end', 300), 'end', 600);
    expect(buildEditSelection(s)).toBeNull();
  });

  it('only the part of the cut inside the trim window is sent', () => {
    const s: TrimState = { duration: 600, trimStart: 100, trimEnd: 500, cutEnabled: true, cutStart: 50, cutEnd: 200 };
    expect(buildEditSelection(s)).toEqual({
      trimStartSec: 100, trimEndSec: 500, cutStartSec: 100, cutEndSec: 200, sourceDurationSec: 600,
    });
    expect(keptDuration(s)).toBe(300);
  });
});

describe('handle constraints', () => {
  const s: TrimState = { duration: 100, trimStart: 10, trimEnd: 90, cutEnabled: true, cutStart: 40, cutEnd: 60 };

  it('keeps every handle in order with minimum gaps', () => {
    expect(clampHandle('start', 55, s)).toBe(39.9);
    expect(clampHandle('start', -3, s)).toBe(0);
    expect(clampHandle('end', 50, s)).toBe(60.1);
    expect(clampHandle('end', 150, s)).toBe(100);
    expect(clampHandle('cut-start', 5, s)).toBe(10.1);
    expect(clampHandle('cut-start', 70, s)).toBe(59.7);
    expect(clampHandle('cut-end', 30, s)).toBe(40.3);
    expect(clampHandle('cut-end', 95, s)).toBe(89.9);
  });

  it('snaps to tenths so the value sent is the value shown', () => {
    expect(clampHandle('start', 12.345, s)).toBe(12.3);
  });

  it('toggleCut places a fresh cut inside the current trim window', () => {
    const t = toggleCut({ ...s, cutEnabled: false, cutStart: 0, cutEnd: 5 });
    expect(t).toMatchObject({ cutEnabled: true, cutStart: 34, cutEnd: 66 });
    expect(toggleCut(t).cutEnabled).toBe(false);
  });
});

describe('withDuration', () => {
  it('an untouched end follows a refined duration estimate', () => {
    expect(withDuration(initialTrimState(298.5), 300).trimEnd).toBe(300);
    const moved = moveHandle(initialTrimState(298.5), 'end', 200);
    expect(withDuration(moved, 300).trimEnd).toBe(200);
  });

  it('initialises from an unknown duration', () => {
    expect(withDuration(initialTrimState(0), 120)).toEqual(initialTrimState(120));
  });
});

describe('appendAudioEdit', () => {
  it('writes only the fields that are set, with fixed precision', () => {
    const form = new FormData();
    appendAudioEdit(form, { trimStartSec: 1.25, cutStartSec: 10, cutEndSec: 20.5, sourceDurationSec: 61.0333 });
    expect(formEntries(form)).toEqual({
      trim_start_sec: '1.250',
      cut_start_sec: '10.000',
      cut_end_sec: '20.500',
      source_duration_sec: '61.033',
    });
  });

  it('is a no-op for "whole file"', () => {
    const form = new FormData();
    appendAudioEdit(form, null);
    appendAudioEdit(form, { sourceDurationSec: 60 });
    appendAudioEdit(form, { cutStartSec: 5, cutEndSec: 5, sourceDurationSec: 60 });
    expect(Array.from(form.keys())).toEqual([]);
  });
});
