/** Portable deterministic random stream, including a serializable normal cache. */
export class Random {
  private state: number;
  private spare: number | null = null;

  constructor(seed = 42) {
    if (!Number.isFinite(seed)) throw new Error('Seed must be finite.');
    this.state = seed >>> 0;
  }

  uniform(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let value = this.state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  }

  normal(mean = 0, deviation = 1): number {
    if (this.spare !== null) {
      const value = this.spare;
      this.spare = null;
      return mean + deviation * value;
    }
    const radius = Math.sqrt(-2 * Math.log(Math.max(Number.MIN_VALUE, this.uniform())));
    const angle = 2 * Math.PI * this.uniform();
    this.spare = radius * Math.sin(angle);
    return mean + deviation * radius * Math.cos(angle);
  }

  integer(limit: number): number {
    if (!Number.isInteger(limit) || limit < 1) throw new Error('Random limit must be a positive integer.');
    return Math.floor(this.uniform() * limit);
  }

  choice<T>(values: readonly T[]): T {
    return values[this.integer(values.length)];
  }

  shuffle<T>(values: T[]): T[] {
    for (let i = values.length - 1; i > 0; i--) {
      const j = this.integer(i + 1);
      [values[i], values[j]] = [values[j], values[i]];
    }
    return values;
  }

  exportState(): { state: number; spare: number | null } {
    return { state: this.state, spare: this.spare };
  }

  importState(value: { state: number; spare: number | null }): void {
    if (!Number.isInteger(value.state) || value.state < 0 || value.state > 0xffffffff ||
      (value.spare !== null && !Number.isFinite(value.spare))) {
      throw new Error('Invalid random-stream checkpoint.');
    }
    this.state = value.state;
    this.spare = value.spare;
  }
}
