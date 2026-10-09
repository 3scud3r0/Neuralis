import { defineConfig } from 'vitest/config';
export default defineConfig({ test: {
  environment: 'node',
  include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
  pool: 'forks',
  maxWorkers: 2,
  testTimeout: 20000,
} });
