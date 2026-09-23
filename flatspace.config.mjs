import { createRequire } from 'module';
import { createRequire } from 'module';

var require = createRequire(import.meta.url);
var module = { exports: {} };

const require = createRequire(import.meta.url);

export default {
  outDir: 'dist',
  contentDir: 'site/content',
  theme: './site/theme.mjs',
};
