import type { Linter } from "eslint";
import { defineConfig } from "eslint/config";

import { baseConfig, restrictEnvAccess } from "@governance/eslint-config/base";
import { reactConfig } from "@governance/eslint-config/react";

// Legacy portal (inherited from Governance-Agent). These strict type-aware rules
// were never enforced upstream; they stay visible as warnings while the portal
// is cleaned up, instead of blocking CI for code the Indexa engine does not touch.
const legacyPortalRules: Linter.Config = {
  files: ["src/**/*.ts", "src/**/*.tsx"],
  rules: {
    // TanStack Router redirects are thrown as plain objects by design.
    "@typescript-eslint/only-throw-error": "off",
    "@typescript-eslint/prefer-nullish-coalescing": "warn",
    "@typescript-eslint/no-unnecessary-condition": "warn",
    "@typescript-eslint/no-non-null-assertion": "warn",
    "@typescript-eslint/no-unsafe-assignment": "warn",
    "@typescript-eslint/prefer-promise-reject-errors": "warn",
    "@typescript-eslint/require-await": "warn",
    "@typescript-eslint/no-unnecessary-type-assertion": "warn",
  },
};

export default defineConfig(
  {
    ignores: [".nitro/**", ".output/**", ".tanstack/**"],
  },
  baseConfig,
  reactConfig,
  restrictEnvAccess,
  legacyPortalRules,
);
