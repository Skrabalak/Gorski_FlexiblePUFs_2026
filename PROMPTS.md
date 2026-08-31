# AI Prompts

## Documentation Generation

```plaintext
You are assisting me in generating documentation for a codebase.

## Context
- I will provide:
  1. A README describing the purpose of the project
  2. Individual code files (in full, one or more at a time)

- Your task is to generate structured documentation based ONLY on the provided files and the README.

## Rules
1. Maintain a file called `workingmemory.md`. After each step, append what you did and why you did it. This is your scratchpad for reasoning and progress tracking.
2. If you are unsure of an action or fact, DO NOT GUESS. Instead, explicitly ask me for clarification or mark it with `TODO: confirm`.
3. Be explicit and consistent in terminology.
4. Keep documentation in **Markdown** format.
5. Provide both:
   - **File-level summaries**: What each file is for
   - **Function/class documentation**: Name, purpose, parameters, return values
   - **Interactions**: How files, functions, and classes call or depend on one another
   - **Examples**: Usage examples if the context allows

## Workflow
- Step 1: Read the README and summarize the overall project purpose.
- Step 2: For each provided code file:
  - Summarize its role in the project.
  - List all functions and classes with signatures.
  - Provide plain-language explanations of each.
  - Note any calls to functions or classes defined elsewhere (but don’t speculate—just mark them for cross-reference).
- Step 3: Explain the relationships between files (who imports or calls whom).
- Step 4: Compile documentation into a coherent Markdown doc (e.g., `docs.md`).
- Step 5: Update `workingmemory.md` with what was done, why, and any open questions.

## Output Format
- Documentation: Structured Markdown
- Working memory: Append notes to `workingmemory.md` each step

## First Task
I will now provide the README file. Begin with Step 1: Summarize the overall project purpose. Then wait for further input.
```

## tracking down where the memory leak in get_percentages was

My computer is running out of memory while running my program and I have tracked it down to the get_percentages function. Please analyze where any memory leak may be and please analyze how I could restructure this function to be SIGNIFICANTLY more memory efficient and memory conscious.

## create implementation plan for how to fix the memory issues
```
- I will provide:
  1. A README describing the purpose of the project
  2. Individual code files (in full, one or more at a time)

- Your task is to generate structured documentation based ONLY on the provided files and the README.

## Rules
1. Maintain a file called `workingmemory.md`. After each step, append what you did and why you did it. This is your scratchpad for reasoning and progress tracking.
2. If you are unsure of an action or fact, DO NOT GUESS. Instead, explicitly ask me for clarification or mark it with `TODO: confirm`.
3. Be explicit and consistent in terminology.
4. Keep documentation in **Markdown** format.
5. Provide both:
   - **File-level summaries**: What each file is for
   - **Function/class documentation**: Name, purpose, parameters, return values
   - **Interactions**: How files, functions, and classes call or depend on one another
   - **Examples**: Usage examples if the context allows

## Workflow
- Step 1: Read the README and summarize the overall project purpose.
- Step 2: For each provided code file:
  - Summarize its role in the project.
  - List all functions and classes with signatures.
  - Provide plain-language explanations of each.
  - Note any calls to functions or classes defined elsewhere (but don’t speculate—just mark them for cross-reference).
- Step 3: Explain the relationships between files (who imports or calls whom).
- Step 4: Compile documentation into a coherent Markdown doc (e.g., `docs.md`).
- Step 5: Update `workingmemory.md` with what was done, why, and any open questions.

## Output Format
- Documentation: Structured Markdown
- Working memory: Append notes to `workingmemory.md` each step
```

