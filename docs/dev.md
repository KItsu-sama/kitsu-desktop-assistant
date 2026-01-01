# 🔍 Prompt Inspection Commands - Usage Guide

## Overview

The prompt inspection system lets you see exactly what's being sent to your LLM. This is crucial for debugging and understanding how your character model (kitsu:character) is being controlled.

---

## Commands

### 1. `/show_pre_prompt` (or `/show_prompt`, `/last_prompt`)

Shows the last prompt that was sent to the LLM.

**Usage:**

```bash
/show_pre_prompt [format]
```

**Formats:**

- `pretty` (default) - Formatted, human-readable output with analysis
- `raw` - Just the raw prompt text
- `json` - Full data as JSON

**Examples:**

```bash
/show_pre_prompt              # Pretty format
/show_pre_prompt raw          # Just the prompt text
/show_pre_prompt json         # Full JSON data
```

**Output includes:**

- Timestamp
- Model name and type (CHARACTER vs STANDARD)
- Parameters (mood, style, emotion)
- Generation options (temperature, num_predict, etc.)
- Full prompt text

---

### 2. `/prompt_breakdown` (or `/breakdown`)

Analyzes and breaks down the prompt structure, showing each component.

**Usage:**

```bash
/prompt_breakdown
```

**For character models (kitsu:character):**

- Shows the `<kitsu.control>` header with parameters
- Shows memory injection (if any)
- Shows the raw user input

**For standard models:**

- Shows the full natural language prompt
- Explains the structure

---

### 3. `/model_config` (or `/show_model`)

Shows the current model configuration and settings.

**Usage:**

```bash
/model_config
```

**Output includes:**

- Model name and type
- Temperature and streaming settings
- Prompt mode (Control Headers vs Full Prompts)
- LoRA status (if applicable)
- LLM config (retries, fallback, etc.)

---

### 4. `/compare_modes` (or `/compare_prompts`)

Shows how the prompt differs across different mood/style combinations.

**Usage:**

```bash
/compare_modes [test_input]
```

**Examples:**

```bash
/compare_modes                    # Uses default "Hello!"
/compare_modes What's up?         # Custom test input
```

**Output:**

- Shows prompt for all combinations of:
  - Moods: behave, mean, flirty
  - Styles: chaotic, sweet, cold, silent

---

### 5. `/export_prompts` (or `/save_prompts`)

Exports the last prompt data to a file for later analysis.

**Usage:**

```bash
/export_prompts
```

**Output file:** `logs/prompt_history.jsonl`

Each prompt is appended as a JSON line, allowing you to track prompt evolution over time.

---

## Character Model Format

Since you're using `kitsu:character`, your prompts use a **minimal control header** format:

```python
<kitsu.control>
emotion=happy
mood=behave
style=chaotic
length=128
</kitsu.control>

Hello!
```

This is **machine-readable** and contains:

- `emotion` - Current emotional state
- `mood` - behave/mean/flirty
- `style` - chaotic/sweet/cold/silent
- `length` - Token budget hint

### Optional Memory Block

If memory injection is enabled:

```python
<kitsu.control>
emotion=happy
mood=behave
style=chaotic
length=128
</kitsu.control>

<kitsu.memory>[{"role":"user","text":"Hi","emotion":"neutral"}]</kitsu.memory>

Hello!
```

---

## Debugging Workflow

### 1. **Response seems off? Check the prompt:**

```bash
/show_pre_prompt
```

Look for:

- Is the mood/style correct?
- Is the emotion what you expected?
- Are generation options reasonable?

### 2. **Understand structure:**

```bash
/prompt_breakdown
```

Verify:

- Control header is properly formed
- User input is clean
- No unexpected injection

### 3. **Compare across modes:**

```bash
/compare_modes How are you?
```

See how different mood/style combos affect the control header.

### 4. **Check model config:**

```bash
/model_config
```

Verify:

- Using CHARACTER model
- Prompt mode is "Control"
- Temperature is reasonable (0.5-0.9)
- LoRA is loaded if expected

### 5. **Track changes over time:**

```bash
/export_prompts
```

Then analyze `logs/prompt_history.jsonl` for patterns.

---

## Integration with Auto-Train

The prompt inspector works seamlessly with the `/auto_train` system:

1. **Issue detected:** `/show_pre_prompt` reveals a problem
2. **Fix the response:** `/train <corrected_response>`
3. **Auto-training kicks in** (if enabled)
4. **Verify next time:** `/show_pre_prompt` after similar input

---

## Tips

### For Character Models

- Keep control headers **minimal** - no natural language
- Parameters are **deterministic** - same input = same control header
- The model learns behavior from **training data**, not instructions
- Check `emotion`, `mood`, `style` match your expectations

### For Standard Models

- Full prompts include character context, templates, and memory
- Much longer than character model prompts
- More prone to prompt injection issues
- Harder to debug due to complexity

### General

- Use `/show_pre_prompt raw` to copy/paste for external analysis
- Use `/export_prompts` before major changes for comparison
- Compare prompts when responses are inconsistent
- Check temperature/num_predict if output length is wrong

---

## Troubleshooting

### "No prompt data available yet"

- Send at least one message first
- The inspector hooks into LLM after initialization

### Prompt looks wrong

1. Check `/model_config` - verify model type
2. Restart Kitsu if recently switched models
3. Check for stuck override with `/train` commands

### Can't see full prompt

- Use `/show_pre_prompt raw` for untruncated output
- Or use `/show_pre_prompt json` and parse manually

---

## Example Session

```bash
You: Hello Kitsu!
Kitsu: Heyyyy! What's up? 🦊

You: /show_pre_prompt
✅ (shows formatted prompt with control header)

You: /prompt_breakdown
✅ (shows detailed analysis)

You: /compare_modes Hi there
✅ (shows all mode/style combinations)

You: That behave/chaotic combo looks best
You: /mood behave
You: /style chaotic

You: /model_config
✅ (confirms settings)
```

---

## Advanced: Analyzing Exported Data

The exported `logs/prompt_history.jsonl` is JSON Lines format:

```python
import json

with open('logs/prompt_history.jsonl') as f:
    for line in f:
        data = json.loads(line)
        print(f"{data['timestamp']}: {data['mood']}/{data['style']}")
        print(f"Length: {data['prompt_length']}")
        print(f"Temperature: {data['options']['temperature']}")
        print()
```

This lets you track patterns like:

- Average prompt length per mode
- Temperature trends
- Emotion distribution
- Style usage over time

---

## Questions?

Type `/help` for the full command list, or check the dev console documentation.

Happy debugging!.
