# Data Format Specification

This document describes the data format used for training and inference in DesignAsCode.

## Overview

All training data is stored in **JSONL** (JSON Lines) format, where each line is a complete JSON object representing one training example.

## File Format

```
data/
├── dataset.jsonl              # Main training dataset
├── g_dataset.jsonl            # Extended dataset variant
└── extra_dataset.jsonl # Additional training examples
```

## JSON Schema

Each line in the JSONL file follows this structure:

```json
{
  "id": "string",
  "prompt": "string",
  "layout_thought": "string",
  "grouping": "string",
  "image_generator": "string",
  "generate_text": "string"
}
```

## Field Descriptions

### `id` (string, required)
- Unique identifier for each training example
- Format: alphanumeric string, typically 24 characters (MongoDB ObjectId format)
- Example: `"5888a29195a7a863ddcc1c8c"`
- Used for tracking and organization

### `prompt` (string, required)
- Natural language design request from user
- Describes what design to create
- Examples:
  - `"A modern promotional design for a hair salon discount."`
  - `"A winter sale flyer with a modern and minimalist design."`
  - `"A vibrant advertisement for a makeup course using bright colors."`
- Length: typically 50-200 characters

### `layout_thought` (string, required)
- Detailed design planning in XML format
- Must be enclosed in `<layout_thought>...</layout_thought>` tags
- Contains reasoning about:
  - Overall layout structure
  - Layer descriptions and positioning
  - Element hierarchy and spacing
  - Color schemes and styling
- Written in natural language, detailed and structured
- Example structure:
  ```xml
  <layout_thought>
  This layout should establish a clean and professional tone...
  
  The first layer should be the background (layer 0), which is...
  
  The second layer should add a solid color overlay (layer 1)...
  
  [More layer descriptions...]
  </layout_thought>
  ```
- Length: typically 2000-5000 characters

### `grouping` (string, required)
- Layer grouping with thematic labels in XML format
- Must be enclosed in `<grouping>...</grouping>` tags
- Contains JSON array of group objects, each with:
  - `group_id`: unique identifier string (e.g., "G1", "G2")
  - `children`: list of layer IDs belonging to this group
  - `theme`: short description (2–6 words) of the group's purpose
- Example:
  ```xml
  <grouping>
  [
    {"group_id": "G1", "children": [1], "theme": "background color"},
    {"group_id": "G2", "children": [0], "theme": "background texture"},
    {"group_id": "G3", "children": [2], "theme": "decorative shape accent"},
    {"group_id": "G4", "children": [4], "theme": "main title text"},
    {"group_id": "G5", "children": [3], "theme": "subtitle text"}
  ]
  </grouping>
  ```

### `image_generator` (string, required)
- Image generation specifications in XML format
- Must be enclosed in `<image_generator>...</image_generator>` tags
- Contains JSON array of layer specifications
- Each layer has:
  - `layer_id`: integer, unique within this design
  - `layer_prompt`: string, detailed description for image generation
- Example:
  ```xml
  <image_generator>
  [
    {
      "layer_id": 0,
      "layer_prompt": "Abstract low-poly geometric background, dimensions approximately 849 × 313 px, filled edge-to-edge with numerous interlocking triangular facets..."
    },
    {
      "layer_id": 1,
      "layer_prompt": "Solid flat-color rectangular shape, size 851 × 315 px, filled uniformly with vibrant medium green..."
    }
  ]
  </image_generator>
  ```
- Each `layer_prompt` should be:
  - Detailed and specific about dimensions, colors, content
  - Actionable for image generation models
  - Include technical specifications (size, color codes, opacity)

### `generate_text` (string, required)
- Text element specifications in XML format
- Must be enclosed in `<generate_text>...</generate_text>` tags
- Contains JSON array of text element definitions
- Each text element has:
  - `layer_id`: integer (typically >= 3, after background/image layers)
  - `type`: typically `"TextElement"`
  - `width`: float, element width in pixels
  - `height`: float, element height in pixels
  - `opacity`: float, 0.0-1.0
  - `text`: string, actual text content
  - `font`: string, font family name
  - `font_size`: float, font size in pixels
  - `text_align`: string, "left", "center", or "right"
  - `angle`: float, rotation angle in degrees (usually 0)
  - `capitalize`: boolean, whether to capitalize text
  - `line_height`: float, line height multiplier
  - `letter_spacing`: float, letter spacing adjustment

- Example:
  ```xml
  <generate_text>
  [
    {
      "layer_id": 3,
      "type": "TextElement",
      "width": 519.65,
      "height": 56.73,
      "opacity": 1.0,
      "text": "Visit, choose from the variety of social projects\nand donate as much as you wish.",
      "font": "Montserrat",
      "font_size": 20.21,
      "text_align": "right",
      "angle": 0.0,
      "capitalize": false,
      "line_height": 1.42,
      "letter_spacing": 1.14
    },
    {
      "layer_id": 4,
      "type": "TextElement",
      "width": 487.0,
      "height": 101.65,
      "opacity": 1.0,
      "text": "Civic Crowdfunding\nPlatform",
      "font": "Montserrat",
      "font_size": 43.0,
      "text_align": "left",
      "angle": 0.0,
      "capitalize": false,
      "line_height": 1.2,
      "letter_spacing": 1.0
    }
  ]
  </generate_text>
  ```



## Layer IDs

Layers are numbered starting from 0, typically organized as:
- **Layer 0-2**: Background and image layers (generated)
- **Layer 3+**: Text elements (generated)
- May include intermediate layers for masks, overlays, etc.

## Complete Example

```json
{
  "id": "5888a29195a7a863ddcc1c8c",
  "prompt": "A promotional banner for a civic crowdfunding platform with a modern geometric background.",
  "layout_thought": "<layout_thought>\nThis layout should establish a clean and professional tone suitable for communicating a clear and uplifting message about a civic crowdfunding platform. The focal point should balance a visually appealing background, emphasize the platform's name and purpose, and maintain alignment for easy readability.\n\nThe first layer should be the background (layer 0), which is a geometric pattern. This layer should occupy the entire canvas to serve as a dynamic yet subdued base that provides texture without overwhelming the design.\n\nThe second layer should add a solid green overlay color (layer 1) with full opacity. This vibrant green background communicates vitality, growth, and positivity, aligning with the idea of civic projects.\n\nNext, the vertical white stripe (layer 2) should be introduced slightly off-center to the left. This vertical block should act as a visual anchor, guiding the viewer's eye across the design.\n\nThe primary title text (layer 4), 'Civic Crowdfunding Platform,' should be centered horizontally and placed in the upper third of the layout.\n\nLayer 5 (subheading) should follow as the secondary message: 'Visit, choose from the variety of social projects and donate as much as you wish.'\n\nThe layout should be concise, maintaining consistency in spacing, alignment, and contrast along with a modern design approach.\n</layout_thought>",
  "image_generator": "<image_generator>\n[{\"layer_id\": 0, \"layer_prompt\": \"Abstract low-poly geometric background, dimensions approximately 849 × 313 px, filled edge-to-edge with numerous interlocking triangular facets, vivid turquoise-to-teal color palette with subtle gradients and shadows that create a crisp 3-D faceted effect, smooth seamless coverage, no text or additional elements, full opacity\"}, {\"layer_id\": 1, \"layer_prompt\": \"Solid flat-color rectangular shape, size 851 × 315 px, filled uniformly with a vibrant medium green (RGB 64,184,112 / #40B870) at 100% opacity; all areas outside this rectangle are fully transparent\"}, {\"layer_id\": 2, \"layer_prompt\": \"A tall, narrow shape measuring 50 × 289 px, filled with solid white (#ffffff, 100% opacity); the lower edge is cut at a diagonal that slopes downward toward the bottom-right corner, so the triangular area beneath this diagonal is fully transparent\"}]\n</image_generator>",
  "grouping": "<grouping>\n[{\"group_id\": \"G1\", \"children\": [1], \"theme\": \"background color\"}, {\"group_id\": \"G2\", \"children\": [0], \"theme\": \"background texture\"}, {\"group_id\": \"G3\", \"children\": [2], \"theme\": \"decorative shape accent\"}, {\"group_id\": \"G4\", \"children\": [4], \"theme\": \"main title text\"}, {\"group_id\": \"G5\", \"children\": [3], \"theme\": \"subtitle text\"}]\n</grouping>",
  "generate_text": "<generate_text>\n[{\"layer_id\": 3, \"type\": \"TextElement\", \"width\": 519.65, \"height\": 56.73, \"opacity\": 1.0, \"text\": \"Visit, choose from the variety of social projects\\nand donate as much as you wish.\", \"font\": \"Montserrat\", \"font_size\": 20.21, \"text_align\": \"right\", \"angle\": 0.0, \"capitalize\": false, \"line_height\": 1.42, \"letter_spacing\": 1.14}, {\"layer_id\": 4, \"type\": \"TextElement\", \"width\": 487.0, \"height\": 101.65, \"opacity\": 1.0, \"text\": \"Civic Crowdfunding\\nPlatform\", \"font\": \"Montserrat\", \"font_size\": 43.0, \"text_align\": \"left\", \"angle\": 0.0, \"capitalize\": false, \"line_height\": 1.2, \"letter_spacing\": 1.0}]\n</generate_text>"
}
```

## Data Quality Guidelines

When creating training data:

1. **Prompts should be:**
   - Clear and descriptive
   - Specific about design style and purpose
   - Varied in length and complexity

2. **Layout thoughts should:**
   - Explain reasoning about design decisions
   - Describe all layers systematically
   - Include specific measurements and colors when applicable
   - Be comprehensive but natural-sounding

3. **Image prompts should be:**
   - Detailed and technical
   - Include dimensions and specifications
   - Describe colors using names, hex codes, or RGB values
   - Be actionable for image generation models

4. **Text elements should:**
   - Have realistic positions and sizes
   - Use actual font families
   - Include appropriate styling (alignment, spacing)
   - Have meaningful text content

## Data Statistics

### Dataset Breakdown

| Dataset | Size | Records | Avg Text Length |
|---------|------|---------|-----------------|
| dataset.jsonl | 147MB | ~30k | 3000 chars |
| g_dataset.jsonl | 140MB | ~28k | 3200 chars |
| extra_dataset.jsonl | 83MB | ~17k | 2800 chars |

## Loading and Processing

### Python Example

```python
import json

def load_jsonl(filepath):
    """Load JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data, filepath):
    """Save data to JSONL file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Load data
data = load_jsonl('data/dataset.jsonl')
print(f"Loaded {len(data)} examples")

# Process example
example = data[0]
print(f"ID: {example['id']}")
print(f"Prompt: {example['prompt']}")
```

## Validation

Before training, validate your data:

```python
def validate_example(example):
    """Validate a single training example"""
    required_fields = ['id', 'prompt', 'layout_thought', 'grouping', 'image_generator', 'generate_text']
    
    # Check required fields
    for field in required_fields:
        if field not in example:
            return False, f"Missing field: {field}"
    
    # Check XML tags
    if '<layout_thought>' not in example['layout_thought']:
        return False, "Invalid layout_thought format"
    if '<grouping>' not in example['grouping']:
        return False, "Invalid grouping format"
    if '<image_generator>' not in example['image_generator']:
        return False, "Invalid image_generator format"
    if '<generate_text>' not in example['generate_text']:
        return False, "Invalid generate_text format"
    
    return True, "Valid"

# Validate all examples
for i, example in enumerate(data):
    valid, msg = validate_example(example)
    if not valid:
        print(f"Example {i} ({example.get('id', 'unknown')}): {msg}")
```

---

For more information, see [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md).
