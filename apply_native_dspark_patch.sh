

#!/usr/bin/env bash
set -euo pipefail

# Apply from the speculators repository root, e.g.:
#   cd /home/.../vLLM_NPU/speculators
#   bash /path/to/native_dspark_dsv4_patch/apply_native_dspark_patch.sh

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(pwd)"

if [ ! -f "scripts/train.py" ] || [ ! -d "src/speculators" ]; then
  echo "ERROR: run this script from the speculators repository root."
  echo "Current directory: $REPO_ROOT"
  exit 1
fi

mkdir -p src/speculators/models/dspark_dsv4_native examples/train
cp -r "$PATCH_DIR/src/speculators/models/dspark_dsv4_native"/* src/speculators/models/dspark_dsv4_native/
cp "$PATCH_DIR/examples/train/dspark_dsv4_native_284b_multinode.sh" examples/train/dspark_dsv4_native_284b_multinode.sh
chmod +x examples/train/dspark_dsv4_native_284b_multinode.sh

# Ensure auto-discovery by explicit import as well.
python - <<'PY'
from pathlib import Path
p = Path('src/speculators/models/__init__.py')
if p.exists():
    text = p.read_text()
else:
    text = ''
line = 'import speculators.models.dspark_dsv4_native  # noqa: F401\n'
if 'speculators.models.dspark_dsv4_native' not in text:
    if text and not text.endswith('\n'):
        text += '\n'
    text += line
    p.write_text(text)
PY

python - <<'PY'
from pathlib import Path

p = Path('scripts/train.py')
text = p.read_text()

# 1) Special-case native DSpark before the generic --from-pretrained path.
needle = '    if args.from_pretrained:\n'
special = '''    if args.speculator_type == "dspark_dsv4_native":
        # Native DeepSeek-V4 DSpark uses the verifier's raw DeepSeek-V4 config
        # and builds official-style mtp.* blocks directly.  Do not synthesize a
        # Qwen/Llama draft decoder for this speculator type.
        transformer_layer_config = get_verifier_config(args.verifier_name_or_path)
        if hasattr(transformer_layer_config, "text_config"):
            transformer_layer_config = transformer_layer_config.text_config
        args.draft_vocab_size = draft_vocab_size
        model = model_class.from_training_args(
            verifier_config=transformer_layer_config,
            t2d=t2d,
            d2t=d2t,
            **vars(args),
        )
        if args.from_pretrained:
            if not hasattr(model, "load_mtp_weights_from_hf"):
                raise TypeError(
                    "dspark_dsv4_native model is missing load_mtp_weights_from_hf()"
                )
            logger.info(
                "Loading native DeepSeek-V4 DSpark mtp.* weights from '%s'",
                args.from_pretrained,
            )
            model.load_mtp_weights_from_hf(args.from_pretrained, strict=False)
        return model

'''
if 'args.speculator_type == "dspark_dsv4_native"' not in text:
    idx = text.find(needle, text.find('def build_draft_model('))
    if idx == -1:
        raise SystemExit('Could not locate build_draft_model() --from-pretrained branch')
    text = text[:idx] + special + text[idx:]

# 2) Native config has dim, not transformer_layer_config.hidden_size.
old = '    hidden_size = draft_model.config.transformer_layer_config.hidden_size\n'
new = '''    transformer_layer_config = getattr(draft_model.config, "transformer_layer_config", None)
    hidden_size = getattr(transformer_layer_config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(draft_model.config, "dim", None)
    if hidden_size is None:
        hidden_size = getattr(draft_model.config, "hidden_size", None)
    if hidden_size is None:
        raise AttributeError(
            "Could not infer hidden size from draft_model.config; expected "
            "transformer_layer_config.hidden_size, dim, or hidden_size."
        )
'''
if old in text:
    text = text.replace(old, new, 1)

# 3) Help text so --help documents the new type.
text = text.replace(
    'help="Type of speculator model to train (eagle3, dflash, dspark, peagle, mtp)",',
    'help="Type of speculator model to train (eagle3, dflash, dspark, dspark_dsv4_native, peagle, mtp)",',
)

p.write_text(text)
PY

python -m py_compile \
  src/speculators/models/dspark_dsv4_native/config.py \
  src/speculators/models/dspark_dsv4_native/modeling.py \
  src/speculators/models/dspark_dsv4_native/core.py \
  src/speculators/models/dspark_dsv4_native/export_hf.py

echo "Applied native DeepSeek-V4 DSpark patch."
echo "Next: run 'python scripts/train.py --help | grep dspark_dsv4_native'."
