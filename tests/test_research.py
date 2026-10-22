"""Parser, portability, and training-guard tests; optional neural checks use random weights."""

import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_utils import (fact_numbers, legacy_extract_numbers, numeric_metrics,
                            parse_fact_string, parse_nutrition5k_csv,
                            read_indexed_csv, template_index)

FACT = "ingredients: rice | mass: 100 g | calories: 120 kcal | protein: 3 g | fat: 2 g | carbohydrates: 20 g"


def notebook_nodes(names):
    notebook = json.loads((ROOT / "notebooks/food_nutrition_captioning.ipynb").read_text(encoding="utf-8"))
    nodes = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            nodes.extend(node for node in tree.body if getattr(node, "name", None) in names)
    if {node.name for node in nodes} != set(names):
        raise AssertionError("Requested notebook definitions missing")
    return compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / "notebooks/food_nutrition_captioning.ipynb"), "exec")


class ResearchTests(unittest.TestCase):
    def test_notebook_startup_from_root_and_notebooks_directory(self):
        notebook = json.loads((ROOT / 'notebooks/food_nutrition_captioning.ipynb').read_text(encoding='utf-8'))
        imports = ''.join(notebook['cells'][2]['source'])
        bootstrap = imports[imports.index('REPO_ROOT = next('):imports.index('from research_utils import')]
        config = ''.join(notebook['cells'][3]['source'])
        previous_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fixture = Path(tmp).resolve()
                (fixture / 'research_utils.py').write_text('', encoding='utf-8')
                (fixture / 'notebooks').mkdir()
                for name in ('imagery/realsense_overhead', 'metadata', 'dish_ids/splits'):
                    (fixture / 'data/nutrition5k_dataset' / name).mkdir(parents=True)
                for cwd in (fixture, fixture / 'notebooks'):
                    with self.subTest(cwd=cwd), patch.dict(os.environ, {}, clear=True), \
                         patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stdout(io.StringIO()):
                        os.chdir(cwd)
                        namespace = dict(Path=Path, os=os, sys=sys, device=SimpleNamespace(type='cpu'))
                        exec(bootstrap, namespace)
                        exec(config, namespace)
                        self.assertEqual(namespace['REPO_ROOT'], fixture)
                        self.assertEqual(namespace['BASE_DIR'], fixture / 'runs/portfolio')
                        self.assertFalse(namespace['TRAIN_MODELS'])
                        self.assertFalse(namespace['FORCE_RETRAIN'])
                        self.assertFalse(namespace['USE_AMP'])
                        self.assertTrue(namespace['MODELS_DIR'].is_dir())
                        with self.assertRaises(RuntimeError):
                            namespace['require_training']()
                        os.chdir(previous_cwd)
        finally:
            os.chdir(previous_cwd)

    def test_fact_zero_is_not_missing(self):
        self.assertEqual(parse_fact_string(FACT.replace("fat: 2", "fat: 0"))["fat"], 0)

    def test_fact_missing_rejected(self):
        with self.assertRaises(ValueError):
            parse_fact_string(FACT.split(" | fat:")[0])

    def test_fact_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            parse_fact_string(FACT + " | mass: 2 g")

    def test_fact_invalid_units_rejected(self):
        with self.assertRaises(ValueError):
            parse_fact_string(FACT.replace("120 kcal", "120 g"))

    def test_fact_negative_rejected(self):
        with self.assertRaises(ValueError):
            parse_fact_string(FACT.replace("120 kcal", "-120 kcal"))

    def test_template_stable_between_processes(self):
        command = [sys.executable, "-c", "from research_utils import template_index; print(template_index('dish_fixture'))"]
        outputs = [subprocess.check_output(command, cwd=ROOT,
                   env={**os.environ, "PYTHONHASHSEED": str(seed)}, text=True).strip()
                   for seed in (1, 999)]
        self.assertEqual(outputs[0], outputs[1])

    def test_template_bounds(self):
        self.assertTrue(all(0 <= template_index(f"dish_{i}", 8) < 8 for i in range(50)))
        with self.assertRaises(ValueError):
            template_index("x", 0)

    def test_metric_tolerance_and_omission_denominator(self):
        truth = {"a": fact_numbers(FACT), "b": fact_numbers(FACT)}
        rows = [{"dish_id": "a", "fact_string": FACT,
                 "generated_caption": "A 100g dish has 121 calories, 3g protein, 2g fat and 20g carbs."},
                {"dish_id": "b", "fact_string": FACT, "generated_caption": "A 100g dish."}]
        result = numeric_metrics(rows, truth)
        self.assertEqual(result["metrics"]["NutAcc_avg"], 60)
        self.assertEqual(result["metrics"]["MAE_calories"], 1)
        self.assertEqual(result["mae_denominators"]["calories"], 1)
        self.assertEqual(result["unparsed_counts"]["calories"], 1)

    def test_metric_empty_rejected(self):
        with self.assertRaises(ValueError):
            numeric_metrics([], {})

    def test_metric_duplicate_dishes_rejected(self):
        row = {"dish_id": "a", "fact_string": FACT, "generated_caption": "A 100g dish."}
        with self.assertRaises(ValueError):
            numeric_metrics([row, row], {"a": fact_numbers(FACT)})

    def test_metric_all_unparsed_is_null_mae(self):
        result = numeric_metrics([{"dish_id": "a", "fact_string": FACT,
                                   "generated_caption": "A dish."}], {"a": fact_numbers(FACT)})
        self.assertIsNone(result["metrics"]["MAE_mass"])
        self.assertEqual(result["metrics"]["NutAcc_avg"], 0)

    def test_legacy_parser_limitation_is_explicit(self):
        # Retained for historical comparability, not endorsed as semantic parsing.
        self.assertEqual(legacy_extract_numbers("Contains 8g protein.")["mass"], 8)

    def test_duplicate_csv_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            path.write_text("dish_id,value\na,1\na,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_indexed_csv(path)

    def test_quoted_ingredient_and_truncated_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.csv"
            path.write_text('dish_a,100,50,2,20,3,ingr_a,"rice, cooked",50,100,2,20,3\n', encoding="utf-8")
            self.assertEqual(parse_nutrition5k_csv(path)["dish_a"]["ingredients"][0]["name"], "rice, cooked")
            path.write_text("dish_a,100,50,2,20,3,ingr_a\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_nutrition5k_csv(path)

    def test_notebook_syntax_outputs_and_training_guard(self):
        notebook = json.loads((ROOT / "notebooks/food_nutrition_captioning.ipynb").read_text(encoding="utf-8"))
        config = None
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                ast.parse(source)
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                if "def require_training" in source:
                    config = source
        self.assertIn("TRAIN_MODELS = False", config)
        namespace = {"TRAIN_MODELS": False}
        exec(notebook_nodes(["require_training"]), namespace)
        with self.assertRaises(RuntimeError):
            namespace["require_training"]()


try:
    import torch
    import torch.nn as nn
    import timm
    from transformers import ViTModel, ViTConfig, T5ForConditionalGeneration, T5Config
    from transformers.modeling_outputs import BaseModelOutput
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


@unittest.skipUnless(ML_AVAILABLE, "Optional model libraries not installed")
class ModelSmokeTests(unittest.TestCase):
    def test_cnn_output_shape(self):
        namespace = dict(nn=nn, torch=torch, timm=timm, N_TARGETS=5, CNN_HIDDEN=256, CNN_DROPOUT=.3)
        exec(notebook_nodes(["NutritionCNN"]), namespace)
        model = namespace["NutritionCNN"](pretrained=False).eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(2, 3, 224, 224)).shape), (2, 5))

    def test_fusion_mask_loss_generation_and_freeze(self):
        vit = ViTModel(ViTConfig(image_size=32, patch_size=16, hidden_size=32,
                               num_hidden_layers=1, num_attention_heads=4, intermediate_size=64))
        t5 = T5ForConditionalGeneration(T5Config(vocab_size=64, d_model=32, d_kv=8, d_ff=64,
                       num_layers=1, num_decoder_layers=1, num_heads=4, decoder_start_token_id=0,
                       pad_token_id=0, eos_token_id=1))
        namespace = dict(torch=torch, nn=nn, ViTModel=ViTModel, T5ForConditionalGeneration=T5ForConditionalGeneration,
                         VIT_ID="unused", T5_ID="unused", BEAM_WIDTH=2, MAX_CAPTION_LEN=12,
                         BaseModelOutput=BaseModelOutput)
        exec(notebook_nodes(["ViTT5Captioner"]), namespace)
        with patch.object(ViTModel, "from_pretrained", return_value=vit), \
             patch.object(T5ForConditionalGeneration, "from_pretrained", return_value=t5):
            model = namespace["ViTT5Captioner"](freeze_vit=True)
        model.train()
        self.assertFalse(model.vit.training)
        self.assertFalse(any(p.requires_grad for p in model.vit.parameters()))
        image = torch.zeros(2, 3, 32, 32)
        tokens = torch.tensor([[2, 3, 0], [4, 5, 6]])
        mask = tokens.ne(0).long()
        encoded, combined_mask = model._encode(image, tokens, mask)
        self.assertEqual(tuple(encoded.shape), (2, 8, 32))
        self.assertEqual(combined_mask[0].tolist(), [1, 1, 0, 1, 1, 1, 1, 1])
        loss = model(image, tokens, mask, labels=torch.tensor([[2, 1], [3, 1]])).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.visual_proj.weight.grad)
        model.eval()
        self.assertEqual(model.generate_caption(image, tokens, mask).shape[0], 2)
        model.use_fact = False
        self.assertEqual(model._encode(image, tokens, mask)[0].shape[1], 5)

    def test_fresh_cnn_training_restores_best_epoch(self):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        with tempfile.TemporaryDirectory() as tmp:
            epoch = [0]
            def train_step(model, *unused):
                epoch[0] += 1
                with torch.no_grad():
                    model.weight.fill_(epoch[0])
                return 1.0
            namespace = dict(torch=torch, nn=nn, json=json, AdamW=AdamW,
                ReduceLROnPlateau=ReduceLROnPlateau, NutritionCNN=lambda: nn.Linear(1, 1, bias=False),
                device=torch.device("cpu"), CNN_LR=.001, CNN_WEIGHT_DECAY=0, CNN_SCHED_PAT=1,
                CNN_MAX_EPOCHS=2, CNN_PATIENCE=1, CNN_CKPT=Path(tmp)/"best.pth", CNN_HIST=Path(tmp)/"hist.json",
                cnn_train_epoch=train_step, cnn_validate=lambda *a: float(epoch[0]),
                cnn_train_loader=[], cnn_val_loader=[], require_training=lambda: None)
            exec(notebook_nodes(["train_cnn_full"]), namespace)
            model = namespace["train_cnn_full"]()
            self.assertEqual(model.weight.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
