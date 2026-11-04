#!/usr/bin/env python3
"""Prepara o Endoscapes2023 para o notebook de treino no Colab.

Gera um único zip (cerca de 1,1 GB) com os 11.090 frames rotulados da base:
  - detecção: 1.933 frames com caixas, convertidas para o formato YOLO;
  - CVS: rótulos C1, C2 e C3 por frame. O notebook deriva deles o rótulo de cada vídeo.

Exemplo:
  python training/prepare_endoscapes.py --dataset-dir "C:\\caminho\\endoscapes"

A base tem licença restrita da PhysioNet. Por isso o script recusa gravar o zip
dentro do projeto, para que nenhum frame acabe indo para o Git.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

SPLITS = ("train", "val", "test")
CRITERIA = ("C1", "C2", "C3")
EXPECTED_CLASSES = ("cystic_plate", "calot_triangle", "cystic_artery", "cystic_duct", "gallbladder", "tool")
FRAME_SIZE = (854, 480)

CVS_THRESHOLD = 0.5  # sobre a média de 3 especialistas; equivale ao consenso por maioria da base

ROOT_NAME = "endoscapes_omnicare"
PROJECT_DIR = Path(__file__).resolve().parent.parent


class DatasetError(Exception):
    """Problema nos dados ou nos argumentos que impede gerar um zip confiável."""


def find_dataset_dir(path: Path) -> Path:
    """Aceita a pasta que contém train/ ou uma pasta acima dela."""
    for candidate in (path, path / "endoscapes"):
        if (candidate / "train" / "annotation_coco.json").is_file():
            return candidate
    raise DatasetError(f"Não encontrei train/annotation_coco.json em {path}")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_frame_name(file_name: str) -> tuple[int, int]:
    video, frame = Path(file_name).stem.split("_")
    return int(video), int(frame)


def load_video_ids(dataset_dir: Path, split: str) -> set[int]:
    text = (dataset_dir / f"{split}_vids.txt").read_text(encoding="utf-8")
    return {int(float(x)) for x in text.replace(",", " ").split()}


def load_class_names(det: dict, split: str) -> list[str]:
    categories = sorted(det["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in categories]
    if tuple(names) != EXPECTED_CLASSES or [c["id"] for c in categories] != list(range(1, 7)):
        raise DatasetError(f"Classes inesperadas em {split}: {categories}")
    return names


def to_yolo_line(annotation: dict) -> str:
    x, y, w, h = annotation["bbox"]
    frame_w, frame_h = FRAME_SIZE
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > frame_w + 1 or y + h > frame_h + 1:
        raise DatasetError(f"Caixa inválida na anotação {annotation['id']}: {annotation['bbox']}")
    x_center = min((x + w / 2) / frame_w, 1.0)
    y_center = min((y + h / 2) / frame_h, 1.0)
    return f"{annotation['category_id'] - 1} {x_center:.6f} {y_center:.6f} {min(w / frame_w, 1.0):.6f} {min(h / frame_h, 1.0):.6f}"


def load_split(dataset_dir: Path, split: str, limit_videos: int | None) -> dict:
    """Lê as anotações de um split, validando-as antes de converter."""
    video_ids = load_video_ids(dataset_dir, split)
    if limit_videos:
        video_ids = set(sorted(video_ids)[:limit_videos])

    det = load_json(dataset_dir / split / "annotation_coco.json")
    cvs = load_json(dataset_dir / split / "annotation_ds_coco.json")
    class_names = load_class_names(det, split)

    # Largura e altura vêm só do annotation_coco.json: o annotation_ds_coco.json
    # traz esses campos invertidos em milhares de frames.
    labels: dict[str, list[str]] = {}
    image_names = {}
    for image in det["images"]:
        if parse_frame_name(image["file_name"])[0] not in video_ids:
            continue
        if (image["width"], image["height"]) != FRAME_SIZE:
            raise DatasetError(f"Dimensões inesperadas em {image['file_name']}")
        labels[image["file_name"]] = []
        image_names[image["id"]] = image["file_name"]
    for annotation in det["annotations"]:
        file_name = image_names.get(annotation["image_id"])
        if file_name:
            labels[file_name].append(to_yolo_line(annotation))

    frames = []
    all_split_videos = load_video_ids(dataset_dir, split)
    for image in cvs["images"]:
        video, frame = parse_frame_name(image["file_name"])
        if video not in all_split_videos:
            raise DatasetError(f"{image['file_name']} não pertence ao split {split}")
        if video not in video_ids:
            continue
        scores = image.get("ds")
        if not scores or len(scores) != 3 or not all(0 <= s <= 1 for s in scores):
            raise DatasetError(f"Rótulo CVS inválido em {image['file_name']}: {scores}")
        frames.append({
            "split": split,
            "file_name": image["file_name"],
            "video_id": video,
            "frame": frame,
            "C1": scores[0],
            "C2": scores[1],
            "C3": scores[2],
            "is_det_keyframe": int(image["file_name"] in labels),
        })

    missing = set(labels) - {f["file_name"] for f in frames}
    if missing:
        raise DatasetError(f"{len(missing)} frames com caixas sem rótulo CVS em {split}")
    return {"class_names": class_names, "labels": labels, "frames": frames}


def check_against_metadata(dataset_dir: Path, frames: list[dict]) -> None:
    """Confere os rótulos do JSON contra o all_metadata.csv."""
    reference = {}
    with (dataset_dir / "all_metadata.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["is_ds_keyframe"] == "True":
                reference[(int(row["vid"]), int(row["frame"]))] = tuple(float(row[c]) for c in CRITERIA)
    divergent = [
        f["file_name"]
        for f in frames
        if (key := (f["video_id"], f["frame"])) not in reference
        or any(abs(f[c] - v) > 1e-3 for c, v in zip(CRITERIA, reference[key]))
    ]
    if divergent:
        raise DatasetError(f"{len(divergent)} frames divergem do all_metadata.csv, ex.: {divergent[:5]}")


def load_checksums(dataset_dir: Path) -> dict[str, str] | None:
    path = dataset_dir.parent / "SHA256SUMS.txt"
    if not path.is_file():
        return None
    checksums = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            digest, _, name = line.strip().partition(" ")
            checksums[name.strip().lstrip("*").replace("\\", "/")] = digest
    return checksums


def read_verified_frame(path: Path, split: str, checksums: dict[str, str] | None) -> bytes:
    data = path.read_bytes()
    if checksums is not None:
        expected = checksums.get(f"endoscapes/{split}/{path.name}")
        if expected is None:
            raise DatasetError(f"{path.name} não consta no SHA256SUMS.txt")
        if hashlib.sha256(data).hexdigest() != expected:
            raise DatasetError(f"Hash divergente em {path}")
    with Image.open(io.BytesIO(data)) as image:
        if image.size != FRAME_SIZE:
            raise DatasetError(f"{path.name} tem tamanho {image.size}, esperado {FRAME_SIZE}")
    return data


def csv_text(rows: list[dict], fields: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def dataset_yaml(class_names: list[str]) -> str:
    # Sem o campo "path": o Ultralytics usa a pasta deste arquivo como base.
    lines = ["train: yolo_train.txt", "val: yolo_val.txt", "test: yolo_test.txt", "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(class_names)]
    return "\n".join(lines) + "\n"


def write_zip(output: Path, dataset_dir: Path, splits: dict, frames: list[dict], manifest: dict, checksums: dict[str, str] | None) -> None:
    partial = output.with_name(output.name + ".part")
    total = len(frames)
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED) as archive:
        def add_text(name: str, text: str) -> None:
            archive.writestr(f"{ROOT_NAME}/{name}", text.encode("utf-8"))

        class_names = splits["train"]["class_names"]
        add_text("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        add_text("dataset.yaml", dataset_yaml(class_names))
        add_text("cvs_labels.csv", csv_text(frames, list(frames[0])))
        for split, content in splits.items():
            names = sorted(content["labels"])
            add_text(f"yolo_{split}.txt", "".join(f"./images/{split}/{n}\n" for n in names))
            for name in names:
                add_text(f"labels/{split}/{Path(name).stem}.txt", "\n".join(content["labels"][name]))

        for done, frame in enumerate(sorted(frames, key=lambda f: (f["split"], f["file_name"])), start=1):
            split, name = frame["split"], frame["file_name"]
            data = read_verified_frame(dataset_dir / split / name, split, checksums)
            archive.writestr(f"{ROOT_NAME}/images/{split}/{name}", data)
            if done % 1000 == 0 or done == total:
                print(f"  {done}/{total} frames")
    os.replace(partial, output)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara o Endoscapes2023 para treino no Colab.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Pasta 'endoscapes' que contém train/, val/ e test/.")
    parser.add_argument("--output", type=Path, help="Zip de saída. Padrão: endoscapes_omnicare.zip ao lado da base.")
    parser.add_argument("--overwrite", action="store_true", help="Substitui o zip de saída, se já existir.")
    parser.add_argument("--skip-sha", action="store_true", help="Não confere os frames contra o SHA256SUMS.txt.")
    parser.add_argument("--limit-videos", type=int, help="Usa só os N primeiros vídeos de cada split (para testes).")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    dataset_dir = find_dataset_dir(args.dataset_dir.resolve())
    output = (args.output or dataset_dir.parent / f"{ROOT_NAME}.zip").resolve()

    if output.is_relative_to(PROJECT_DIR):
        raise DatasetError(f"Grave o zip fora do projeto ({PROJECT_DIR}): a licença não permite publicar os frames.")
    if output.exists() and not args.overwrite:
        raise DatasetError(f"{output} já existe. Use --overwrite para substituir.")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Base: {dataset_dir}")
    splits = {split: load_split(dataset_dir, split, args.limit_videos) for split in SPLITS}
    if {tuple(s["class_names"]) for s in splits.values()} != {EXPECTED_CLASSES}:
        raise DatasetError("As classes diferem entre os splits.")

    frames = [frame for split in SPLITS for frame in splits[split]["frames"]]
    check_against_metadata(dataset_dir, frames)

    checksums = None
    if not args.skip_sha:
        checksums = load_checksums(dataset_dir)
        if checksums is None:
            raise DatasetError("SHA256SUMS.txt não encontrado ao lado da base. Use --skip-sha para continuar sem conferir.")

    manifest = {
        "dataset": "Endoscapes2023",
        "classes": splits["train"]["class_names"],
        "criteria": list(CRITERIA),
        "cvs_threshold": CVS_THRESHOLD,
        "frame_size": list(FRAME_SIZE),
        "limit_videos": args.limit_videos,
        "frames_cvs": dict(Counter(f["split"] for f in frames)),
        "frames_det": {split: len(splits[split]["labels"]) for split in SPLITS},
        "boxes": {split: sum(len(v) for v in splits[split]["labels"].values()) for split in SPLITS},
        "videos": {split: len({f["video_id"] for f in frames if f["split"] == split}) for split in SPLITS},
    }

    print(f"Gravando {output}")
    write_zip(output, dataset_dir, splits, frames, manifest, checksums)

    digest = sha256_file(output)
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")

    print("\nResumo")
    print(f"  Frames com rótulo CVS: {manifest['frames_cvs']}")
    print(f"  Frames com caixas:     {manifest['frames_det']}")
    print(f"  Caixas:                {manifest['boxes']}")
    print(f"  Vídeos:                {manifest['videos']}")
    print(f"  Frames conferidos com SHA256: {'sim' if checksums else 'não'}")
    print(f"  Zip: {output} ({output.stat().st_size / 1024**3:.2f} GB)")
    print(f"  SHA256: {digest}")


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except DatasetError as error:
        print(f"ERRO: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
