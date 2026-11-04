"""One real frame per registered candidate, through the local router only."""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

from PIL import Image
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from skell_e_router import classify, RouterError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new output path; previous evidence is never overwritten.")
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("Read TYPESAFE_API_KEY from the agent vault into this process first.")

    # Discovery is read-only; every inference request below uses classify().
    response = requests.get("https://api.typesafe.ai/v1/models",
                            headers={"Authorization": f"Bearer {key}"},
                            timeout=20, allow_redirects=False)
    if response.status_code != 200:
        raise SystemExit(f"Model discovery failed with HTTP {response.status_code}.")
    discovered = response.json()
    with Path("D:/solar-sailer/labeling/dataset/index.csv").open(encoding="utf-8-sig", newline="") as source:
        frame = next(csv.DictReader(source))
    with Image.open(frame["localFrameJpg"]) as image:
        image = image.convert("RGB")
        image.thumbnail((1024, 1024))
        dimensions = image.size
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    data = buffer.getvalue()
    placeholder = str(uuid4())
    assets = {placeholder: {"type": "image", "format": "jpg",
                             "data_base64": base64.b64encode(data).decode("ascii")}}
    questions = {"hand": {"type": "noul", "instructions": "Does `frame` show a human hand?"}}
    record = {"phase": "smoke_only", "models": discovered, "frame_id": frame["frameId"],
              "dimensions": dimensions, "image_sha256": hashlib.sha256(data).hexdigest(),
              "state": {"frame": placeholder}, "questions": questions,
              "expected_answer": True, "results": []}

    # At most two logical requests, each with the router's three-attempt limit.
    # Published text rates do not establish image pricing. Never launch a scored
    # sweep from this script; review these results and pricing before proceeding.
    for model in ("jev", "jev-latest"):
        started = time.perf_counter()
        try:
            answer = classify(model, record["state"], questions, assets=assets,
                              config={"typesafe_api_key": key}, timeout=20)
            result = {"requested_model": model, "model": answer.model, "http_status": 200,
                      "answers": answer.answers, "input_tokens": answer.input_tokens,
                      "output_tokens": answer.output_tokens, "cost": answer.cost}
        except RouterError as error:
            result = {"requested_model": model, "error_code": error.code,
                      "error": str(error), "details": error.details}
        result["duration_seconds"] = time.perf_counter() - started
        record["results"].append(result)
        encoded = json.dumps(record, indent=2)
        if key in encoded or assets[placeholder]["data_base64"] in encoded:
            raise SystemExit("Sensitive output detected; evidence was not written.")
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
