#!/usr/bin/env python3

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def clean_log(text):
    return ANSI_RE.sub("", text).replace("\r", "")


def last_match(pattern, text, flags=0):
    matches = list(re.finditer(pattern, text, flags))
    return matches[-1] if matches else None


def yes_no(value):
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def gib(mib):
    return mib / 1024.0


def print_table(headers, rows):
    if not rows:
        return
    widths = [len(header) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))
    print("  " + "  ".join(str(value).ljust(widths[i]) for i, value in enumerate(headers)))
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print("  " + "  ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))


def parse_context(text):
    values = {}
    patterns = {
        "slots": r"initializing, n_slots = (\d+), n_ctx_slot = (\d+), kv_unified = '([^']+)'",
        "ctx": r"llama_context: n_ctx\s*=\s*(\d+)",
        "ctx_seq": r"llama_context: n_ctx_seq\s*=\s*(\d+)",
        "batch": r"llama_context: n_batch\s*=\s*(\d+)",
        "ubatch": r"llama_context: n_ubatch\s*=\s*(\d+)",
        "flash": r"llama_context: flash_attn\s*=\s*([^\n]+)",
    }
    for key, pattern in patterns.items():
        match = last_match(pattern, text)
        if match:
            values[key] = match.groups()
    return values


def parse_features(text, context):
    flash = None
    if "flash" in context:
        flash = context["flash"][0].strip().lower() in ("enabled", "true", "1", "on")

    slots = None
    kv_unified = None
    if "slots" in context:
        slots = int(context["slots"][0])
        kv_unified = context["slots"][2].lower() == "true"

    dense_fallback = "running DENSE attention" in text
    is_minimax_m3 = "MiniMax-M3" in text or "minimax-m3" in text.lower()
    msa = None
    if dense_fallback:
        msa = False
    elif is_minimax_m3 and flash is True and slots is not None:
        msa = slots == 1 or not kv_unified

    rot_k = last_match(r"attn_rot_k = ([01])", text)
    rot_v = last_match(r"attn_rot_v = ([01])", text)
    return {
        "Flash attention": yes_no(flash),
        "MiniMax MSA": yes_no(msa),
        "KV rotation K": yes_no(bool(int(rot_k.group(1)))) if rot_k else "unknown",
        "KV rotation V": yes_no(bool(int(rot_v.group(1)))) if rot_v else "unknown",
        "Prompt cache": yes_no("prompt cache is enabled" in text),
        "Cache reuse": "disabled (unsupported)" if "cache_reuse is not supported" in text else "not reported",
        "Multimodal projector": yes_no("loaded multimodal model" in text),
        "Speculative decoding": "disabled" if "no implementations specified for speculative decoding" in text else "not reported",
    }


def parse_fit(text):
    projected = last_match(
        r"projected to use (\d+) MiB of device memory vs\. (\d+) MiB of free device memory", text)
    deficit = last_match(r"need to use (\d+) MiB less in total", text)
    success = last_match(r"successfully fit params to free device memory", text)
    failed = last_match(r"failed to fit params to free device memory: ([^\n]+)", text)
    result = []
    if projected:
        used = int(projected.group(1))
        free = int(projected.group(2))
        result.append(("Projected device use", f"{used} MiB ({gib(used):.2f} GiB)"))
        result.append(("Free device memory", f"{free} MiB ({gib(free):.2f} GiB)"))
    if deficit:
        value = int(deficit.group(1))
        result.append(("Fit deficit incl. margins", f"{value} MiB ({gib(value):.2f} GiB)"))
    if failed and (not success or failed.start() > success.start()):
        result.append(("Fit result", "FAILED: " + failed.group(1).strip()))
    elif success:
        result.append(("Fit result", "success"))
    else:
        result.append(("Fit result", "not reported"))
    return result


def actual_load_section(text):
    fit = last_match(r"successfully fit params to free device memory", text)
    if fit:
        return text[fit.end():]
    loading = last_match(r"loading model tensors, this can take a while", text)
    return text[loading.start():] if loading else text


def parse_buffers(text):
    section = actual_load_section(text)
    buffers = defaultdict(lambda: defaultdict(float))
    pattern = re.compile(
        r":\s+([A-Za-z0-9_.-]+)\s+(model|KV|compute|output|RS|LoRA) buffer size (?:=|is)\s+([0-9.]+) MiB")
    for match in pattern.finditer(section):
        device, category, value = match.groups()
        buffers[device][category.lower()] += float(value)
    return buffers


def parse_last_memory_table(text):
    starts = [match.start() for match in re.finditer(r"\| memory breakdown \[MiB\]", text)]
    if not starts:
        return []
    lines = text[starts[-1]:].splitlines()
    rows = []
    for line in lines[2:]:
        if "|" not in line:
            if rows:
                break
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < 3 or not fields[1].startswith("-"):
            continue
        name = fields[1].lstrip("- ")
        numbers = [int(value) for value in re.findall(r"-?\d+", fields[2])]
        if "Host" == name and len(numbers) >= 4:
            rows.append((name, "-", "-", numbers[0], numbers[1], numbers[2], numbers[3], "-"))
        elif len(numbers) >= 7:
            rows.append((name, numbers[0], numbers[1], numbers[2], numbers[3], numbers[4], numbers[5], numbers[6]))
    return rows


def parse_offload(text):
    section = actual_load_section(text)
    offloaded = last_match(r"offloaded (\d+)/(\d+) layers to GPU", section)
    repeating = last_match(r"offloading (\d+) repeating layers to GPU", section)
    output = "offloading output layer to GPU" in section
    result = []
    if offloaded:
        result.append(("GPU layers", f"{offloaded.group(1)}/{offloaded.group(2)}"))
    if repeating:
        result.append(("Repeating layers on GPU", repeating.group(1)))
    result.append(("Output layer on GPU", yes_no(output)))
    return result


def parse_tensor_counts(text):
    total = last_match(r"loaded meta data with \d+ key-value pairs and (\d+) tensors", text)
    overrides = defaultdict(int)
    pattern = re.compile(r"tensor \S+ \([^\n]*?\) buffer type overridden to ([A-Za-z0-9_.-]+)")
    for match in pattern.finditer(actual_load_section(text)):
        overrides[match.group(1)] += 1
    return int(total.group(1)) if total else None, overrides


def main():
    parser = argparse.ArgumentParser(description="Summarize llama-server configuration and memory placement from a log")
    parser.add_argument("log", help="log file path, or - for stdin")
    args = parser.parse_args()

    if args.log == "-":
        text = sys.stdin.read()
        source = "stdin"
    else:
        path = Path(args.log)
        if not path.is_file():
            parser.error(f"log file does not exist: {path}")
        text = path.read_text(errors="replace")
        source = str(path)
    text = clean_log(text)

    if not text.strip():
        print(f"Log: {source}")
        print("Error: log is empty; make sure llama-server stderr and stdout are redirected to this file.")
        return 2

    print(f"Log: {source}")
    context = parse_context(text)

    print("\nFeatures")
    for name, value in parse_features(text, context).items():
        print(f"  {name}: {value}")

    print("\nContext")
    if "slots" in context:
        slots, ctx_slot, unified = context["slots"]
        print(f"  slots: {slots}")
        print(f"  context per slot: {ctx_slot}")
        print(f"  unified KV: {unified}")
    for key, label in (("ctx", "context total"), ("ctx_seq", "context per sequence"),
                       ("batch", "logical batch"), ("ubatch", "physical ubatch"), ("flash", "flash attention")):
        if key in context:
            print(f"  {label}: {context[key][0].strip()}")

    print("\nFit")
    for name, value in parse_fit(text):
        print(f"  {name}: {value}")

    print("\nLayer placement")
    for name, value in parse_offload(text):
        print(f"  {name}: {value}")

    buffers = parse_buffers(text)
    print("\nFinal allocated buffers after fit/load")
    if buffers:
        categories = ("model", "kv", "compute", "output", "rs", "lora")
        rows = []
        totals = defaultdict(float)
        for device, values in buffers.items():
            total = sum(values.values())
            for category, value in values.items():
                totals[category] += value
            totals["total"] += total
            rows.append((device, *(f"{values[name]:.2f}" for name in categories), f"{total:.2f}"))
        rows.append(("TOTAL", *(f"{totals[name]:.2f}" for name in categories), f"{totals['total']:.2f}"))
        print_table(("device", "model", "KV", "compute", "output", "RS", "LoRA", "total MiB"), rows)
        def is_host_buffer(device):
            name = device.lower()
            return name.startswith(("cpu", "host")) or name.endswith("_host")

        ram = sum(sum(values.values()) for device, values in buffers.items() if is_host_buffer(device))
        vram = totals["total"] - ram
        print(f"  RAM buffers:  {ram:.2f} MiB ({gib(ram):.2f} GiB)")
        print(f"  VRAM buffers: {vram:.2f} MiB ({gib(vram):.2f} GiB)")
    else:
        print("  No final buffer-size lines found. Use --log-verbosity 3 or higher and capture the complete startup.")

    memory_rows = parse_last_memory_table(text)
    print("\nLatest memory breakdown snapshot")
    if memory_rows:
        print("  Note: during --fit this can be the latest probe, not the accepted final allocation.")
        print_table(("device", "total", "free", "self", "model", "context", "compute", "unaccounted"), memory_rows)
    else:
        print("  No memory breakdown table found.")

    total_tensors, overrides = parse_tensor_counts(text)
    print("\nTensor counts")
    print(f"  tensors in GGUF: {total_tensors if total_tensors is not None else 'unknown'}")
    if overrides:
        print("  explicit tensor buffer overrides found in verbose log:")
        for device, count in sorted(overrides.items()):
            print(f"    {device}: {count}")
    else:
        print("  exact tensors per RAM/VRAM: unavailable in this log")
        print("  use the model-buffer MiB table and GPU layer count above as placement ground truth")


if __name__ == "__main__":
    sys.exit(main())
