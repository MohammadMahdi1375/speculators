#!/usr/bin/env python3
"""Read an exact tuning report/log and print the explicit FUSION setting.

This does not certify validation completion; finish the validation evaluator
and check the log for a parity failure before using the resulting fast mode.
No torch import, GPU access, package changes, or checkpoint mutations.
"""
import argparse
import hashlib
import json
from pathlib import Path


def selected_variant(path, draft=None):
    text = path.read_text()
    if any(marker in text for marker in (
            'PREFIX_PARITY_FAIL', 'PREFIX_COMBINED_GRAPH_CAPTURE_FAILED',
            'Traceback (most recent call last):', 'Engine core initialization failed')):
        raise ValueError('The supplied log contains a validation/capture failure; do not benchmark it.')
    if path.suffix == '.json':
        reports = [json.loads(text)]
    else:
        reports = [json.loads(line.split('PREFIX_FUSION_RESULT ', 1)[1])
                   for line in text.splitlines() if 'PREFIX_FUSION_RESULT ' in line]
    if len(reports) != 1:
        raise ValueError('Expected exactly one fusion result. Use a fresh log or the exact PID-suffixed JSON report.')
    report = reports[0]
    winner = report['selected']
    if winner not in {'paired', 'select', 'local', 'select_softmax', 'local_softmax'}:
        raise ValueError('Unknown selected variant')
    if winner in report.get('rejected', {}):
        raise ValueError('Selected variant appears in rejected list')
    if report.get('tuning_blocks', 0) < 2:
        raise ValueError('Tuning did not finish')
    if draft is not None:
        if not report.get('draft_model') or Path(report['draft_model']).resolve() != draft.resolve():
            raise ValueError('Draft path differs from the tuning run')
        if hashlib.sha256((draft / 'config.json').read_bytes()).hexdigest() != report.get('draft_config_sha256'):
            raise ValueError('Draft config differs from the tuning run')
    return 'off' if winner == 'paired' else winner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--draft', type=Path)
    args = parser.parse_args()
    try:
        print(selected_variant(args.report, args.draft))
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(str(exc))


if __name__ == '__main__':
    main()
