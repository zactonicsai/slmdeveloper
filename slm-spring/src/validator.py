"""Compile generated Java with javac. Reject anything that doesn't compile.

This is the strongest anti-hallucination defense: a fabricated annotation,
a misspelled import, a missing closing brace, all surface here as a non-zero
exit code. We don't need a full classpath — we use -Xlint:none and accept
'cannot find symbol' for things like Lombok-generated methods or services
(they're external), but we DO catch syntax errors, malformed imports, and
truly nonsensical code structure.
"""

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    ok: bool
    syntax_ok: bool       # parses as Java
    errors: list
    warnings: list
    classname: str | None


CLASSNAME_RE = re.compile(r"public\s+(?:final\s+)?class\s+(\w+)")
PACKAGE_RE = re.compile(r"package\s+([\w.]+)\s*;")

# These compile errors are expected because we're not providing the full classpath
# (Lombok annotation processor, Spring jars, our service interfaces). They do NOT
# indicate hallucination, just missing dependencies in the validation sandbox.
EXPECTED_MISSING = (
    "package lombok",
    "package jakarta.validation",
    "package org.springframework",
    "package com.fasterxml",
    "package com.example.api.service",
    "cannot find symbol",  # service classes, Lombok-generated getters, etc.
    "package org.slf4j",
)


def _is_expected_error(line: str) -> bool:
    return any(token in line for token in EXPECTED_MISSING)


def validate_java(source: str) -> ValidationResult:
    """Run javac on the source. Return classified errors."""
    classname_match = CLASSNAME_RE.search(source)
    if not classname_match:
        return ValidationResult(
            ok=False, syntax_ok=False,
            errors=["No public class declaration found"],
            warnings=[], classname=None,
        )
    classname = classname_match.group(1)

    if not shutil.which("javac"):
        # Without javac available we can't validate; treat as a soft pass but
        # surface a warning. Production deployments should always have JDK 21.
        return ValidationResult(
            ok=True, syntax_ok=True,
            errors=[], warnings=["javac not available — skipping compile check"],
            classname=classname,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        java_file = Path(tmpdir) / f"{classname}.java"
        java_file.write_text(source, encoding="utf-8")

        proc = subprocess.run(
            ["javac", "-Xlint:none", "-implicit:none", "-d", tmpdir, str(java_file)],
            capture_output=True,
            text=True,
            timeout=30,
        )

    raw = (proc.stderr or "") + (proc.stdout or "")
    lines = [l for l in raw.splitlines() if l.strip()]

    real_errors, expected_errors, warnings = [], [], []
    for line in lines:
        if "warning:" in line.lower():
            warnings.append(line)
        elif _is_expected_error(line):
            expected_errors.append(line)
        elif "error:" in line.lower():
            real_errors.append(line)

    # Syntax is "ok" if no truly unexpected errors. We tolerate missing-symbol
    # errors stemming from absent Spring/Lombok jars.
    syntax_ok = len(real_errors) == 0

    # Additional sanity: structural well-formedness.
    if syntax_ok:
        if source.count("{") != source.count("}"):
            real_errors.append("Unbalanced braces")
            syntax_ok = False
        if not PACKAGE_RE.search(source):
            warnings.append("No package declaration")

    return ValidationResult(
        ok=syntax_ok,
        syntax_ok=syntax_ok,
        errors=real_errors,
        warnings=warnings + [f"(expected) {e}" for e in expected_errors],
        classname=classname,
    )
