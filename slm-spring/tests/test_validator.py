"""Tests for validator.validate_java.

These run without ChromaDB or any model — they only need a JDK in $PATH.
"""

import shutil
import pytest

from src.validator import validate_java


pytestmark = pytest.mark.skipif(
    not shutil.which("javac"),
    reason="javac not on PATH; install JDK 21 to run validator tests",
)


VALID_DTO = """package com.example.api.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class FooDTO {
    @NotBlank
    private String name;
}
"""


VALID_CONTROLLER = """package com.example.api.controller;

import com.example.api.dto.FooDTO;
import com.example.api.service.FooService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/foos")
@RequiredArgsConstructor
public class FooController {
    private final FooService service;

    @PostMapping
    public ResponseEntity<FooDTO> create(@Valid @RequestBody FooDTO req) {
        return ResponseEntity.ok(req);
    }
}
"""


SYNTAX_ERROR = """package com.example.api.dto;

public class BrokenDTO {
    private String name
    private int age;
}
"""


UNBALANCED_BRACES = """package com.example.api.dto;

public class BadBraces {
    private String name;
"""


NO_CLASS = """package com.example.api.dto;
import java.util.List;
"""


def test_valid_dto_passes():
    result = validate_java(VALID_DTO)
    assert result.ok is True
    assert result.syntax_ok is True
    assert result.classname == "FooDTO"
    assert result.errors == []


def test_valid_controller_passes():
    result = validate_java(VALID_CONTROLLER)
    assert result.ok is True
    assert result.classname == "FooController"


def test_syntax_error_caught():
    result = validate_java(SYNTAX_ERROR)
    assert result.ok is False
    assert result.syntax_ok is False
    assert len(result.errors) > 0


def test_unbalanced_braces_caught():
    result = validate_java(UNBALANCED_BRACES)
    assert result.ok is False
    assert any("brace" in e.lower() or "error" in e.lower() for e in result.errors)


def test_no_class_caught():
    result = validate_java(NO_CLASS)
    assert result.ok is False
    assert result.classname is None


def test_classname_extracted():
    src = "package x; public final class WeirdName { }"
    result = validate_java(src)
    assert result.classname == "WeirdName"
