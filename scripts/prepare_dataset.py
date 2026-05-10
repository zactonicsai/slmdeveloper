import json
from pathlib import Path

DATA_DIR = Path("/workspace/data")
EXAMPLES_DIR = DATA_DIR / "java_dto_examples"
DATA_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

examples = [
    {
        "class_name": "CreateUserProfileRequest",
        "instruction": "Create a Java DTO for creating a user profile using Lombok and Jakarta Validation.",
        "response": """
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@Schema(description = "Request DTO for creating a user profile")
public class CreateUserProfileRequest {
    @NotBlank(message = "externalId is required")
    @Size(max = 100)
    @Schema(example = "auth0|123456")
    private String externalId;

    @NotBlank(message = "email is required")
    @Email(message = "email must be valid")
    @Schema(example = "user@example.com")
    private String email;

    @NotBlank(message = "displayName is required")
    @Size(min = 2, max = 120)
    @Schema(example = "Ava Johnson")
    private String displayName;
}
""".strip(),
    },
    {
        "class_name": "CreateOrderRequest",
        "instruction": "Create a Java DTO for an order request with idempotency key and metadata.",
        "response": """
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.PastOrPresent;
import jakarta.validation.constraints.Size;
import lombok.Builder;
import lombok.Data;

import java.time.LocalDate;
import java.util.Map;

@Data
@Builder
@Schema(description = "Request DTO for creating an order")
public class CreateOrderRequest {
    @NotBlank(message = "userId is required")
    @Size(max = 255)
    @Schema(example = "user-123")
    private String userId;

    @NotBlank(message = "status is required")
    @Size(max = 32)
    @Schema(example = "PENDING")
    private String status;

    @Email(message = "contactEmail must be valid")
    @Schema(example = "buyer@example.com")
    private String contactEmail;

    @PastOrPresent(message = "orderDate cannot be in the future")
    @Schema(example = "2026-05-10")
    private LocalDate orderDate;

    @Size(max = 255)
    @Schema(example = "checkout-abc-123")
    private String idempotencyKey;

    @Schema(description = "Flexible order metadata")
    private Map<String, Object> metadata;
}
""".strip(),
    },
    {
        "class_name": "DocumentIngestionRequest",
        "instruction": "Create a Java DTO for document ingestion with validation and Swagger examples.",
        "response": """
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@Schema(description = "Request DTO for document ingestion")
public class DocumentIngestionRequest {
    @NotBlank(message = "fileName is required")
    @Size(max = 255)
    @Schema(example = "incident-report.pdf")
    private String fileName;

    @NotBlank(message = "contentType is required")
    @Pattern(regexp = "application/pdf|image/png|image/jpeg|text/plain",
             message = "Unsupported content type")
    @Schema(example = "application/pdf")
    private String contentType;

    @NotBlank(message = "s3Bucket is required")
    @Schema(example = "file-uploads")
    private String s3Bucket;

    @NotBlank(message = "s3Key is required")
    @Schema(example = "uploads/2026/incident-report.pdf")
    private String s3Key;

    @Schema(example = "32.3792")
    private Double latitude;

    @Schema(example = "-86.3077")
    private Double longitude;
}
""".strip(),
    },
    {
        "class_name": "DocumentSearchRequest",
        "instruction": "Create a Java DTO for search filters with pagination validation.",
        "response": """
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Size;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@Schema(description = "Search filter DTO for document search")
public class DocumentSearchRequest {
    @Size(max = 200)
    @Schema(example = "contract renewal")
    private String query;

    @Size(max = 64)
    @Schema(example = "LEGAL")
    private String documentType;

    @Min(0)
    @Schema(example = "0")
    private int page;

    @Min(1)
    @Max(100)
    @Schema(example = "25")
    private int size;

    @Schema(example = "createdAt,desc")
    private String sort;
}
""".strip(),
    },
    {
        "class_name": "ApiErrorResponse",
        "instruction": "Create a Java response DTO using Lombok Builder for a safe API error.",
        "response": """
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Data;

import java.time.Instant;
import java.util.Map;

@Data
@Builder
@Schema(description = "Standard API error response")
public class ApiErrorResponse {
    @Schema(example = "2026-05-10T18:30:00Z")
    private Instant timestamp;

    @Schema(example = "404")
    private int status;

    @Schema(example = "S3_BUCKET_NOT_FOUND")
    private String errorCode;

    @Schema(example = "The requested S3 bucket was not found")
    private String message;

    @Schema(example = "/api/documents/upload")
    private String path;

    @Schema(description = "Optional safe error details")
    private Map<String, Object> details;
}
""".strip(),
    },
    {
        "class_name": "PatientIntakeRequest",
        "instruction": "Create a HIPAA-conscious patient intake DTO with Jakarta Validation and no sensitive logging fields.",
        "response": """
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@Schema(description = "Request DTO for patient intake. Avoid logging this object because it may contain PHI.")
public class PatientIntakeRequest {
    @NotBlank(message = "patientExternalId is required")
    @Size(max = 100)
    @Schema(example = "patient-789")
    private String patientExternalId;

    @NotBlank(message = "firstName is required")
    @Size(max = 100)
    @Schema(example = "Jordan")
    private String firstName;

    @NotBlank(message = "lastName is required")
    @Size(max = 100)
    @Schema(example = "Rivera")
    private String lastName;

    @Email(message = "contactEmail must be valid")
    @Schema(example = "patient@example.com")
    private String contactEmail;

    @Pattern(regexp = "LOW|MEDIUM|HIGH", message = "riskLevel must be LOW, MEDIUM, or HIGH")
    @Schema(example = "LOW")
    private String riskLevel;
}
""".strip(),
    },
]

prompt_templates = [
    "### Instruction:\n{instruction}\n\n### Response:\n{response}",
    "User request: {instruction}\nJava DTO answer:\n{response}",
    "Generate production-ready Java 21 DTO code.\nTask: {instruction}\nAnswer:\n{response}",
    "You are a Java Spring Boot DTO generator.\nRequest: {instruction}\nCode:\n{response}",
]

rows = []
for ex in examples:
    for template in prompt_templates:
        rows.append({"text": template.format(**ex)})
    (EXAMPLES_DIR / f"{ex['class_name']}.java").write_text(ex["response"] + "\n", encoding="utf-8")

output = DATA_DIR / "java_dto_dataset.jsonl"
with output.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote {len(rows)} training rows to {output}")
print(f"Wrote Java examples to {EXAMPLES_DIR}")
