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
