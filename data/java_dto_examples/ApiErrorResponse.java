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
