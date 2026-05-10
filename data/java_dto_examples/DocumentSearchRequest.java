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
