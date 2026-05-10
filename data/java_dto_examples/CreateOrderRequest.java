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
