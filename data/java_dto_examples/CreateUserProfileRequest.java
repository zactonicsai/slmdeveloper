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
