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
